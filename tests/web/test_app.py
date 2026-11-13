"""Tests for the web GUI's JSON API (prodigy web)."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from prodigy.web.app import create_app

HOST = {"Host": "localhost:8080"}

GEOMETRIC = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Pr[c = 0]
"""


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def wait_for(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}", headers=HOST).get_json()
        if status["state"] not in ("starting", "running"):
            return status
        time.sleep(0.1)
    raise AssertionError("analysis did not finish")


def test_page_and_meta(client):
    page = client.get("/", headers=HOST)
    assert page.status_code == 200 and b'id="editor"' in page.data
    meta = client.get("/api/meta", headers=HOST).get_json()
    assert [s["name"] for s in meta["strategies"]][:6] == \
           ["invariant", "unroll", "mass_threshold", "evt_iterate", "evt_invariant", "evt_synthesis"]
    unroll = next(s for s in meta["strategies"] if s["name"] == "unroll")
    assert unroll["params"][0]["name"] == "iterations"
    assert all(p["kind"] != "file" for s in meta["strategies"] for p in s["params"])
    engines = {e["name"]: e for e in meta["engines"]}
    assert list(engines) == ["sympy", "ginac", "symengine"]
    assert all(e["available"] == (e["reason"] is None) for e in engines.values())
    assert sum(e["default"] for e in engines.values()) == 1


def test_engine_is_chosen_in_the_toolbar_not_in_the_setup(client):
    """The engine pop-up sits in the toolbar right before Analyze (#308); Setup no longer has it."""
    page = client.get("/", headers=HOST).get_data(as_text=True)
    toolbar = page[page.index('<header class="toolbar"'):page.index("</header>")]
    assert '<label for="engine-select"' in toolbar and ">Engine</label>" in toolbar
    assert toolbar.index('id="engine-select"') < toolbar.index('id="btn-run"')
    setup = (Path(__file__).parents[2] / "prodigy/web/static/app/setup.js").read_text(encoding="utf-8")
    assert "engineControl" not in setup and "options.engine" not in setup


def test_check_reports_syntax_errors_with_position(client):
    result = client.post("/api/check", json={"source": "nat x;\nwhile (x < 3 { x := x+1 }"}, headers=HOST).get_json()
    assert not result["ok"]
    error = result["diagnostics"][0]
    assert (error["severity"], error["line"], error["column"]) == ("error", 2, 14)
    assert result["outline"] is None


def test_check_outline_and_type_warnings(client):
    result = client.post("/api/check", json={"source": "nat x;\ny := 2\n" + GEOMETRIC.split("\n", 1)[1]},
                         headers=HOST).get_json()
    assert result["ok"]  # type-checker findings are warnings
    assert result["diagnostics"][0]["severity"] == "warning"
    assert result["diagnostics"][0]["line"] == 2
    assert result["outline"]["loops"][0]["line"] == 4


def test_requests_from_other_sites_are_rejected(client):
    assert client.post("/api/check", data="{}", headers={**HOST, "Content-Type": "text/plain"}).status_code == 415
    assert client.post("/api/check", json={"source": ""},
                       headers={**HOST, "Origin": "http://evil.example"}).status_code == 403
    assert client.post("/api/check", json={"source": ""}, headers={"Host": "evil.example"}).status_code == 403


def test_analysis_job_returns_result(client):
    response = client.post("/api/analyze", json={"source": GEOMETRIC, "engine": "sympy",
                                                 "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 4}}}},
                           headers=HOST)
    assert response.status_code == 202
    status = wait_for(client, response.get_json()["job"])
    assert status["state"] == "done"
    result = status["result"]
    assert result["status"] == "ok" and result["bounds"]["kind"] == "lower"
    assert (result["queries"][0]["lower"], result["queries"][0]["upper"]) == ("1/2", "9/16")


def test_yule_simon_example_with_its_invariant(client):
    # the example's comment says the invariant gives upper bounds; the GUI must show them, with Ex[red] ≤ ∞
    path = Path(__file__).parents[2] / "pgfexamples/inference/loopy/hard_distributions/d_finite/yule_simon_urn.ppy"
    invariant = "-log(1 - red) + stop*(-log(1 - red) + log(1 - red)/red + 1)"
    response = client.post("/api/analyze", json={"source": path.read_text(), "syntax": "ppy", "engine": "sympy",
                                                 "decisions": {"0": {"strategy": "evt_invariant",
                                                                     "params": {"invariant": invariant}}}},
                           headers=HOST)
    assert response.status_code == 202
    status = wait_for(client, response.get_json()["job"])
    assert status["state"] == "done"
    result = status["result"]
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "upper"
    assert result["distribution"] is not None
    assert [q["upper"] for q in result["queries"]] == ["1/2", "1/6", "1/12", "\\infty"]
    assert all(q["bound"] == "upper" for q in result["queries"])


def test_invalid_requests_are_rejected_before_running(client):
    response = client.post("/api/analyze", json={"source": GEOMETRIC,
                                                 "decisions": {"0": "unroll:iterations=many"}}, headers=HOST)
    assert response.status_code == 400
    assert "whole number" in response.get_json()["error"]
    files = client.post("/api/analyze", json={"source": GEOMETRIC, "decisions": {
        "0": {"strategy": "invariant", "params": {"invariant_file": "/etc/passwd"}}}}, headers=HOST)
    status = wait_for(client, files.get_json()["job"])
    assert status["result"]["status"] == "needs_decision"


def test_cancel_and_time_limit(client):
    # does not terminate with probability 1/2, so the threshold 0.9 is never reached
    source = "nat x;\nx := 1\nwhile (x > 0) { {x := x - 1} [1/3] {x := x + 1} }\n?Pr[x = 0]\n"
    request = {"source": source, "decisions": {"0": "mass_threshold:threshold=0.9"}}
    job = client.post("/api/analyze", json=request, headers=HOST).get_json()["job"]
    time.sleep(1.5)
    cancelled = client.delete(f"/api/jobs/{job}", headers=HOST).get_json()
    assert cancelled["state"] == "cancelled"
    job = client.post("/api/analyze", json={**request, "time_limit": 2}, headers=HOST).get_json()["job"]
    status = wait_for(client, job)
    assert status["state"] == "timeout" and status["result"]["status"] == "timeout"


def test_examples(client):
    examples = client.get("/api/examples", headers=HOST).get_json()["examples"]
    ids = {e["id"] for e in examples}
    assert "equivalence/loopy/geometric.ppy" in ids
    assert not any("/invariants/" in i for i in ids)
    loaded = client.get("/api/examples/equivalence/loopy/geometric.ppy", headers=HOST).get_json()
    decision = loaded["suggestion"]["decisions"]["0"]
    assert decision["strategy"] == "invariant" and "geometric(1/2)" in decision["params"]["invariant"]
    assert client.get("/api/examples/../../pyproject.toml", headers=HOST).status_code == 404


def test_from_heuristics_with_chosen_heuristics(client):
    meta = client.get("/api/meta", headers=HOST).get_json()
    assert "from_heuristics" in meta["synthesis_strategies"]
    assert meta["template_heuristics"] == ["default", "poly"]
    assert "polynomial" in meta["positivity_heuristics"] and "or" not in meta["positivity_heuristics"]
    source = "nat x;\nx := 1\nwhile (x = 1) { {x := x - 1} [1/2] {skip} }\n"
    rejected = client.post("/api/analyze", json={"source": source, "synthesis_strategy": "from_heuristics",
                                                 "positivity_heuristic": "or"}, headers=HOST)
    assert rejected.status_code == 400 and "positivity heuristic" in rejected.get_json()["error"]
    response = client.post("/api/analyze", json={
        "source": source, "engine": "sympy", "decisions": {"0": {"strategy": "evt_synthesis"}},
        "synthesis_strategy": "from_heuristics", "template_heuristic": "poly",
        "positivity_heuristic": "polynomial"}, headers=HOST)
    assert response.status_code == 202
    status = wait_for(client, response.get_json()["job"])
    assert status["state"] == "done"
    certificate = status["result"]["loops"][0]["certificates"][0]
    assert certificate["synthesis_strategy"] == "FROM_HEURISTICS" and certificate["invariant"] == "2*x + 1"


def test_analysis_records_the_step_trace(client):
    response = client.post("/api/analyze", json={
        "source": GEOMETRIC, "engine": "sympy", "record_steps": True, "watches": ["Pr[c = 0]"], "step_iterations": 2,
        "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 4}}}}, headers=HOST)
    assert response.status_code == 202
    result = wait_for(client, response.get_json()["job"])["result"]
    assert result["status"] == "ok"
    steps = result["steps"]
    assert [(s["line"], s["kind"]) for s in steps] == [(2, "assign"), (3, "while"), (4, "query")]
    loop = steps[1]["loop"]
    assert [i["iteration"] for i in loop["iterates"]][-1] == 4 and len(loop["iterates"]) <= 4
    assert steps[1]["watches"][0]["lower"] == "1/2" and steps[1]["watches"][0]["upper"] == "9/16"
    assert result["trace"]["watches"] == [{"text": "Pr[c = 0]", "kind": "probability"}]


def test_bad_watches_are_rejected_before_running(client):
    for payload in ({"watches": ["Pr[c = "]}, {"watches": "Pr[c = 0]"}, {"watches": ["Pr[c = 0]"] * 9},
                    {"step_iterations": 0}, {"step_iterations": True}):
        response = client.post("/api/analyze", json={"source": GEOMETRIC, "record_steps": True, **payload},
                               headers=HOST)
        assert response.status_code == 400, payload
    # a name that is not a variable is found once the program is parsed
    response = client.post("/api/analyze", json={
        "source": GEOMETRIC, "record_steps": True, "watches": ["Pr[y = 0]"],
        "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 2}}}}, headers=HOST)
    result = wait_for(client, response.get_json()["job"])["result"]
    assert result["status"] == "error" and "not a variable" in result["error"]["message"]


COMPONENTS = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"


def test_page_has_the_analysis_card_and_no_steps_tab(client):
    """The Steps tab is folded into the result's analysis card (#311): the inspector has Setup and
    Result only, and the stepper's files are gone."""
    page = client.get("/", headers=HOST)
    assert b'id="tab-steps"' not in page.data and b'id="panel-steps"' not in page.data
    assert b'id="tab-setup"' in page.data and b'id="tab-result"' in page.data and b"analysis-card.css" in page.data
    for path in ("app/components/analysis-card.js", "app/components/analysis-card.css", "app/components/timeline.js",
                 "app/components/parts.js"):
        response = client.get(f"/static/{path}", headers=HOST)
        assert response.status_code == 200, path
        response.close()
    for path in ("app/components/stepper.js", "app/components/stepper.css"):
        response = client.get(f"/static/{path}", headers=HOST)
        assert response.status_code == 404, path
        response.close()
    main = (COMPONENTS.parent / "main.js").read_text(encoding="utf-8")
    assert 'TABS = ["setup", "result"]' in main and "stepper" not in main


def test_formulas_are_rendered_in_bounded_boxes(client):
    """Every KaTeX rendering goes through formula-box.js, whose box is bounded (issue #196)."""
    components = COMPONENTS
    for name in ("analysis-card.js", "parts.js"):
        source = (components / name).read_text(encoding="utf-8")
        assert 'from "./formula-box.js"' in source and "renderMath" not in source, name
    assert "renderMath" not in (components / "result-view.js").read_text(encoding="utf-8")
    box = (components / "formula-box.js").read_text(encoding="utf-8")
    assert "aria-expanded" in box and "tabindex" in box and "ResizeObserver" in box
    css = (components / "result-view.css").read_text(encoding="utf-8")
    assert "max-height: var(--fb-collapsed)" in css and "max-height: var(--fb-expanded)" in css
    response = client.get("/static/app/components/formula-box.js", headers=HOST)
    assert response.status_code == 200
    response.close()


def test_timeline_respects_reduced_motion_and_never_plays_by_itself(client):
    """Moving along the timeline tweens the bars on requestAnimationFrame, and only jumps with
    reduced motion (#198, #311); playing starts only from the Play button (no autoplay)."""
    motion = (COMPONENTS / "motion.js").read_text(encoding="utf-8")
    assert "requestAnimationFrame" in motion and "prefers-reduced-motion" in motion
    card = (COMPONENTS / "analysis-card.js").read_text(encoding="utf-8")
    assert 'from "./motion.js"' in card and "reducedMotion()" in card
    timeline = (COMPONENTS / "timeline.js").read_text(encoding="utf-8")
    # the only timer of the timeline is started by a click on Play
    assert timeline.count("setInterval(") == 1
    assert timeline.index('play.addEventListener("click"') < timeline.index("setInterval(")
    css = (COMPONENTS / "analysis-card.css").read_text(encoding="utf-8")
    assert "@media (prefers-reduced-motion: reduce)" in css
    response = client.get("/static/app/components/motion.js", headers=HOST)
    assert response.status_code == 200
    response.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_log_axis_of_the_bar_charts(tmp_path):
    """The log scale of the bar charts (issue #218): whole decades, the smallest value above the
    baseline, at most 12 decades, no axis without a positive value."""
    charts = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components" / "charts.js"
    script = tmp_path / "check.mjs"
    script.write_text(
        f"const c = await import({str(charts.as_uri())!r});\n"
        "console.log(JSON.stringify([c.logAxis([0.5, 0.25, 0.001, 0]), c.logAxis([0, 0]),"
        " c.logAxis([1e-300, 0.9]), c.logAxis([1e-5, 0.5]), c.powerOfTen(-12), c.powerOfTen(0)]));\n",
        encoding="utf-8")
    output = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(output) == [
        {"lo": -4, "hi": 0, "ticks": [0, -1, -2, -3, -4]},
        None,
        {"lo": -12, "hi": 0, "ticks": [0, -3, -6, -9, -12]},
        {"lo": -6, "hi": 0, "ticks": [0, -2, -4, -6]},
        "10⁻¹²", "1"]


def test_timeline_does_not_move_while_stepping():
    """The parts of the analysis card above the timeline keep their largest height while the
    timeline moves, so that its buttons stay under the pointer (#206, #311)."""
    card = (COMPONENTS / "analysis-card.js").read_text(encoding="utf-8")
    assert "const reserve = (reset)" in card and "part.style.minHeight" in card and "reserve(reset);" in card


TIMELINE_CHECK = """
const t = await import(%(module)r);
const result = JSON.parse(await (await import("node:fs/promises")).readFile(%(path)r, "utf-8"));
const model = t.timelinePositions(result);
const labels = model.positions.map((p) => t.positionLabel(result, p));
const data = model.positions.map((p) => t.positionData(result, p));
console.log(JSON.stringify({
  kinds: model.positions.map((p) => p.kind), segments: model.segments, final: model.final, labels,
  snapshots: data.map((d) => d.snapshot ? d.snapshot.size : null),
  sizes: t.sizeSeries(result, model), times: t.timeSeries(result, model).map((v) => v === null ? null : "t"),
  ofStep: model.positions.map((_, i) => i).filter((i) => model.positions[i].kind === "step").map((i) => model.positions[i].step),
  stepOfWhile: t.positionOfStep(model, 1), missing: t.positionOfStep(model, 99),
  breakNext: t.breakpointTarget(result, model, 0, new Set([4]), 1), breakNone: t.breakpointTarget(result, model, 0, new Set([77]), 1),
  breakBack: t.breakpointTarget(result, model, model.final, new Set([2]), -1),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_trace_to_timeline_mapping(client, tmp_path):
    """The timeline of the analysis card (#311) maps a recorded trace to positions: one per step,
    the recorded iterates of a loop inside its segment before the loop's exit, the final result
    last; sizes and times per position, null where the trace has none (iterates are not timed)."""
    response = client.post("/api/analyze", json={
        "source": GEOMETRIC, "engine": "sympy", "record_steps": True, "step_iterations": 3,
        "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 3}}}}, headers=HOST)
    result = wait_for(client, response.get_json()["job"])["result"]
    assert result["status"] == "ok"
    iterates = len(result["steps"][1]["loop"]["iterates"])
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    script = tmp_path / "check.mjs"
    script.write_text(TIMELINE_CHECK % {"module": (COMPONENTS / "timeline.js").as_uri(), "path": str(path)},
                      encoding="utf-8")
    output = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60,
                                       check=True).stdout)
    n = 1 + iterates + 1 + 1 + 1
    assert output["kinds"] == ["step"] + ["iterate"] * iterates + ["step", "step", "final"]
    assert output["final"] == n - 1
    assert output["segments"] == [{"loop": 0, "step": 1, "from": 1, "to": 1 + iterates}]
    assert output["ofStep"] == [0, 1, 2]
    assert output["labels"][0] == "after step 1 of 3 · line 2"
    assert output["labels"][1].startswith("in step 2 of 3 · loop 0, iteration ") and output["labels"][1].endswith(" of 3")
    assert output["labels"][1 + iterates] == "after step 2 of 3 · line 3 · loop 0 done"
    assert output["labels"][-1] == "final result"
    # every position but the final one has a snapshot of the trace, with its size
    assert all(isinstance(size, int) for size in output["snapshots"][:-1]) and output["snapshots"][-1] is None
    assert output["sizes"][:-1] == output["snapshots"][:-1] and output["sizes"][-1] == len(result["distribution"]["closed_form"])
    assert output["times"] == ["t"] + [None] * iterates + ["t", "t", None]
    assert output["stepOfWhile"] == 1 + iterates and output["missing"] == n - 1
    assert output["breakNext"] == 2 + iterates and output["breakNone"] == n - 1 and output["breakBack"] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_unrecorded_marginals_are_unknown_not_zero(tmp_path):
    """A marginal that the trace did not record is reported with its reason, never as 0 (#202)."""
    script = tmp_path / "check.mjs"
    script.write_text(
        f"const t = await import({(COMPONENTS / 'timeline.js').as_uri()!r});\n"
        "const snap = { marginals: { x: { rows: [{ value: 0, p: 1 }] }, y: { error: 'boom', rows: [] } },"
        " marginals_not_recorded: { z: 'closed form too large' } };\n"
        "console.log(JSON.stringify([t.marginalGap(snap, 'x'), t.marginalGap(snap, 'y'), t.marginalGap(snap, 'z'),"
        " t.marginalGap({ marginals: {}, marginals_omitted: 'time budget used up' }, 'x'), t.marginalGap({}, 'x')]));\n",
        encoding="utf-8")
    output = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(output) == [None, "not available: boom", "closed form too large", "time budget used up",
                                  "not recorded"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_setup_decisions_survive_a_reload_and_follow_a_conversion(tmp_path):
    """The loop decisions are kept while the loops are unknown, e.g. right after a reload, and only
    dropped for loops the program no longer has (issue #194). The invariant programs of a conversion
    to .ppy follow its undo and redo, unless they were edited since (issue #180)."""
    setup = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "setup.js"
    script = tmp_path / "check.mjs"
    script.write_text(
        f"const s = await import({str(setup.as_uri())!r});\n"
        "const inv = (text) => ({ strategy: 'invariant', params: { invariant: text } });\n"
        "const decisions = { '0': inv('nat x;'), '1': { strategy: 'unroll', params: { iterations: 3 } } };\n"
        "s.pruneDecisions(decisions, null);\n"
        "const afterReload = Object.keys(decisions);\n"
        "s.pruneDecisions(decisions, { loops: [{ id: 0 }] });\n"
        "const afterEdit = Object.keys(decisions);\n"
        "const conversion = { before: { '0': 'nat x;' }, after: { '0': 'x: nat;' } };\n"
        "const undo = { before: conversion.after, after: conversion.before };\n"
        "const steps = [s.swapInvariants(decisions, conversion), decisions['0'].params.invariant,\n"
        "  s.swapInvariants(decisions, undo), decisions['0'].params.invariant];\n"
        "decisions['0'].params.invariant = 'nat x; nat c;';\n"
        "steps.push(s.swapInvariants(decisions, conversion), decisions['0'].params.invariant);\n"
        "decisions['0'] = { strategy: 'unroll', params: {} };\n"
        "steps.push(s.swapInvariants(decisions, conversion));\n"
        "console.log(JSON.stringify([afterReload, afterEdit, steps]));\n",
        encoding="utf-8")
    output = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(output) == [["0", "1"], ["0"],
                                  [True, "x: nat;", True, "nat x;", False, "nat x; nat c;", False]]


def test_result_tab_is_an_answer_card_an_analysis_card_and_a_details_line():
    """The Result tab (#288, #311): an answer card with the guarantee once, the analysis card with
    the lenses Distribution, Size, Time and Checks (one control row, one figure, one fact line,
    details as disclosures) and one timeline, and a disclosure line for warnings, console output
    and JSON."""
    components = COMPONENTS
    view = (components / "result-view.js").read_text(encoding="utf-8")
    for part in ("function answerCard(", "function detailsLine(", "function warningNotice(", "Other expected values",
                 "analysisCard(result, options)"):
        assert part in view, part
    # removed: the summary box, the Steps section, the expectations table, the distribution card and
    # the card on how it was computed (both are lenses of the analysis card now)
    for gone in ("function summary(", "function stepsSection(", "rv-table rv-expectations", "Marginal distributions",
                 "function distributionCard(", "function computedCard(", "Step Through", "onOpenSteps"):
        assert gone not in view, gone
    card = (components / "analysis-card.js").read_text(encoding="utf-8")
    assert '[["distribution", "Distribution"], ["size", "Size"], ["time", "Time"], ["checks", "Checks"]]' in card
    for part in ("massSummary(", "positionChart(", "timelineControl(", "Record Steps", "function certificateView(",
                 "Table of all positions", "Queries after every step"):
        assert part in card, part
    joint = (components / "joint-view.js").read_text(encoding="utf-8")
    # the joint view no longer has its own switches or mass box; the result view's control row and mass line serve both
    assert "massBar" not in joint and "modeToggle" not in joint and "export const JOINT_VIEWS" in joint
    charts = (components / "charts.js").read_text(encoding="utf-8")
    assert "export function massSummary(" in charts
    css = (components / "result-view.css").read_text(encoding="utf-8")
    assert ".rv-answer[data-kind=\"exact\"]" in css and ".rv-mass-items" in css and ".jv-mass" not in css


STALENESS_CHECK = """
const s = await import(%(module)r);
const base = {source: "x := 1;", syntax: "ppy", engine: "sympy", decisions: {"0": {strategy: "unroll", params: {iterations: 3}}},
              time_limit: 60};
const reordered = {time_limit: 60, decisions: {"0": {params: {iterations: 3}, strategy: "unroll"}}, engine: "sympy",
                   syntax: "ppy", source: "x := 1;"};
console.log(JSON.stringify([
  s.changeSince(base, {...base}), s.changeSince(base, reordered), s.changeSince(undefined, base),
  s.changeSince(base, {...base, source: "x := 2;"}), s.changeSince(base, {...base, syntax: "pgcl"}),
  s.changeSince(base, {...base, engine: "ginac"}),
  s.changeSince(base, {...base, decisions: {"0": {strategy: "unroll", params: {iterations: 4}}}}),
  s.changeSince(base, {...base, source: "x := 2;", engine: "ginac"}),
  s.changeSince(base, {...base, watches: undefined}),
  s.staleText("program"), s.staleText("setup"), s.staleText("both"),
]));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_result_staleness(tmp_path):
    """A result remembers the request of its analysis; it is outdated as soon as the program or the
    setup differs from it, also when they changed while the analysis ran (#310). The order of keys
    does not matter, and changing back makes it current again."""
    module = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "staleness.js"
    script = tmp_path / "check.mjs"
    script.write_text(STALENESS_CHECK % {"module": module.as_uri()}, encoding="utf-8")
    output = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60,
                                       check=True).stdout)
    assert output == [
        None, None, None, "program", "program", "setup", "setup", "both", None,
        "This result is for an earlier version of the program.",
        "This result is for an earlier version of the setup.",
        "This result is for an earlier version of the program and the setup."]


def test_finished_run_keeps_edits_and_marks_its_result():
    """finish() keeps the program and setup as they are and compares them with the request of the
    run instead of declaring the result current (#310); the outdated result offers Analyze Again."""
    main = (COMPONENTS.parent / "main.js").read_text(encoding="utf-8")
    finish = main[main.index("function finish("):main.index("async function stop(")]
    assert "changeSince(request, buildRequest())" in finish and "__request: request" in finish
    assert "resultStale = false" not in finish and "setSource" not in finish and "state.setup =" not in finish
    assert "state.job = { id: response.job, request," in main
    assert "markStale" not in main  # replaced by refreshStale, which compares both ways
    view = (COMPONENTS / "result-view.js").read_text(encoding="utf-8")
    assert '"Analyze Again"' in view and "staleText(options.change)" in view


def test_meta_groups_the_strategies(client):
    """Every strategy has a group, one plain sentence and its outcome for Setup's grouped pop-up (#324)."""
    meta = client.get("/api/meta", headers=HOST).get_json()
    strategies = {s["name"]: s for s in meta["strategies"]}
    assert strategies["auto"]["group"] == "automatic"
    assert strategies["evt_iterate"]["group"] == "explore" and strategies["evt_iterate"]["outcome"] == "exploration"
    assert {s["group"] for s in strategies.values()} == {"automatic", "exact", "bounds", "explore"}
    assert all(s["summary"] and s["outcome"] for s in strategies.values())


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_setup_groups_defaults_and_progressive_disclosure(client, tmp_path):
    """Setup: the pop-up's sections follow the groups, a new loop gets "auto", and parameters with a
    default sit behind the "Parameters" disclosure while those without one are shown (#324)."""
    meta = client.get("/api/meta", headers=HOST).get_json()
    module = COMPONENTS.parent / "setup.js"
    script = tmp_path / "check.mjs"
    script.write_text(
        f"const s = await import({str(module.as_uri())!r});\n"
        f"const strategies = {json.dumps(meta['strategies'])};\n"
        "const by = Object.fromEntries(strategies.map((x) => [x.name, x]));\n"
        "const names = (ps) => ps.map((p) => p.name);\n"
        "console.log(JSON.stringify({\n"
        "  groups: s.groupStrategies(strategies).map((g) => [g.title, g.members.map((m) => m.name)]),\n"
        "  fallback: s.groupStrategies([{ ...by.unroll, group: 'mystery' }]).map((g) => g.key),\n"
        "  decision: s.defaultDecision(strategies),\n"
        "  withoutAuto: s.defaultDecision(strategies.filter((x) => x.name !== 'auto')).strategy,\n"
        "  unroll: [names(s.splitParams(by.unroll).required), names(s.splitParams(by.unroll).more)],\n"
        "  invariant: [names(s.splitParams(by.evt_invariant).required), names(s.splitParams(by.evt_invariant).more)],\n"
        "}));\n", encoding="utf-8")
    output = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60,
                                       check=True).stdout)
    assert output["groups"] == [
        ["Automatic", ["auto"]],
        ["Exact result", ["invariant", "evt_invariant", "evt_synthesis", "ai_invariant"]],
        ["Bounds", ["unroll", "mass_threshold", "evt_subinvariant", "evt_sandwich"]],
        ["Explore", ["evt_iterate"]],
    ]
    assert output["fallback"] == ["other"]
    assert output["decision"] == {"strategy": "auto", "params": {"iterations": 25}}
    assert output["withoutAuto"] == "unroll"
    assert output["unroll"] == [[], ["iterations"]]
    assert output["invariant"] == [["invariant"], []]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_evt_exploration_candidate(tmp_path):
    """The candidate of an exploration: the rational fit if there is one, else the chosen iterate unless
    its closed form was cut (#324)."""
    module = COMPONENTS / "evt-exploration.js"
    script = tmp_path / "check.mjs"
    fit = {"guess": {"candidate": "1/(1-c/2)"}, "iterates": [{"n": 1, "closed_form": "1", "truncated": False}]}
    plain = {"guess": None, "iterates": [{"n": 1, "closed_form": "x", "truncated": False},
                                         {"n": 9, "closed_form": "x+…", "truncated": True}]}
    result = {"loops": [{"id": 0, "certificates": [{"type": "evt_iteration", "exploration": plain}]},
                        {"id": 1, "certificates": [{"type": "unrolling"}]}]}
    script.write_text(
        f"const e = await import({str(module.as_uri())!r});\n"
        f"const fit = {json.dumps(fit)}, plain = {json.dumps(plain)}, result = {json.dumps(result)};\n"
        "console.log(JSON.stringify([e.candidateFor(fit, 0), e.candidateFor(plain, 0), e.candidateFor(plain, 1),"
        " e.explorations(result).map((x) => x.loop.id)]));\n", encoding="utf-8")
    output = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60,
                                       check=True).stdout)
    assert output == [{"text": "1/(1-c/2)", "kind": "fit"}, {"text": "x", "kind": "iterate", "n": 1}, None, [0]]
