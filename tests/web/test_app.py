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


def test_page_has_the_stepper(client):
    page = client.get("/", headers=HOST)
    assert b'id="tab-steps"' in page.data and b'id="panel-steps"' in page.data and b"stepper.css" in page.data
    for path in ("app/components/stepper.js", "app/components/stepper.css"):
        response = client.get(f"/static/{path}", headers=HOST)
        assert response.status_code == 200, path
        response.close()


def test_formulas_are_rendered_in_bounded_boxes(client):
    """Every KaTeX rendering goes through formula-box.js, whose box is bounded (issue #196)."""
    components = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"
    for name in ("result-view.js", "stepper.js"):
        source = (components / name).read_text(encoding="utf-8")
        assert 'from "./formula-box.js"' in source and "renderMath" not in source, name
    box = (components / "formula-box.js").read_text(encoding="utf-8")
    assert "aria-expanded" in box and "tabindex" in box and "ResizeObserver" in box
    css = (components / "result-view.css").read_text(encoding="utf-8")
    assert "max-height: var(--fb-collapsed)" in css and "max-height: var(--fb-expanded)" in css
    response = client.get("/static/app/components/formula-box.js", headers=HOST)
    assert response.status_code == 200
    response.close()


def test_stepper_transitions_respect_reduced_motion(client):
    """The stepper tweens between steps on requestAnimationFrame and shows only the end state
    with reduced motion (issue #198)."""
    components = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"
    motion = (components / "motion.js").read_text(encoding="utf-8")
    assert "requestAnimationFrame" in motion and "prefers-reduced-motion" in motion
    # interpolated numbers are hidden from assistive technology; the exact value stays readable
    assert '"aria-hidden": "true"' in motion and "tw-exact" in motion
    stepper = (components / "stepper.js").read_text(encoding="utf-8")
    assert 'from "./motion.js"' in stepper and "reducedMotion()" in stepper
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


def test_stepper_instruction_has_a_fixed_box(client):
    """The instruction of a step sits in a reserved formula box, so stepping does not move the view (#206)."""
    components = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"
    stepper = (components / "stepper.js").read_text(encoding="utf-8")
    assert 'className: "st-code"' in stepper and 'h("pre", { class: "rv-pre st-code" }' not in stepper
    css = (components / "stepper.css").read_text(encoding="utf-8")
    assert ".formula-box.st-code { --fb-collapsed:" in css and ".st-instruction-notes { min-height:" in css
    assert client.get("/static/app/components/stepper.css", headers=HOST).status_code == 200


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
