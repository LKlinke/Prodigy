"""Compare mode as a mode of its own: its setup does not come from Analyze, each program's loops take their own
decisions, and the distributions of both programs are plotted (streamed while they are computed, joint grids of
either program)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from prodigy.web.app import create_app
from prodigy.web.jobs import merge_stream
from prodigy.web.requests import compared_result, run_equivalence

from tests.web.test_app import HOST, wait_for

APP = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app"
COMPONENTS = APP / "components"
EXAMPLES = Path(__file__).parents[2] / "pgfexamples"

GEOMETRIC = (EXAMPLES / "equivalence/loopy/geometric.ppy").read_text(encoding="utf-8")
GEOMETRIC_INVARIANT = (EXAMPLES / "equivalence/loopy/invariants/geometric_invariant.ppy").read_text(encoding="utf-8")
TWO = "x: nat;\ny: nat;\n{ x = 1; } [1/2] { x = 2; }\ny = x + 1;\n"
TWO_OTHER = "x: nat;\ny: nat;\n{ x = 1; } [1/3] { x = 2; }\ny = x + 1;\n"
INVARIANT = {"strategy": "invariant", "params": {"invariant": GEOMETRIC_INVARIANT}}


def node(script: str, tmp_path: Path) -> dict:
    path = tmp_path / "check.mjs"
    path.write_text(script, encoding="utf-8")
    return json.loads(subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=60,
                                     check=True).stdout)


# ---- each program's loops take their own decisions -------------------------------------------------

def test_the_loops_of_program_b_take_their_own_decisions_in_the_check():
    """Program B has the same loop as program A. Its loop is unrolled (a lower bound), A's is checked with an
    invariant (an upper bound). Before, the check found B's loop among A's loops by its text and checked it with
    A's invariant."""
    request = {"source": GEOMETRIC, "syntax": "ppy", "other_source": GEOMETRIC, "engine": "sympy",
               "decisions": {"0": INVARIANT},
               "other_decisions": {"0": {"strategy": "unroll", "params": {"iterations": 5}}}}
    equivalence = run_equivalence(request)["equivalence"]
    assert (equivalence["check"]["first"], equivalence["check"]["second"]) == ("upper", "lower")
    assert equivalence["other"]["loops"][0]["strategy"] == "unroll"
    # and the other way round: A's decision does not reach B
    swapped = {**request, "decisions": request["other_decisions"], "other_decisions": request["decisions"]}
    check = run_equivalence(swapped)["equivalence"]["check"]
    assert (check["first"], check["second"]) == ("lower", "upper")


def test_program_b_without_decisions_does_not_borrow_those_of_program_a():
    """Program A's decisions (here a default "*" for every loop) belong to A; B's loop needs its own."""
    request = {"source": GEOMETRIC, "syntax": "ppy", "other_source": GEOMETRIC, "engine": "sympy",
               "decisions": {"*": INVARIANT}}
    equivalence = run_equivalence(request)["equivalence"]
    assert equivalence["verdict"] == "unknown"
    assert equivalence["check"]["answer"] == "failed" and equivalence["check"]["error"]["type"] == "decision"


# ---- plots: streamed coefficients and grids of both programs --------------------------------------------

def test_both_programs_stream_their_coefficients():
    events = []
    result = run_equivalence({"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER, "engine": "sympy",
                              "coefficients": 5}, progress=events.append)
    coefficients = [e for e in events if e.get("kind") == "coefficients"]
    assert {e["program"] for e in coefficients} == {"first", "second"}
    stream: dict = {}
    for event in coefficients:
        merge_stream(stream, event)
    assert stream["variables"] == ["x", "y"] and stream["second"]["variables"] == ["x", "y"]
    assert "program" not in stream["marginals"]["x"]
    assert stream["marginals"]["x"]["rows"] == result["distribution"]["marginals"]["x"]["rows"]
    assert stream["second"]["marginals"]["x"]["rows"] == result["equivalence"]["other"]["distribution"]["marginals"]["x"]["rows"]
    assert stream["marginals"]["x"]["rows"] != stream["second"]["marginals"]["x"]["rows"]


def test_compared_result_picks_the_program():
    result = {"status": "ok", "equivalence": {"other": {"status": "ok", "distribution": {}}}}
    assert compared_result(result, {}) is result and compared_result(result, {"program": "first"}) is result
    assert compared_result(result, {"program": "second"}) is result["equivalence"]["other"]
    for payload in ({"program": "third"}, {"program": None}):
        with pytest.raises(ValueError):
            compared_result(result, payload)
    with pytest.raises(ValueError):
        compared_result({"status": "ok"}, {"program": "second"})


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def test_joint_grids_of_both_programs_of_a_comparison(client):
    response = client.post("/api/equivalence", json={"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER,
                                                     "engine": "sympy"}, headers=HOST)
    job = response.get_json()["job"]
    assert wait_for(client, job)["state"] == "done"
    grids = {}
    for program in ("first", "second"):
        started = client.post(f"/api/jobs/{job}/joint", json={"x": "x", "y": "y", "program": program}, headers=HOST)
        assert started.status_code == 202, started.get_json()
        grids[program] = wait_for(client, started.get_json()["job"])["result"]["joint"]
    cell = lambda grid, x, y: grid["cells"][grid["y_values"].index(y)][grid["x_values"].index(x)]["prob"]
    assert cell(grids["first"], 1, 2) == "1/2" and cell(grids["second"], 1, 2) == "1/3"
    refused = client.post(f"/api/jobs/{job}/joint", json={"x": "x", "y": "y", "program": "third"}, headers=HOST)
    assert refused.status_code == 400


# ---- the setups of the two modes (frontend state) --------------------------------------------------------

SETUP_CHECK = """
const m = await import(%(module)r);
const meta = { engines: [{ name: "sympy", available: true }, { name: "ginac", available: true, default: true }],
  limits: { time_limit: 300 } };
// what Analyze kept: its own strategies, input and options
const analyze = { input: "x**3", decisions: { 0: { strategy: "auto", params: { iterations: 25 } } },
  options: { engine: "sympy", recordSteps: true, coefficients: 7, timeLimit: 999 } };
const store = { [m.SETUP_KEYS.analyze]: analyze };
const restored = {
  analyze: m.restoreSetup(store[m.SETUP_KEYS.analyze], meta, "analyze"),
  compare: m.restoreSetup(store[m.SETUP_KEYS.compare], meta, "compare"),
};
const loop = { id: 0, cond: "x == 1" };
const one = { loops: [loop] };
const free = { loops: [] };
const first = { outline: one, source: "A", decisions: restored.compare.decisions };
const second = { outline: free, source: "B", decisions: {} };
const changed = m.applyCompareDefaults(first, second, [{ name: "invariant" }, { name: "auto" }]);
const chosen = { outline: one, source: "A", decisions: { 0: { strategy: "auto", params: {} } } };
m.applyCompareDefaults(chosen, second, [{ name: "invariant" }, { name: "auto" }]);
// a default follows program B; B gaining a loop makes it a loop to write an invariant for
const followed = { ...first, decisions: structuredClone(first.decisions) };
m.applyCompareDefaults(followed, { outline: free, source: "B2", decisions: {} }, [{ name: "invariant" }]);
const loopyB = { outline: one, source: "B", decisions: {} };
const both = { outline: one, source: "A", decisions: {} };
m.applyCompareDefaults(both, loopyB, [{ name: "invariant" }]);
console.log(JSON.stringify({
  keys: m.SETUP_KEYS, restored, changed,
  compareRequest: m.requestDecisions(first.decisions, one),
  analyzeRequest: m.requestDecisions(restored.analyze.decisions, one),
  chosen: chosen.decisions,
  followed: followed.decisions[0].params.invariant,
  both: [both.decisions[0], loopyB.decisions[0]],
  reasons: [m.defaultReason(first.decisions[0], "A"), m.defaultReason(both.decisions[0], "A"), m.defaultReason(chosen.decisions[0], "A")],
  inputs: [m.setupInput({ input: " x ", inputKind: "default" }, "compare"), m.setupInput({ input: " x ", inputKind: "given" }, "compare"),
    m.setupInput({ input: " x " }, "analyze"), m.setupInput({ input: "  " }, "analyze")],
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_an_analyze_setup_does_not_reach_compare(tmp_path):
    """Analyze and Compare keep their setups apart (#154): what Analyze kept (the strategy "auto", an input, step
    recording, another engine) does not reach a comparison, whose loops get Compare's own defaults."""
    output = node(SETUP_CHECK % {"module": (APP / "mode-setup.js").as_uri()}, tmp_path)
    assert output["keys"] == {"analyze": "setup", "compare": "compareSetup"}
    analyze, compare = output["restored"]["analyze"], output["restored"]["compare"]
    assert analyze["decisions"]["0"]["strategy"] == "auto" and analyze["input"] == "x**3"
    assert analyze["options"]["engine"] == "sympy" and analyze["options"]["recordSteps"] is True
    assert analyze["options"]["timeLimit"] == 300  # capped by the server
    assert compare["input"] == "" and compare["inputKind"] == "default" and "recordSteps" not in compare["options"]
    assert compare["options"]["engine"] == "ginac" and compare["options"]["coefficients"] == 30
    # the loop of A gets the Compare default: program B (loop-free) as its invariant
    assert output["changed"] is True
    assert output["compareRequest"] == {"0": {"strategy": "invariant", "params": {"invariant": "B"}}}
    assert output["analyzeRequest"] == {"0": {"strategy": "auto", "params": {"iterations": 25}}}
    # a strategy somebody chose stays; a default follows program B
    assert output["chosen"] == {"0": {"strategy": "auto", "params": {}}}
    assert output["followed"] == "B2"
    # both programs with loops: an invariant to write for each, never "auto"
    for decision in output["both"]:
        assert decision == {"strategy": "invariant", "params": {"invariant": ""}, "byDefault": True}
    assert "program B is this loop's invariant" in output["reasons"][0]
    assert "write the loop-free program" in output["reasons"][1] and output["reasons"][2] is None
    assert output["inputs"] == [None, "x", "x", None]


def test_main_keeps_one_setup_per_mode():
    """The window reads and writes the setup of the shown mode only; nothing copies Analyze's setup into Compare."""
    main = (APP / "main.js").read_text(encoding="utf-8")
    assert "get setup() { return this.setups[this.mode]; }" in main
    assert 'save("setup", state.setup)' not in main and 'load("setup"' not in main
    assert "restoreSetup(load(SETUP_KEYS[mode], null), state.meta, mode)" in main
    # "Compare this loop" takes only the loop-free program along, not the loop's strategy
    compare_loop = main[main.index("async function compareLoop"):main.index("/** What the verdict card offers")]
    assert "structuredClone(current)" not in compare_loop and "state.setup.decisions = {};" in compare_loop
    setup = (APP / "setup.js").read_text(encoding="utf-8")
    assert "comparing ? null : toggleRow(\"Record each step\"" in setup


# ---- plots of both programs ---------------------------------------------------------------------------

PLOTS_CHECK = """
const m = await import(%(model)r);
const ok = (kind, rows, extra = {}) => ({ status: "ok", bounds: { kind, ...extra }, distribution: { variables: ["x"],
  marginals: { x: { rows, truncated: false } } } });
const exact = ok("exact", [{ value: 0, prob: "1/2", p: 0.5 }, { value: 1, prob: "1/2", p: 0.5 }]);
const upper = ok("upper", [{ value: 0, prob: "1/4", p: 0.25 }, { value: 1, prob: "3/4", p: 0.75 }]);
const lower = ok("lower", [{ value: 0, prob: "1/8", p: 0.125, upper: "1/4", upper_p: 0.25 }], { residual_mass_float: 0.125 });
console.log(JSON.stringify({
  upper: m.compareMarginals(exact, upper, "x").map((r) => r.differs),
  lower: m.compareMarginals(exact, lower, "x").map((r) => r.differs),
  intervals: [m.rowInterval(exact, null), m.rowInterval(upper, { p: 0.25 }), m.rowInterval(lower, null),
    m.rowInterval({ bounds: { kind: "unknown" } }, { p: 0.5 })],
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_differing_values_also_with_bounds(tmp_path):
    """A value is marked where the probabilities provably differ: exact values that differ, or intervals that do
    not overlap (an upper bound below an exact value, a lower bound above it)."""
    output = node(PLOTS_CHECK % {"model": (COMPONENTS / "equivalence-model.js").as_uri()}, tmp_path)
    # exact 1/2 against [0, 1/4] for x = 0 differs; against [0, 3/4] for x = 1 it may not
    assert output["upper"] == [True, None]
    # exact 1/2 against [1/8, 1/4] differs; x = 1 has no row in the lower bound: [0, 1/8] against 1/2
    assert output["lower"] == [True, True]
    assert output["intervals"] == [[0, 0], [0, 0.25], [0, 0.125], None]


CHART_CHECK = """
globalThis.Node = Object;
globalThis.document = {
  createElementNS: (ns, tag) => element(tag),
  createElement: (tag) => element(tag),
  createTextNode: (text) => ({ text }),
};
function element(tag) {
  return { tag, attrs: {}, children: [], classList: { add() {} }, dataset: {},
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; },
    append(...c) { this.children.push(...c); }, appendChild(c) { this.children.push(c); return c; },
    addEventListener() {} };
}
const c = await import(%(charts)r);
const table = { rows: [{ value: 0, prob: "1/2", p: 0.5 }, { value: 1, prob: "1/2", p: 0.5 }] };
const svg = c.marginalChart(table, { variable: "x", width: 300, differ: new Set([1]), peers: [0.9] });
const texts = [];
const walk = (n) => { if (!n || typeof n !== "object") return; if (n.tag === "text") texts.push(n.children.map((x) => x.text ?? "").join(""));
  (n.children || []).forEach(walk); };
walk(svg);
const groups = [];
const find = (n) => { if (!n || typeof n !== "object") return; if (n.tag === "g" && /mc-bar/.test(n.attrs.class || "")) groups.push(n.attrs.class);
  (n.children || []).forEach(find); };
find(svg);
console.log(JSON.stringify({ label: svg.attrs["aria-label"], max: svg.attrs["data-max"], groups, marks: texts.filter((t) => t === "≠").length }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_chart_marks_differing_values_and_shares_the_scale(tmp_path):
    output = node(CHART_CHECK % {"charts": (COMPONENTS / "charts.js").as_uri()}, tmp_path)
    # the other program's 0.9 sets the scale of both charts
    assert output["max"] == "1"
    assert output["groups"] == ["mc-bar mc-value", "mc-bar mc-value mc-differs"]
    # status not by colour alone: the glyph above the bar and the words in the chart's label
    assert output["marks"] == 1 and "1 value differs from the other program (marked ≠)" in output["label"]


def test_compare_view_uses_the_shared_components():
    plots = (COMPONENTS / "compare-plots.js").read_text(encoding="utf-8")
    for used in ('from "./charts.js"', 'from "./joint-view.js"', 'from "./parts.js"', 'from "./analysis-card.js"'):
        assert used in plots
    view = (COMPONENTS / "equivalence-view.js").read_text(encoding="utf-8")
    assert "comparePlots(" in view and "loadJoint: undefined" not in view
