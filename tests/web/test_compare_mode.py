"""Compare mode as a mode of its own: its setup does not come from Analyze, each program's loops take their own
decisions, and a comparison is only the check for every input: no input, no analysis of either program on one, no
output distributions."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from prodigy.web.app import create_app
from prodigy.analysis.session import AnalysisSession
from prodigy.web import compare as web_compare
from prodigy.web.compare import run_equivalence, validate_equivalence

from tests.web.test_app import HOST, all_app_sources, app_source, wait_for

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

@pytest.mark.slow
def test_the_loops_of_program_b_take_their_own_decisions_in_the_check():
    """Program B has the same loop as program A. Its loop is unrolled (a lower bound), A's is checked with an
    invariant (an upper bound). Before, the check found B's loop among A's loops by its text and checked it with
    A's invariant."""
    request = {"source": GEOMETRIC, "syntax": "ppy", "other_source": GEOMETRIC, "engine": "sympy",
               "decisions": {"0": INVARIANT},
               "other_decisions": {"0": {"strategy": "unroll", "params": {"iterations": 5}}}}
    result = run_equivalence(request)
    equivalence = result["equivalence"]
    assert (equivalence["check"]["first"], equivalence["check"]["second"]) == ("upper", "lower")
    assert result["loops"][0]["strategy"] == "invariant" and result["other_loops"][0]["strategy"] == "unroll"
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


# ---- only the check: no input, no analysis of a single program --------------------------------------

def test_a_comparison_never_analyses_a_single_program(monkeypatch):
    """The verdict comes from the check for every input and the witness search; no program runs as an analysis
    of its own (AnalysisSession.run), and nothing streams coefficients."""
    def refuse(self):
        raise AssertionError("a comparison ran a single-program analysis")
    monkeypatch.setattr(AnalysisSession, "run", refuse)
    events = []
    same = run_equivalence({"source": TWO, "syntax": "ppy", "other_source": TWO, "engine": "sympy"},
                           progress=events.append)
    assert same["equivalence"]["verdict"] == "equivalent"
    differ = run_equivalence({"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER, "engine": "sympy"},
                             progress=events.append)
    assert differ["equivalence"]["verdict"] == "not_equivalent"
    assert not [e for e in events if e.get("kind") == "coefficients"]
    for result in (same, differ):
        assert result["schema"] == "prodigy/equivalence-result@1" and result["status"] == "ok"
        assert "distribution" not in result and "queries" not in result
        assert {"other", "input"}.isdisjoint(result["equivalence"])


def test_a_comparison_takes_no_input():
    request = {"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER, "input_distribution": "x**2"}
    assert any("no initial distribution" in p for p in validate_equivalence(request))
    assert run_equivalence(request)["status"] == "error"
    assert validate_equivalence({**request, "input_distribution": "  "}) == []


def test_the_witness_search_has_a_time_limit(monkeypatch):
    """A witness search that runs out of time leaves the verdict open and says so (#348: witnesses on
    logarithmic closed forms can take minutes)."""
    from prodigy.analysis import tools

    def slow(*_args, **_kwargs):
        import time  # pylint: disable=import-outside-toplevel
        time.sleep(30)
    monkeypatch.setattr(web_compare, "WITNESS_SECONDS", 0.3)
    monkeypatch.setattr(tools, "analyse_on_input", slow)
    equivalence = run_equivalence({"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER,
                                   "engine": "sympy"})["equivalence"]
    assert equivalence["verdict"] == "unknown" and equivalence["witness_search"] == "timeout"
    assert equivalence["witness"] is None and "ran out of time" in equivalence["reason"]


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def test_the_endpoint_refuses_an_input_and_a_comparison_has_no_grids(client):
    refused = client.post("/api/equivalence", json={"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER,
                                                    "input_distribution": "x"}, headers=HOST)
    assert refused.status_code == 400 and "initial distribution" in refused.get_json()["error"]
    response = client.post("/api/equivalence", json={"source": TWO, "syntax": "ppy", "other_source": TWO_OTHER,
                                                     "engine": "sympy"}, headers=HOST)
    job = response.get_json()["job"]
    status = wait_for(client, job)
    assert status["state"] == "done" and not status.get("stream")
    assert status["result"]["equivalence"]["witness"]["input"] == {"x": 0, "y": 0}
    grid = client.post(f"/api/jobs/{job}/joint", json={"x": "x", "y": "y"}, headers=HOST)
    assert grid.status_code == 404


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
  savedInput: m.restoreSetup({ input: "x", inputKind: "given", options: { coefficients: 9, engine: "sympy" } }, meta, "compare"),
  inputs: [m.setupInput({ input: " x ", inputKind: "given" }, "compare"),
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
    # Compare has no input and only the options of the check
    assert "input" not in compare and "inputKind" not in compare
    assert compare["options"] == {"engine": "ginac", "timeLimit": 120}
    assert output["savedInput"] == {"decisions": {}, "options": {"engine": "sympy", "timeLimit": 120}}
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
    assert output["inputs"] == [None, "x", None]


def test_main_keeps_one_setup_per_mode():
    """The window reads and writes the setup of the shown mode only; nothing copies Analyze's setup into Compare."""
    assert "get setup() { return this.setups[this.mode]; }" in app_source("app-state.js")
    everything = all_app_sources()
    assert 'save("setup", state.setup)' not in everything and 'load("setup"' not in everything
    assert "restoreSetup(load(SETUP_KEYS[mode], null), state.meta, mode)" in app_source("main.js")
    # "Compare this loop" takes only the loop-free program along, not the loop's strategy
    modes = app_source("compare-mode.js")
    compare_loop = modes[modes.index("async function compareLoop"):modes.index("/** What the verdict card offers")]
    assert "structuredClone(current)" not in compare_loop and "state.setup.decisions = {};" in compare_loop
    # the request of a comparison has no input and only the options of the check
    running = app_source("run.js")
    build = running[running.index("function buildRequest()"):running.index("async function requestSuggestion")]
    compare_request = build[:build.index("const steps")]
    assert 'state.mode === "compare"' in compare_request and "input_distribution" not in compare_request
    assert "coefficients" not in compare_request and "normalize" not in compare_request
    # Compare's Setup: no input section, loops only for programs that have loops
    setup = (APP / "setup.js").read_text(encoding="utf-8")
    compare_setup = setup[setup.index("function renderCompareSetup"):setup.index("function loopSection")]
    assert "inputKind" not in setup and "Initial distribution" not in compare_setup and "compareInput" not in setup
    assert "hasLoops(ctx.outline" in compare_setup and "hasLoops(ctx.second.outline" in compare_setup



# ---- the view: verdict and witness only ------------------------------------------------------------------

def test_compare_view_shows_only_the_verdict():
    view = (COMPONENTS / "equivalence-view.js").read_text(encoding="utf-8")
    assert "renderResult" not in view and "comparePlots" not in view and "Result of" not in view
    assert "Output distributions" not in view and "witnessTable(" in view
    assert not (COMPONENTS / "compare-plots.js").exists()
    assert "equivalence?.other" not in all_app_sources() and "result?.other_loops" in app_source("panels.js")
