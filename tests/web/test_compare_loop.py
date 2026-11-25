"""A loop against a loop-free program belongs to Compare mode (issue #399): the loop as a program of its
own ("Compare this loop"), the continue flow back to Analyze, and Analyze without the strategy
``invariant`` in its menu."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from prodigy.analysis.session import LoopRegistry, isolate_loop
from prodigy.language import parse_program
from prodigy.web.app import create_app
from prodigy.web.requests import loop_program, run_analysis

from tests.web.test_app import HOST
from tests.web.test_equivalence import compare

APP = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app"

GEOMETRIC = """x: nat;
c: nat;
while (x == 1) {
    { x = 0; } [1/2] { c = c + 1; }
}
query Ex[c];
"""

# code before and after the loop, and a nested loop
SURROUNDED = """x: nat;
c: nat;
x = 1;
c = 2;
if (c == 2) {
    while (x == 1) {
        { x = 0; } [1/2] { c = c + 1; }
    }
} else {
    skip;
}
c = c + 1;
query Ex[c];
"""

CLOSED_FORM = """x: nat;
c: nat;
if (x == 1) {
    c = c + geometric(1/2);
    x = 0;
} else {
    skip;
}
"""


@pytest.fixture(name="client")
def fixture_client():
    return create_app(time_limit=60).test_client()


# ---- the loop as a program of its own -------------------------------------------------------------

def test_a_loop_that_is_the_whole_program_stays_as_it_is():
    isolated = loop_program(GEOMETRIC, 0)
    assert isolated == {"program": GEOMETRIC, "loop": 0, "declarations": "x: nat;\nc: nat;\n", "whole": True}


def test_a_loop_inside_a_program_keeps_only_the_declarations():
    isolated = loop_program(SURROUNDED, 0)
    assert not isolated["whole"] and isolated["loop"] == 0
    program = parse_program(isolated["program"], "ppy")
    # the declarations and the loop: the code before it does not matter, the comparison holds for every input
    assert list(program.variables) == ["x", "c"] and len(program.instructions) == 1
    assert type(program.instructions[0]).__name__ == "WhileInstr"
    assert isolated["declarations"] == "x: nat;\nc: nat;\n"


def test_loops_inside_functions_and_unknown_loops_are_refused():
    source = """def f(a: nat) -> nat {
    while (a > 0) { a = a - 1; }
    return a;
}
x: nat;
x = f(3);
"""
    with pytest.raises(ValueError, match="inside the function f"):
        loop_program(source, 0)
    with pytest.raises(ValueError, match="no loop 3"):
        loop_program(GEOMETRIC, 3)
    with pytest.raises(ValueError, match="does not parse"):
        loop_program("x: nat;\nwhile (", 0)


def test_isolate_loop_keeps_the_loop_id_of_the_second_loop():
    source = GEOMETRIC.replace("query Ex[c];\n", "") + "while (c > 5) { c = c - 1; }\n"
    program = parse_program(source, "ppy")
    registry = LoopRegistry(program)
    second = registry.instruction(1)
    assert second is not None and registry.instruction(2) is None
    isolated = isolate_loop(program, second, source)
    assert isolated is not None and isolated.loop == 0 and not isolated.whole
    assert "c > 5" in isolated.program and "x == 1" not in isolated.program


def test_the_endpoint(client):
    response = client.post("/api/loop-program", json={"source": SURROUNDED, "syntax": "ppy", "loop": 0}, headers=HOST)
    assert response.status_code == 200 and response.get_json()["loop"] == 0
    response = client.post("/api/loop-program", json={"source": SURROUNDED, "syntax": "ppy", "loop": 7}, headers=HOST)
    assert response.status_code == 422 and "no loop 7" in response.get_json()["error"]
    assert client.post("/api/loop-program", json={"source": SURROUNDED}, headers=HOST).status_code == 400
    assert client.post("/api/loop-program", json={"source": SURROUNDED, "loop": True}, headers=HOST).status_code == 400


# ---- the round trip: Compare, then continue in Analyze with B ----------------------------------------

def test_compare_this_loop_then_continue_with_an_upper_bound():
    """The loop of SURROUNDED against its closed form: Park's lemma proves A ⪯ B, and Analyze continues
    after the loop with B (the strategy ``invariant``), an upper bound."""
    isolated = loop_program(SURROUNDED, 0)
    decision = {"strategy": "invariant", "params": {"invariant": CLOSED_FORM}}
    result = compare(isolated["program"], CLOSED_FORM, decisions={str(isolated["loop"]): decision})
    assert result["equivalence"]["relation"] == "first_below_second"
    analysis = run_analysis({"source": SURROUNDED, "syntax": "ppy", "engine": "sympy", "decisions": {"0": decision}})
    assert analysis["status"] == "ok" and analysis["bounds"]["kind"] == "upper"
    # c = 2, plus a geometric number of tails with mean 1, plus 1
    assert analysis["queries"][0]["value"] == "4"


def test_the_backend_strategy_invariant_stays():
    """Analyze no longer offers it, but the server still runs it (saved setups, the continue flow, the CLI)."""
    decision = {"strategy": "invariant", "params": {"invariant": CLOSED_FORM}}
    result = run_analysis({"source": GEOMETRIC, "syntax": "ppy", "engine": "sympy", "decisions": {"0": decision},
                           "input_distribution": "x"})
    assert result["status"] == "ok" and result["bounds"]["kind"] == "upper"


# ---- frontend ---------------------------------------------------------------------------------------

FRONTEND_CHECK = """
const m = await import(%(model)r);
const setup = await import(%(setup)r);
const strategies = [{ name: "auto" }, { name: "invariant" }, { name: "unroll" }];
const decision = { strategy: "auto", params: { iterations: 25 } };
console.log(JSON.stringify({
  offered: setup.offeredStrategies(strategies, new Set(["invariant"])).map((s) => s.name),
  all: setup.offeredStrategies(strategies, null).map((s) => s.name),
  equivalent: m.continueOffer({ verdict: "equivalent" }, { loopsOfA: 1, loopsOfB: 0 }),
  upper: m.continueOffer({ verdict: "unknown", relation: "first_below_second" }, { loopsOfA: 1, loopsOfB: 0 }),
  lower: m.continueOffer({ verdict: "unknown", relation: "second_below_first" }, { loopsOfA: 1, loopsOfB: 0 }),
  noLoop: m.continueOffer({ verdict: "equivalent" }, { loopsOfA: 0, loopsOfB: 0 }),
  twoLoops: m.continueOffer({ verdict: "equivalent" }, { loopsOfA: 2, loopsOfB: 0 }),
  loopyB: m.continueOffer({ verdict: "equivalent" }, { loopsOfA: 1, loopsOfB: 1 }),
  parameters: m.continueOffer({ verdict: "equivalent", constraints: [{ p: "1/2" }] }, { loopsOfA: 1, loopsOfB: 0 }),
  different: m.continueOffer({ verdict: "not_equivalent" }, { loopsOfA: 1, loopsOfB: 0 }),
  undecided: m.continueOffer({ verdict: "unknown" }, { loopsOfA: 1, loopsOfB: 0 }),
  keep: m.continueDecision("exact", "B", decision),
  park: m.continueDecision("upper", "B", decision),
  fallback: m.continueDecision("exact", "B", { strategy: "invariant", params: { invariant: "B" } }),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_frontend_continue_offer(tmp_path):
    script = tmp_path / "check.mjs"
    script.write_text(FRONTEND_CHECK % {"model": (APP / "components" / "equivalence-model.js").as_uri(),
                                        "setup": (APP / "setup.js").as_uri()}, encoding="utf-8")
    output = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60,
                                       check=True).stdout)
    assert output["offered"] == ["auto", "unroll"] and output["all"] == ["auto", "invariant", "unroll"]
    assert output["equivalent"] == {"guarantee": "exact"}
    assert output["upper"] == {"guarantee": "upper"}
    # B ⪯ A makes B a lower bound of the loop, which the strategy invariant cannot give: a note, no button
    assert "lower bound" in output["lower"]["reason"]
    assert output["noLoop"] is None and output["twoLoops"] is None
    assert "loop-free" in output["loopyB"]["reason"] and "parameter values" in output["parameters"]["reason"]
    assert output["different"] is None and output["undecided"] is None
    # exact: the loop keeps the strategy that proved it (invariant alone gives only an upper bound)
    assert output["keep"] == {"strategy": "auto", "params": {"iterations": 25},
                              "compared": {"guarantee": "exact", "program": "B"}}
    assert output["park"] == {"strategy": "invariant", "params": {"invariant": "B"},
                              "compared": {"guarantee": "upper", "program": "B"}}
    assert output["fallback"]["strategy"] == "invariant" and output["fallback"]["compared"]["guarantee"] == "upper"


def test_analyze_hides_the_strategy_invariant_and_compare_offers_the_continue_flow():
    main = (APP / "main.js").read_text(encoding="utf-8")
    assert 'const COMPARE_ONLY = new Set(["invariant"]);' in main
    assert '{ hidden: COMPARE_ONLY, onCompareLoop: compareLoop }' in main
    assert "continueTo: continueTarget()" in main and "api.loopProgram(" in main
    setup = (APP / "setup.js").read_text(encoding="utf-8")
    assert "Compare this loop" in setup and "Open in Compare" in setup and "Choose a strategy" in setup
    view = (APP / "components" / "equivalence-view.js").read_text(encoding="utf-8")
    assert "Continue in Analyze with B as the loop's result" in view
