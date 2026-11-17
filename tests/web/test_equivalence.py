"""The comparison of two programs in the GUI (issue #154): sound verdicts, the job and its endpoint,
the example library, and the frontend model of the view."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence import verdict as eq
from prodigy.analysis.loop_strategies import Bound
from prodigy.web.app import create_app
from prodigy.web.requests import run_equivalence, validate_equivalence

from tests.web.test_app import HOST, wait_for

COMPONENTS = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"
EXAMPLES = Path(__file__).parents[2] / "pgfexamples"

BERNOULLI = "x: nat;\n{ x = 0; } [1/3] { x = 1; }\n"
BERNOULLI_TWO_THIRDS = "x: nat;\nx = bernoulli(2/3);\n"
FAIR_COIN = "x: nat;\nx = bernoulli(1/2);\n"

# The pair of the comment in #366: the second-order check needs comparisons on generating functions
# with infinite mass, which are not implemented.
GAMBLER = """x: nat;
n: nat;
while (x > 0 and x < n) {
    { x = x - 1; } [1/2] { x = x + 1; }
}
"""
GAMBLER_CLOSED = """x: nat;
n: nat;
if (n > 0 and x < n) {
    { x = 0; } [x/n] { x = n; }
}
"""


def compare(first, second, **extra):
    request = {"source": first, "syntax": "ppy", "other_source": second, "engine": "sympy", **extra}
    return run_equivalence(request)


# ---- verdicts -------------------------------------------------------------------------------

def test_equal_and_exact_is_equivalent():
    check = eq.CheckOutcome("equal")
    assert eq.decide(check, None)["verdict"] == eq.EQUIVALENT


def test_equality_of_a_park_bound_proves_one_direction_only():
    """A loop handled by an invariant is an upper bound (Park's lemma): equal only if it terminates."""
    verdict = eq.decide(eq.CheckOutcome("equal", first=Bound.UPPER), None)
    assert verdict["verdict"] == eq.UNKNOWN and verdict["relation"] == eq.FIRST_BELOW_SECOND
    assert "terminate almost surely" in verdict["reason"]
    verdict = eq.decide(eq.CheckOutcome("equal", first=Bound.LOWER), None)
    assert verdict["verdict"] == eq.UNKNOWN and verdict["relation"] == eq.SECOND_BELOW_FIRST
    both = eq.decide(eq.CheckOutcome("equal", first=Bound.UPPER, second=Bound.UPPER), None)
    assert both["verdict"] == eq.UNKNOWN and both["relation"] is None


def test_not_equivalent_needs_a_witness():
    """A raw "different" of the solver is not enough; a not implemented never becomes not equivalent."""
    assert eq.decide(eq.CheckOutcome("different", counterexample={"x": 0}), None)["verdict"] == eq.UNKNOWN
    failed = eq.CheckOutcome("failed", error={"type": "NotImplementedError", "message": "comparison on oo"})
    verdict = eq.decide(failed, None)
    assert verdict["verdict"] == eq.UNKNOWN and verdict["reason"].startswith("Not implemented")
    witness = {"state": {"x": 0}, "first": {}, "second": {}}
    assert eq.decide(eq.CheckOutcome("different"), witness)["verdict"] == eq.NOT_EQUIVALENT
    assert eq.decide(eq.CheckOutcome("undecided"), None)["verdict"] == eq.UNKNOWN


def _side(text, variables, kind=Bound.EXACT, residual=0):
    config = ForwardAnalysisConfig()
    with config.activate():
        return eq.Side(config.factory.from_expr(text, *variables), kind, residual)


def test_witness_needs_disjoint_intervals():
    config = ForwardAnalysisConfig()
    with config.activate():
        third = _side("1/3 + 2/3*x", ["x"])
        half = _side("1/2 + 1/2*x", ["x"])
        witness = eq.find_witness(third, half)
        assert witness["state"] == {"x": 0}
        assert (witness["first"]["lower"], witness["second"]["upper"]) == ("1/3", "1/2")
        # a lower bound with residual 3/4 may still reach 1/2 in both states: no witness
        assert eq.find_witness(_side("1/3", ["x"], Bound.LOWER, "3/4"), half) is None
        # with residual 1/4, x = 1 gets at most 1/4 < 1/2: a witness
        assert eq.find_witness(_side("1/3", ["x"], Bound.LOWER, "1/4"), half)["state"] == {"x": 1}
        # a lower bound above the exact value of the other program is one
        assert eq.find_witness(_side("2/3", ["x"], Bound.LOWER, "1/4"), half)["first"]["kind"] == Bound.LOWER
        # without a guarantee there is no witness
        assert eq.find_witness(_side("1/3", ["x"], Bound.UNKNOWN), half) is None


def test_different_variables_are_not_compared():
    from prodigy.pgcl import parse_pgcl
    reason = eq.different_variables(parse_pgcl("x: nat;\n", "ppy"), parse_pgcl("y: nat;\n", "ppy"))
    assert "only the first program declares x" in reason and "only the second program declares y" in reason
    result = compare("x: nat;\nx = 1;\n", "y: nat;\ny = 1;\n")
    assert result["equivalence"]["verdict"] == eq.UNKNOWN
    assert "different variables" in result["equivalence"]["reason"]


# ---- the job ---------------------------------------------------------------------------------

def test_equivalent_programs():
    result = compare(BERNOULLI, BERNOULLI_TWO_THIRDS)
    assert result["schema"] == "prodigy/analysis-result@1" and result["status"] == "ok"
    equivalence = result["equivalence"]
    assert equivalence["verdict"] == "equivalent" and equivalence["witness"] is None
    assert equivalence["check"]["answer"] == "equal"
    assert (equivalence["check"]["first"], equivalence["check"]["second"]) == ("exact", "exact")
    assert equivalence["other"]["schema"] == "prodigy/analysis-result@1"
    assert equivalence["other"]["status"] == "ok"
    assert equivalence["input"] == {"distribution": None, "source": "default"}


def test_programs_that_differ_get_a_witness():
    equivalence = compare(BERNOULLI, FAIR_COIN)["equivalence"]
    assert equivalence["verdict"] == "not_equivalent"
    assert equivalence["check"]["counterexample"] == {"x": 0}
    # both programs are shown on the counterexample of the check
    assert equivalence["input"] == {"distribution": "1", "source": "counterexample"}
    witness = equivalence["witness"]
    assert witness["state"] in ({"x": 0}, {"x": 1})
    assert {witness["first"]["lower"], witness["second"]["lower"]} <= {"1/3", "2/3", "1/2"}
    assert witness["first"]["kind"] == witness["second"]["kind"] == "exact"


def test_a_loop_with_an_invariant_is_one_direction():
    source = (EXAMPLES / "equivalence/loopy/geometric.ppy").read_text(encoding="utf-8")
    invariant = (EXAMPLES / "equivalence/loopy/invariants/geometric_invariant.ppy").read_text(encoding="utf-8")
    equivalence = compare(source, invariant, decisions={"0": {"strategy": "invariant",
                                                              "params": {"invariant": invariant}}})["equivalence"]
    assert equivalence["verdict"] == "unknown" and equivalence["relation"] == "first_below_second"
    assert (equivalence["check"]["first"], equivalence["check"]["second"]) == ("upper", "exact")


def test_parameter_values_of_a_synthesis():
    folder = EXAMPLES / "template_parameter_synthesis"
    source = (folder / "geometric_parameter.ppy").read_text(encoding="utf-8")
    invariant = (folder / "invariants/geometric_parameter_invariant.ppy").read_text(encoding="utf-8")
    equivalence = compare(source, invariant, decisions={"0": {"strategy": "invariant",
                                                              "params": {"invariant": invariant}}})["equivalence"]
    assert equivalence["constraints"] and all("p" in c for c in equivalence["constraints"])
    assert equivalence["verdict"] == "unknown" and equivalence["relation"] == "first_below_second"


def test_gamblers_ruin_is_not_implemented():
    """The pair of #366: the check for every input is not implemented, and says so."""
    equivalence = compare(GAMBLER, GAMBLER_CLOSED, decisions={"0": {"strategy": "invariant",
                                                                    "params": {"invariant": GAMBLER_CLOSED}}}
                          )["equivalence"]
    assert equivalence["verdict"] == "unknown"
    assert equivalence["check"]["answer"] == "failed"
    assert equivalence["check"]["error"]["type"] == "NotImplementedError"
    assert equivalence["reason"].startswith("Not implemented")


def test_gamblers_ruin_on_one_input_has_a_witness():
    """On x = 1, n = 3 the unrolled loop is ruined with probability ≥ 0.66, the closed form says 1/3 (the
    branches of the pair in #366 are swapped), so the programs provably differ."""
    equivalence = compare(GAMBLER, GAMBLER_CLOSED, input_distribution="x*n**3",
                          decisions={"0": {"strategy": "unroll", "params": {"iterations": 30}}})["equivalence"]
    assert equivalence["verdict"] == "not_equivalent"
    witness = equivalence["witness"]
    assert witness["state"] == {"x": 0, "n": 3} and witness["input"] == "x*n**3"
    assert witness["first"]["kind"] == "lower" and witness["second"]["lower"] == "1/3"
    assert witness["first"]["lower_float"] > 0.66


def test_request_problems():
    assert validate_equivalence({"source": BERNOULLI, "syntax": "ppy"}) == ["The second program is empty."]
    problems = validate_equivalence({"source": BERNOULLI, "syntax": "ppy", "other_source": FAIR_COIN,
                                     "other_decisions": {"0": {"strategy": "no_such"}}})
    assert problems and problems[0].startswith("Second program:")
    syntax_error = compare(BERNOULLI, "x: nat;\nx = ;\n")
    assert syntax_error["status"] == "error" and syntax_error["error"]["message"].startswith("The second program")


# ---- endpoint ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def test_endpoint_runs_a_job(client):
    response = client.post("/api/equivalence", json={"source": BERNOULLI, "syntax": "ppy",
                                                     "other_source": FAIR_COIN}, headers=HOST)
    assert response.status_code == 202
    status = wait_for(client, response.get_json()["job"])
    assert status["state"] == "done" and status["summary"]["kind"] == "equivalence"
    assert status["result"]["equivalence"]["verdict"] == "not_equivalent"
    assert "Checking the equivalence for every input" in status["log"]


def test_endpoint_rejects_a_missing_second_program(client):
    response = client.post("/api/equivalence", json={"source": BERNOULLI, "syntax": "ppy"}, headers=HOST)
    assert response.status_code == 400 and "second program" in response.get_json()["error"]


def test_endpoint_can_be_cancelled(client):
    slow = "x: nat;\nc: nat;\nx = 1;\nwhile (x == 1) {\n    { x = 0; } [1/1000] { c = c + 1; }\n}\n"
    response = client.post("/api/equivalence", json={
        "source": slow, "syntax": "ppy", "other_source": slow,
        "decisions": {"*": {"strategy": "unroll", "params": {"iterations": 100000}}},
        "other_decisions": {"*": {"strategy": "unroll", "params": {"iterations": 100000}}}}, headers=HOST)
    job = response.get_json()["job"]
    cancelled = client.delete(f"/api/jobs/{job}", headers=HOST).get_json()
    assert cancelled["state"] == "cancelled"
    assert cancelled["result"]["error"]["message"] == "The comparison was cancelled."


def test_equivalence_examples_load_both_programs(client):
    loaded = client.get("/api/examples/equivalence/loop_free/bernoulli.ppy", headers=HOST).get_json()
    assert loaded["other"]["id"] == "equivalence/loop_free/bernoulli2.ppy"
    assert "bernoulli(2/3)" in loaded["other"]["source"] and loaded["other"]["syntax"] == "ppy"
    assert "both programs are loaded" in loaded["suggestion"]["note"]
    assert "analyses the first program" not in loaded["suggestion"]["note"]
    loopy = client.get("/api/examples/equivalence/loopy/geometric.ppy", headers=HOST).get_json()
    assert loopy["other"]["id"] == "equivalence/loopy/invariants/geometric_invariant.ppy"
    assert loopy["suggestion"]["decisions"]["0"]["params"]["invariant"] == loopy["other"]["source"]
    plain = client.get("/api/examples/" + next(
        e["id"] for e in client.get("/api/examples", headers=HOST).get_json()["examples"] if e["command"] == "main"),
        headers=HOST).get_json()
    assert "other" not in plain


def test_every_equivalence_example_has_its_second_program(client):
    examples = client.get("/api/examples", headers=HOST).get_json()["examples"]
    pairs = [e["id"] for e in examples if e["command"] == "check_equality"]
    assert pairs
    for example_id in pairs:
        assert client.get(f"/api/examples/{example_id}", headers=HOST).get_json().get("other"), example_id


# ---- frontend model --------------------------------------------------------------------------

MODEL_CHECK = """
const m = await import(%(module)r);
const exact = (rows) => ({ status: "ok", bounds: { kind: "exact" }, distribution: { variables: ["x"],
  marginals: { x: { rows, truncated: false } } } });
const first = exact([{ value: 0, prob: "1/3", p: 1 / 3 }, { value: 1, prob: "2/3", p: 2 / 3 }]);
const second = exact([{ value: 0, prob: "1/2", p: 0.5 }, { value: 1, prob: "1/2", p: 0.5 }, { value: 2, prob: "0", p: 0 }]);
const same = exact([{ value: 0, prob: "1/3", p: 1 / 3 }, { value: 1, prob: "2/3", p: 2 / 3 }]);
const lower = { ...second, bounds: { kind: "lower" } };
console.log(JSON.stringify({
  differ: m.compareMarginals(first, second, "x").map((r) => [r.value, r.differs]),
  same: m.compareMarginals(first, same, "x").map((r) => r.differs),
  bound: m.compareMarginals(first, lower, "x").map((r) => r.differs),
  notImplemented: m.verdictOf({ verdict: "unknown", check: { error: { type: "NotImplementedError" } } }),
  forged: m.verdictOf({ verdict: "bogus" }).key,
  proven: m.verdictOf({ verdict: "not_equivalent" }),
  relation: m.relationText("first_below_second"),
  constraints: m.constraintTexts([{ p: "1/3" }, { p: "1/2", q: "1" }]),
  interval: [m.intervalText({ lower: "1/3", upper: "1/3" }),
             m.intervalText({ lower: "1/3", upper: "123456789/987654321", upper_float: 0.125 }, (e, n) => String(n))],
  input: [m.inputText({ input: { source: "counterexample", distribution: "1" } }),
          m.inputText({ input: { source: "request", distribution: "x*n**3" } }), m.inputText({})],
  check: m.checkLines({ answer: "failed", error: { type: "NotImplementedError", message: "boom" } }),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_frontend_model(tmp_path):
    script = tmp_path / "check.mjs"
    script.write_text(MODEL_CHECK % {"module": (COMPONENTS / "equivalence-model.js").as_uri()}, encoding="utf-8")
    output = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60,
                                       check=True).stdout)
    assert output["differ"] == [[0, True], [1, True], [2, False]]
    assert output["same"] == [False, False]
    assert output["bound"] == [None, None, None]  # bounds are not compared
    assert output["notImplemented"]["title"] == "Not decided: not implemented"
    assert output["notImplemented"]["glyph"] == "?" and not output["notImplemented"]["proven"]
    assert output["forged"] == "unknown"
    assert output["proven"]["glyph"] == "≠" and output["proven"]["subtitle"] == "Proven by a witness."
    assert output["relation"].startswith("Program ⪯ Second program")
    assert output["constraints"] == ["p = 1/3", "p = 1/2, q = 1"]
    assert output["interval"] == ["1/3", "[1/3, 0.125]"]
    assert output["input"][0] == "Both programs on the input where every variable is 0 (the solver's counterexample)."
    assert output["input"][1] == "Both programs on the initial distribution x*n**3."
    assert output["input"][2] == "Both programs on the default input (every variable 0)."
    assert output["check"] == ["The check did not finish.", "Not implemented: boom"]


def test_page_has_the_second_editor(client):
    page = client.get("/", headers=HOST).get_data(as_text=True)
    assert 'id="editor-other"' in page and 'id="btn-compare"' in page and 'aria-pressed="false"' in page
    assert 'role="tablist" aria-label="Program to edit"' in page
    assert "equivalence-view.css" in page
