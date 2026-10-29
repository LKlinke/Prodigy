"""Tests for the non-interactive analysis API (sessions, loop strategies, JSON results)."""
import json
from fractions import Fraction

import pytest
from click.testing import CliRunner
from probably.pgcl import parse_pgcl

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.while_handler import finite_mass
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import (AnalysisSession, LoopDecision, LoopRegistry, parse_decisions,
                                      validate_decisions)

GEOMETRIC = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Pr[c = 0]
?Ex[c]
"""

# terminates with probability 1/2 only
BIASED_WALK = """nat x;
x := 1
while (x > 0) { {x := x - 1} [1/3] {x := x + 1} }
?Pr[x = 0]
"""

CONDITIONED = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
observe(c < 3)
?Pr[c = 0]
"""


def analyse(source, decisions, **kwargs):
    config = ForwardAnalysisConfig(use_simplification=True)
    session = AnalysisSession(parse_pgcl(source), source=source, config=config, decisions=decisions,
                              capture_output=True, **kwargs)
    return session.run().to_json()


def fraction(text):
    return Fraction(text.replace("^", "**")) if text is not None else None


def query(result, expr):
    return next(q for q in result["queries"] if q["expr"] == expr)


def test_loop_registry_assigns_ids_in_textual_order():
    source = """nat x; nat y;
fun f := { nat z; while (z > 0) { z := z - 1 } return z };
while (x > 0) {
  x := x - 1
  # while (comment) is ignored
  while (y > 0) { y := y - 1 }
}
"""
    loops = list(LoopRegistry(parse_pgcl(source), source))
    assert [(l.id, l.line, l.depth, l.function) for l in loops] == \
           [(0, 2, 0, "f"), (1, 3, 0, None), (2, 6, 1, None)]
    assert loops[1].cond == "x > 0"


def test_missing_decision_is_reported_not_asked():
    result = analyse(GEOMETRIC, {})
    assert result["status"] == "needs_decision"
    assert result["error"]["loop"]["id"] == 0
    assert result["error"]["loop"]["line"] == 3


def test_unrolling_gives_sound_interval_with_residual_mass():
    result = analyse(GEOMETRIC, {"0": "unroll:iterations=5"})
    assert result["status"] == "ok"
    assert result["bounds"]["kind"] == Bound.LOWER
    assert fraction(result["bounds"]["residual_mass"]) == Fraction(1, 32)
    pr = query(result, "c = 0")
    assert (fraction(pr["lower"]), fraction(pr["upper"])) == (Fraction(1, 2), Fraction(17, 32))
    expectation = query(result, "c")
    assert expectation["bound"] == Bound.LOWER and expectation["upper"] is None
    assert expectation["line"] == 5


def test_unrolling_until_fixpoint_is_exact():
    source = "nat x; nat y;\nx := 3\nwhile (x > 0) { x := x - 1; y := y + 2 }\n?Ex[y]\n"
    result = analyse(source, {"*": "unroll:iterations=10"})
    assert result["bounds"]["kind"] == Bound.EXACT
    assert query(result, "y")["value"] == "6"


def test_residual_is_rescaled_when_conditioning():
    result = analyse(CONDITIONED, {"0": {"strategy": "unroll", "params": {"iterations": 5}}})
    pr = query(result, "c = 0")
    lower, upper = fraction(pr["lower"]), fraction(pr["upper"])
    assert (lower, upper) == (Fraction(16, 29), Fraction(17, 29))
    assert lower <= Fraction(4, 7) <= upper  # exact posterior probability


def test_interval_without_termination_assumption():
    result = analyse(BIASED_WALK, {"0": "unroll:iterations=10"})
    pr = query(result, "x = 0")
    assert pr["bound"] == Bound.LOWER
    assert fraction(pr["lower"]) <= Fraction(1, 2) <= fraction(pr["upper"])


def test_evt_invariant_with_finite_mass_is_exact():
    result = analyse(GEOMETRIC, {"0": {"strategy": "evt_invariant",
                                       "params": {"invariant": "(x + 1/2)/(1 - c/2)"}}})
    assert result["bounds"]["kind"] == Bound.EXACT
    certificate = result["loops"][0]["certificates"][0]
    assert certificate["nonnegative"] is True and certificate["finite_mass"] is True
    assert certificate["mass"] == "3"
    assert query(result, "c")["value"] == "1"


def test_spurious_fixed_point_is_only_an_upper_bound():
    # The synthesised invariant has infinite mass: Prodigy's value 1 is wrong (truth: 1/2),
    # and the result must not claim more than an upper bound.
    result = analyse(BIASED_WALK, {"0": "evt_synthesis"})
    assert result["bounds"]["kind"] == Bound.UPPER
    pr = query(result, "x = 0")
    assert pr["lower"] is None and fraction(pr["upper"]) >= Fraction(1, 2)
    assert result["loops"][0]["certificates"][0]["finite_mass"] is False


def test_invariant_strategy_is_an_upper_bound_by_park():
    # pgfexamples/equivalence/loopy/geometric.pgcl with its invariant
    source = """nat x; nat c; nat temp;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1}; temp := 0 }
?Pr[c = 0]
"""
    invariant = """nat x; nat c; nat temp;
if (x = 1) { temp := geometric(1/2); c := c + temp; x := 0; temp := 0 } else { skip }
"""
    result = analyse(source, {"0": {"strategy": "invariant", "params": {"invariant": invariant}}})
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == Bound.UPPER
    assert result["loops"][0]["certificates"][0]["verdict"] == "verified"
    assert fraction(query(result, "c = 0")["upper"]) == Fraction(1, 2)


def test_refuted_invariant_reports_counterexample():
    invariant = "nat x; nat c;\nx := 0\n"
    result = analyse(GEOMETRIC, {"0": {"strategy": "invariant", "params": {"invariant": invariant}}})
    assert result["status"] == "error"
    assert result["error"]["details"]["verdict"] == "refuted"
    assert result["loops"][0]["failed"]


def test_interactive_session_keeps_historic_prompts(monkeypatch):
    prompts = []
    answers = iter(["2", "4"])

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    program = parse_pgcl(GEOMETRIC)
    session = AnalysisSession.interactive(program, source=GEOMETRIC, capture_output=True)
    result = session.run().to_json()
    assert prompts[0].startswith("While Instruction has only limited support. Choose an option:\n[1]: ")
    assert "[6]: EVT Invariant syntesis\n[q]: Quit.\n" in prompts[0]
    assert prompts[1] == "Specify a maximum iteration limit: "
    assert fraction(result["bounds"]["residual_mass"]) == Fraction(1, 16)


def test_decision_parsing_and_validation():
    assert LoopDecision.parse_spec("unroll:iterations=7") == LoopDecision("unroll", {"iterations": "7"})
    decisions = parse_decisions({"0": "unroll:iterations=x", "1": "nonsense", "*": {"strategy": "evt_synthesis"}})
    problems = validate_decisions(decisions, LoopRegistry(parse_pgcl(GEOMETRIC), GEOMETRIC))
    assert any("Iterations" in p for p in problems)
    assert any("Unknown loop strategy 'nonsense'" in p for p in problems)
    assert any("no loop with this id" in p for p in problems)


@pytest.mark.parametrize("expression, expected", [
    ("(x + 1/2)/(1 - c/2)", (True, 3)),
    ("1/(1 - 2*x)", (False, None)),
    ("(1 + x)/(1 - x)", (False, None)),
    ("1/(1 - p*x)", (None, None)),
    ("(1 - sqrt(1 - c^2))/c", (None, None)),
])
def test_finite_mass(expression, expected):
    finite, mass = finite_mass(expression, ["x", "c"])
    assert finite is expected[0]
    assert (mass == expected[1]) if expected[1] is not None else mass is None


def test_result_is_json_serialisable_and_complete():
    result = analyse(GEOMETRIC, {"0": "unroll:iterations=3"}, record_steps=True, coefficient_limit=5)
    text = json.dumps(result)
    assert json.loads(text)["schema"] == "prodigy/analysis-result@1"
    marginal = result["distribution"]["marginals"]["c"]
    assert [row["value"] for row in marginal["rows"]] == [0, 1, 2]
    assert all("upper" in row for row in marginal["rows"])
    assert len(result["steps"]) == 4
    assert result["program"]["variables"] == {"x": "nat", "c": "nat"}


def test_cli_analyze(tmp_path):
    from prodigy.cli import cli  # pylint: disable=import-outside-toplevel
    program = tmp_path / "geometric.pgcl"
    program.write_text(GEOMETRIC)
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(program), "--loop", "0=unroll:iterations=5", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok" and data["bounds"]["kind"] == "lower"

    missing = runner.invoke(cli, ["analyze", str(program)])
    assert missing.exit_code == 2
    assert "No strategy chosen for loop 0" in missing.output


def test_inner_upper_bound_keeps_outer_invariant_sound():
    from prodigy.web.examples import ExampleLibrary  # pylint: disable=import-outside-toplevel
    example = ExampleLibrary("pgfexamples").load("equivalence/loopy/nested_while.pgcl")
    decisions = example["suggestion"]["decisions"]
    assert set(decisions) == {"0", "1"}
    result = analyse(example["source"], decisions, input_distribution="x")
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == Bound.UPPER
    loops = {loop["id"]: loop for loop in result["loops"]}
    assert loops[1]["inside_checks"] == 1 and loops[0]["kind"] == Bound.UPPER
