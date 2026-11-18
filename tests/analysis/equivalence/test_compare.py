"""The sound verdict of ``prodigy check_equality`` and of every caller of the second-order check (#395):
equivalence only if both analyses are exact, "not equivalent" only with a witness, undecided otherwise."""
import builtins
import os
from glob import glob
from typing import Dict, List

import pytest
from click.testing import CliRunner

from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence import equivalence_check
from prodigy.analysis.equivalence import verdict as eq
from prodigy.analysis.equivalence.equivalence_check import (CheckNotImplemented, DifferentVariablesError,
                                                            check_equivalence)
from prodigy.analysis.session import AnalysisSession, InteractiveDecisions
from prodigy.cli import EXIT_EQUIVALENT, EXIT_NOT_EQUIVALENT, EXIT_UNDECIDED, cli
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.pgcl import parse_pgcl

from tests.analysis.equivalence.test_equivalence_check import _loopy_invariants

# The pair of #395: the loop diverges for x >= 1, the invariant sets x to 0; Park's lemma proves only
# that the loop is below the invariant.
DIVERGING = "nat x;\nwhile (x >= 1) { x := x }\n"
DIVERGING_INVARIANT = "nat x;\nif (x >= 1) { x := 0 } else {skip}\n"


def _session(answers: List[str], monkeypatch: pytest.MonkeyPatch, syntax: str = "pgcl") -> AnalysisSession:
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _=None: next(replies))
    return AnalysisSession(None, syntax=syntax, config=ForwardAnalysisConfig(),
                           provider=InteractiveDecisions(remember=True), parameter_synthesis="allow")


def _compare(first: str, second: str, session: AnalysisSession, syntax: str = "pgcl") -> eq.Comparison:
    program, other = parse_pgcl(first, syntax), parse_pgcl(second, syntax)
    with session.activate(), session.config.activate():
        return eq.compare(program, other, session)


def _cli(tmp_path, first: str, second: str, suffix: str = ".pgcl", answers: str = ""):
    one, two = tmp_path / f"a{suffix}", tmp_path / f"b{suffix}"
    one.write_text(first)
    two.write_text(second)
    return CliRunner().invoke(cli, ["check_equality", str(one), str(two)], input=answers.replace("{b}", str(two)))


# ---- the pair of the issue -------------------------------------------------------------------

def test_park_upper_bound_proves_one_direction_only(tmp_path):
    result = _cli(tmp_path, DIVERGING, DIVERGING_INVARIANT, answers="1\n{b}\n")
    assert result.exit_code == EXIT_UNDECIDED, result.output
    assert "is equivalent" not in result.output
    assert "Verdict: " in result.output and "undecided" in result.output
    assert "one direction proven: first_below_second" in result.output
    assert "analysis of A: upper, of B: exact" in result.output
    assert "terminate almost surely" in result.output


def test_park_upper_bound_in_the_library(monkeypatch):
    invariant_file = "pgfexamples/equivalence/loopy/invariants/geometric_invariant.ppy"
    with open("pgfexamples/equivalence/loopy/geometric.ppy", encoding="utf-8") as handle:
        source = handle.read()
    with open(invariant_file, encoding="utf-8") as handle:
        invariant = handle.read()
    comparison = _compare(source, invariant, _session(["1", invariant_file], monkeypatch, "ppy"), "ppy")
    assert comparison.check.answer == "equal"
    assert comparison.verdict["verdict"] == eq.UNKNOWN
    assert comparison.verdict["relation"] == eq.FIRST_BELOW_SECOND


# ---- "not equivalent" needs a witness --------------------------------------------------------

@pytest.mark.parametrize("first, second", [
    ("nat x;\nnat y;\nx := 1\n", "nat x;\nx := 1\n"),
    ("nat x;\nx := 1\n", "nat x;\nnat y;\nx := 1\n"),
])
def test_different_variables_are_no_refutation(first, second, monkeypatch, tmp_path):
    # both directions: before #395 only variables of the first program were noticed, as "not equivalent"
    program, other = parse_pgcl(first), parse_pgcl(second)
    with AnalysisSession.detached().activate(), pytest.raises(DifferentVariablesError, match="different variables"):
        check_equivalence(program, other, ForwardAnalysisConfig(), compute_semantics)
    comparison = _compare(first, second, _session([], monkeypatch))
    assert comparison.verdict["verdict"] == eq.UNKNOWN and "only program" in comparison.verdict["reason"]
    result = _cli(tmp_path, first, second)
    assert result.exit_code == EXIT_UNDECIDED, result.output
    assert "Check for every input: not run" in result.output and "declare different variables" in result.output


class _Solver:
    """Equality of the distributions needs p = 1/2, equality of the error masses p = 1/3."""

    def __init__(self) -> None:
        self.answers = iter([(True, [{"p": "1/2"}]), (True, [{"p": "1/3"}])])

    def solve(self, *_):
        import sympy  # pylint: disable=import-outside-toplevel
        found, solutions = next(self.answers)
        return found, [{sympy.Symbol(k): sympy.S(v) for k, v in s.items()} for s in solutions]


def test_constraints_without_a_common_solution_are_no_refutation(monkeypatch):
    monkeypatch.setattr(equivalence_check.SolverType, "make", lambda *_: _Solver())
    program = parse_pgcl("nat x;\nrparam p;\n{x := 1} [p] {x := 0}\n")
    with AnalysisSession.detached().activate():
        answer, difference = check_equivalence(program, program, ForwardAnalysisConfig(), compute_semantics)
    assert answer is None and difference is not None


def test_a_witness_proves_programs_different(tmp_path):
    result = _cli(tmp_path, "x: nat;\nx = unif(1, 3);\n", "x: nat;\n{x = 1;} [1/2] {x = 2;}\n", ".ppy")
    assert result.exit_code == EXIT_NOT_EQUIVALENT, result.output
    assert "A is not equivalent to B" in result.output
    assert "Witness: on the input every variable 0, the output state x = 1 has probability" in result.output
    assert "1/3 in program A (exact)" in result.output and "1/2 in program B (exact)" in result.output
    # the reason does not repeat the witness (#413)
    assert "Reason: The programs give provably different probabilities to an output state (witness below)." \
        in result.output
    assert result.output.count("x = 1") == 1


def test_valuations_are_written_like_the_program_not_as_a_dict():
    assert eq.show_valuation({"x": 1, "n": 3}) == "x = 1, n = 3"
    assert eq.show_valuation({}) == "every variable 0"
    assert eq.show_valuation({}, "the empty state") == "the empty state"


def test_a_witness_from_a_lower_bound_asks_for_the_loop_once(tmp_path):
    loop = "x: nat;\nc: nat;\nx = 1;\nwhile (x == 1) {\n    {x = 0;} [1/2] {c = c + 1;}\n}\n"
    closed = "x: nat;\nc: nat;\nx = 0;\nc = geometric(1/3);\n"
    # strategy 2 (unroll) with 20 iterations, given once: the witness search takes the same answers
    result = _cli(tmp_path, loop, closed, ".ppy", answers="2\n20\n")
    assert result.exit_code == EXIT_NOT_EQUIVALENT, result.output
    assert result.output.count("Choose an option") == 1
    assert "analysis of A: lower, of B: exact" in result.output
    assert "in [1/2, 524289/1048576] in program A (lower)" in result.output


def test_a_lower_bound_without_a_witness_stays_undecided(monkeypatch):
    # equivalent, but three unrolled iterations only give a lower bound with residual mass 1/8
    loop = "x: nat;\nc: nat;\nx = 1;\nc = 0;\nwhile (x == 1) {\n    {x = 0;} [1/2] {c = c + 1;}\n}\n"
    closed = "x: nat;\nc: nat;\nx = 0;\nc = geometric(1/2);\n"
    comparison = _compare(loop, closed, _session(["2", "3"], monkeypatch, "ppy"), "ppy")
    assert comparison.check.answer == "different" and comparison.witness is None
    assert comparison.verdict["verdict"] == eq.UNKNOWN and "lower bound" in comparison.verdict["reason"]


def test_a_witness_on_a_boolean_counterexample(monkeypatch):
    first = "f: bool;\nc: nat;\nif (f) { c = 1; }\n"
    second = "f: bool;\nc: nat;\nif (f) { c = 2; }\n"
    comparison = _compare(first, second, _session([], monkeypatch, "ppy"), "ppy")
    assert comparison.verdict["verdict"] == eq.NOT_EQUIVALENT
    assert comparison.input.get("f") == 1 and comparison.witness is not None
    assert comparison.witness["state"]["f"] == 1


def test_equivalent_programs(tmp_path):
    result = _cli(tmp_path, "x: nat;\nx = unif(1, 3);\n", "x: nat;\n{x = 1;} [1/3] {{x = 2;} [1/2] {x = 3;}}\n",
                  ".ppy")
    assert result.exit_code == EXIT_EQUIVALENT, result.output
    assert "A is equivalent to B" in result.output and "Proven: both programs" in result.output


# ---- limitations of the check are "not implemented" ------------------------------------------

def test_a_non_finite_marginal_is_a_limitation_of_the_check(monkeypatch):
    def analyzer(*_):
        raise NonFiniteMarginalError("the marginal of x diverges")

    program = parse_pgcl("nat x;\nx := 1\n")
    with AnalysisSession.detached().activate(), pytest.raises(CheckNotImplemented, match="infinite mass"):
        check_equivalence(program, program, ForwardAnalysisConfig(), analyzer)
    monkeypatch.setattr("prodigy.analysis.analyzer.compute_semantics", analyzer)
    session = _session([], monkeypatch)
    with session.activate(), session.config.activate():
        check = eq.run_check(program, program, session)
    assert check.answer == "failed" and check.error["type"] == "NotImplementedError"
    verdict = eq.decide(check, None)
    assert verdict["verdict"] == eq.UNKNOWN and verdict["reason"].startswith("Not implemented:")


# ---- the example library keeps its verdicts or becomes more precise --------------------------

# Pairs whose check ends without a result (on main as an error of `check_equality`), with the reason
LIBRARY_FAILING: Dict[str, str] = {}  # #403 fixed the last two (the check lost the parameters of the loop)
# Pairs left out here, with the reason
LIBRARY_SKIPPED = {
    "brp_obs_parameter": "the check takes minutes (conditioning on the second-order generating function)",
}


def _library_pairs():
    pairs = [pytest.param(path, path.replace(".ppy", "2.ppy"), [], eq.EQUIVALENT, id=os.path.basename(path))
             for path in sorted(glob("pgfexamples/equivalence/loop_free/*.ppy")) if not path.endswith("2.ppy")]
    for folder in ("pgfexamples/equivalence/loopy", "pgfexamples/template_parameter_synthesis"):
        for path in sorted(glob(os.path.join(folder, "*.ppy"))):
            name = os.path.basename(path).removesuffix(".ppy")
            with open(path, encoding="utf-8") as handle:
                if handle.readline().lstrip("#").strip() == "skip" or name in LIBRARY_SKIPPED:
                    continue
            counterpart, invariants = _loopy_invariants(path)
            # the loops are handled by Park's lemma: an upper bound, so one direction is proven
            expected = "failed" if name in LIBRARY_FAILING else eq.FIRST_BELOW_SECOND
            pairs.append(pytest.param(path, counterpart, invariants, expected, id=name))
    return pairs


# random_walk_parameter takes 25 s on a laptop (Python 3.11); a loaded CI runner needed more than the default 120 s
@pytest.mark.timeout(300)
@pytest.mark.parametrize("path, counterpart, invariants, expected", _library_pairs())
def test_library_verdicts(monkeypatch, path, counterpart, invariants, expected):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    with open(counterpart, encoding="utf-8") as handle:
        other = handle.read()
    session = _session([answer for invariant in invariants for answer in ("1", invariant)], monkeypatch, "ppy")
    comparison = _compare(source, other, session, "ppy")
    if expected == "failed":
        assert comparison.check.answer == "failed"
        assert comparison.verdict["verdict"] == eq.UNKNOWN and "refuted" in comparison.verdict["reason"]
        return
    assert comparison.check.answer == "equal", comparison.check.error
    if expected == eq.EQUIVALENT:
        assert comparison.verdict["verdict"] == eq.EQUIVALENT
    else:
        assert comparison.verdict["verdict"] == eq.UNKNOWN
        assert comparison.verdict["relation"] == expected


def test_library_pairs_are_found():
    assert len(_library_pairs()) >= 25
    names = {os.path.basename(path).removesuffix(".ppy")
             for path in glob("pgfexamples/template_parameter_synthesis/*.ppy")}
    assert set(LIBRARY_FAILING) | set(LIBRARY_SKIPPED) <= names
