"""Parameter synthesis with the strategy ``invariant`` and the second-order check (#403): the check keeps the
parameters of the loop, the GiNaC engine gets them as parameters, "refuted" needs a witness, and the
test distribution of the check (infinite mass by design) does not crash unrolling on GiNaC."""
import builtins
from pathlib import Path

import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence import verdict as eq
from prodigy.analysis.instructionhandler.while_handler import InvariantRefutedError
from prodigy.analysis.session import AnalysisSession, InteractiveDecisions
from prodigy.distribution.distribution import NonFiniteMarginalError, known_convergent_series
from prodigy.language import parse_program
from prodigy.language.operations import cav_phi

Engine = ForwardAnalysisConfig.Engine
SYNTHESIS = Path("pgfexamples/template_parameter_synthesis")
ENGINES = [Engine.SYMPY, Engine.GINAC]


def _engine(engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine)


def _compare(source, other, answers, engine, monkeypatch):
    replies = iter(answers * 10)
    monkeypatch.setattr(builtins, "input", lambda _=None: next(replies))
    session = AnalysisSession(None, syntax="ppy", config=_engine(engine),
                              provider=InteractiveDecisions(remember=True), parameter_synthesis="allow")
    with session.activate(), session.config.activate():
        return eq.compare(parse_program(source, "ppy"), parse_program(other, "ppy"), session)


def test_one_unrolling_keeps_the_parameters_of_the_loop():
    loop = parse_program("c: nat;\nparam p: real;\nwhile (c == 1) { { c = 0; } [p] { skip; } }\n", "ppy")
    invariant = parse_program("c: nat;\nc = 0;\n", "ppy")
    assert "p" in cav_phi(loop, invariant).parameters


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name, constraints, relation", [
    # the invariants are those of p = 1/2; the check used to lose p and refute them
    ("geometric_observe_parameter", [{"p": "1/2"}], eq.FIRST_BELOW_SECOND),
    # its invariant samples iid((1 - sqrt(1 - c^2))/c, s), proven a probability generating function by the shift
    # and the monomial substitution (#431; before, no relation was proven, #424)
    ("random_walk_parameter", [{"p": "1/2"}], eq.FIRST_BELOW_SECOND),
    # GiNaC took the parameter of the iid distribution for a variable ("must be univariate")
    ("negative_binomial_parameter", [], eq.FIRST_BELOW_SECOND),
    ("negative_binomial_reversed_parameter", [{"p": "1/2"}], eq.FIRST_BELOW_SECOND),
])
def test_synthesis_examples_find_their_parameter_values(engine, name, constraints, relation, monkeypatch):
    invariant = SYNTHESIS / "invariants" / f"{name}_invariant.ppy"
    comparison = _compare((SYNTHESIS / f"{name}.ppy").read_text(encoding="utf-8"),
                          invariant.read_text(encoding="utf-8"), ["1", str(invariant)], engine, monkeypatch)
    assert comparison.check.answer == "equal", comparison.check.error
    assert comparison.check.constraints == constraints
    assert comparison.verdict["relation"] == relation
    if relation is None:
        assert comparison.verdict["verdict"] == eq.UNKNOWN
        assert "could not prove that" in comparison.verdict["reason"]
        assert "(1 - (1 - c * c) ** (1/2)) / c" in comparison.verdict["reason"]


LOOP = "c: nat;\nx: nat;\n{param}while (c == 1) {{\n    {{ c = 0; }} [{prob}] {{ x = x + 1; }}\n}}\n"
WRONG = "c: nat;\nx: nat;\nif (c == 1) {\n    { c = 0; } [1/3] { c = 0; x = x + 1; }\n} else { skip; }\n"


@pytest.mark.parametrize("engine", ENGINES)
def test_refuted_only_with_a_witness(engine, monkeypatch, tmp_path):
    invariant = tmp_path / "inv.ppy"
    invariant.write_text(WRONG, encoding="utf-8")
    comparison = _compare(LOOP.format(param="", prob="1/2"), WRONG, ["1", str(invariant)], engine, monkeypatch)
    error = comparison.check.error
    assert error["verdict"] == "refuted" and "provably different" in error["message"]
    witness = error["details"]["witness"]
    # the engines list the states in different orders: I gives x = 0 and x = 1 probabilities 1/3 and 2/3,
    # one unrolling 1/2 and 1/6
    expected = {0: ("1/3", "1/2"), 1: ("2/3", "1/6")}[witness["state"]["x"]]
    assert witness["state"]["c"] == 0
    assert (witness["first"]["lower"], witness["second"]["lower"]) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_no_parameter_values_is_not_proven_not_refuted(engine, monkeypatch, tmp_path):
    """The solver finds no value of p, but the probabilities depend on p: no witness, so not refuted."""
    invariant = tmp_path / "inv.ppy"
    invariant.write_text(WRONG, encoding="utf-8")
    comparison = _compare(LOOP.format(param="param p: real;\n", prob="p"), WRONG, ["1", str(invariant)], engine,
                          monkeypatch)
    error = comparison.check.error
    assert error["type"] == InvariantRefutedError.__name__
    assert error["verdict"] == "unknown" and "not proven" in error["message"]
    assert "witness" not in error["details"] and error["details"]["counterexample"]
    assert comparison.verdict["verdict"] == eq.UNKNOWN


@pytest.mark.parametrize("engine, answers", [
    pytest.param(Engine.SYMPY, ["2", "3"], id="unroll-sympy"),
    pytest.param(Engine.GINAC, ["2", "3"], id="unroll-ginac"),
    # auto tries the synthesis first, which takes its time limit on SymPy; the failure was GiNaC's
    pytest.param(Engine.GINAC, ["10"], id="auto-ginac"),
])
def test_unrolling_in_the_check_is_undecided(engine, answers, monkeypatch):
    """GiNaC failed with "power::eval(): division by zero" on the mass of the test distribution."""
    source = Path("pgfexamples/equivalence/loopy/geometric.ppy").read_text(encoding="utf-8")
    other = Path("pgfexamples/equivalence/loopy/invariants/geometric_invariant.ppy").read_text(encoding="utf-8")
    comparison = _compare(source, other, answers, engine, monkeypatch)
    assert comparison.check.error is None
    assert comparison.check.first == "lower"
    assert comparison.verdict["verdict"] == eq.UNKNOWN


def test_ginac_mass_at_a_pole_is_no_value_error():
    pytest.importorskip("prodigy_ginac")
    dist = ForwardAnalysisConfig(engine=Engine.GINAC).factory.from_expr("1/(1-x)", "x")
    # inside an analysis (as in the second-order check) the convergence guard of #398 does not apply, and the
    # engine itself must not fail with a ValueError
    with known_convergent_series(True), pytest.raises(NonFiniteMarginalError, match="division by zero"):
        dist.get_probability_mass()


@pytest.mark.parametrize("expression, variables", [
    ("(1/2)*c/(1-(1/2)*c)", ("c", "x", "temp")),  # variables that do not occur
    ("6/((2-x)*(3-y))", ("x", "y")),
    ("p*x/(1-(1-p)*x*y)", ("x", "y", "z")),  # a parameter
])
def test_ginac_multivariate_iteration_matches_the_filtered_probabilities(expression, variables):
    """The iteration expands one variable at a time (the witness search took minutes when every state was
    filtered on its own); it lists the same states, in the graded order, with the same probabilities."""
    pytest.importorskip("prodigy_ginac")
    import itertools  # pylint: disable=import-outside-toplevel
    import sympy  # pylint: disable=import-outside-toplevel
    dist = ForwardAnalysisConfig(engine=Engine.GINAC).factory.from_expr(expression, *variables)
    terms = list(itertools.islice(iter(dist), 10))
    assert len(terms) == 10
    totals = [sum(state.valuations.values()) for _, state in terms]
    assert totals == sorted(totals)
    for probability, state in terms:
        condition = " and ".join(f"{k} == {v}" for k, v in state.valuations.items())
        expected = dist.get_probability_of(condition)
        assert sympy.simplify(sympy.sympify(probability.replace("^", "**"))
                              - sympy.sympify(str(expected).replace("^", "**"))) == 0
