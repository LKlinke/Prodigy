"""Undefined values (nan, zoo, oo) never end with status ``ok`` (issue #165)."""
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import QueryResult, is_non_finite, sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.generating_function import GeneratingFunction, _continuous_at_one, _substitute_one
from prodigy.pgcl import parse_pgcl

ROOT = Path(__file__).resolve().parents[2]
MONTY_HALL = [
    pytest.param(ROOT / "pgfexamples/inference/loop_free/conditioning/monty_hall_nested.ppy", id="ppy"),
    pytest.param(ROOT / "tests/pgcl/former_examples/inference/loop_free/conditioning/monty_hall_nested.pgcl",
                 id="pgcl"),
]


def _engine(name):
    if name == "GINAC":
        pytest.importorskip("prodigy_ginac")
    return getattr(ForwardAnalysisConfig.Engine, name)


ENGINES = ["SYMPY", "GINAC", "SYMENGINE"]
SIMPLIFICATION = [pytest.param(True, id="simplified"), pytest.param(False, id="unsimplified")]


def _expected_monty_hall():
    door, player, prize = sympy.symbols("door player prize")
    # the door opened is neither the player's first choice nor the prize; the player switches
    expected = sympy.Integer(0)
    for first in (1, 2, 3):
        for prize_value in (1, 2, 3):
            doors = [d for d in (1, 2, 3) if d not in (first, prize_value)]
            for door_value in doors:
                switched = next(d for d in (1, 2, 3) if d not in (first, door_value))
                expected += sympy.Rational(1, 9) / len(doors) * door ** door_value \
                    * player ** switched * prize ** prize_value
    return expected


@pytest.mark.parametrize("path", MONTY_HALL)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_monty_hall_nested_is_exact(path, engine, simplification):
    source = path.read_text(encoding="utf-8")
    syntax = "ppy" if path.suffix == ".ppy" else "pgcl"
    config = ForwardAnalysisConfig(engine=_engine(engine), use_simplification=simplification)
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config,
                              decisions={}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == Bound.EXACT
    (query,) = result["queries"]
    assert sym(query["value"]) == sympy.Rational(2, 3)
    assert query["bound"] == Bound.EXACT
    computed = sym(result["distribution"]["closed_form"])
    computed = computed.subs({s: sympy.Symbol(s.name) for s in computed.free_symbols})
    assert sympy.cancel(computed - _expected_monty_hall()) == 0


# x is uniform on {1, 2, 3}, written with a removable singularity at x = 1
UNSIMPLIFIED = "x*(x^3 - 1)/(3*(x - 1))*y"
SIMPLIFIED = "(x + x^2 + x^3)/3*y"
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment", ["y = x", "y = x + 1", "y = 2 * x", "y = x - 1", "y = 2",
                                        "x = y", "x = 5", "x = x + y", "y = x * x", "y = x // 2",
                                        "y = x % 2"])
def test_assignment_at_removable_singularity(engine, simplification, assignment):
    config = ForwardAnalysisConfig(engine=_engine(engine), use_simplification=simplification)
    program = parse_pgcl(f"x: nat;\ny: nat;\n{assignment};\n", "ppy")
    results = []
    with config.activate():
        for function in (UNSIMPLIFIED, SIMPLIFIED):
            dist = config.factory.from_expr(function, "x", "y")
            final, _ = compute_discrete_distribution(program, dist, config)
            results.append(sym(str(final)))
    assert not is_non_finite(results[0])
    assert sympy.cancel(results[0] - results[1]) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_assignment_at_removable_singularity_in_a_session(engine):
    source = "x: nat;\ny: nat;\nx = y;\nquery Pr[x == 1];\n"
    config = ForwardAnalysisConfig(engine=_engine(engine))
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                              decisions={}, input_distribution=UNSIMPLIFIED, capture_output=True)
    result = session.run().to_json()
    # all engines, GiNaC since prodigy-ppl-ginac 0.0.5, cancel the removable singularity
    assert result["status"] == "ok"
    assert sym(result["queries"][0]["value"]) == 1  # y is 1 with certainty
    assert result["bounds"]["kind"] == Bound.EXACT


def test_substitute_one_takes_the_limit_at_a_removable_singularity():
    x, y = sympy.Symbol("x", real=True), sympy.Symbol("y", real=True)
    assert _substitute_one((x ** 3 - 1) / (x - 1) * y, x) == 3 * y
    assert _substitute_one(x ** 2 * y, x) == y
    # a pole stays undefined, so the guard below sees it
    assert is_non_finite(_substitute_one(y / (1 - x), x))


def test_substitute_one_cancels_before_taking_the_limit():
    # SymPy's limit does not finish on this form of 0 (#259); cancelling first gives the value at once
    z = sympy.Symbol("z", real=True)
    assert _substitute_one(-z ** 2 / 2 - z / 2 + z * (z ** 2 - 1) / (2 * (z - 1)), z) == 0
    # a function that may jump at 1 takes the limit from below, not the value at 1
    assert _substitute_one(sympy.floor(z), z) == 0


@pytest.mark.parametrize("function, continuous", [
    ("(z**3 - 1)/(z - 1)", True),  # continuous where finite; the value at 1 comes from the cancelled form
    ("exp(2*z - 2)*z**5", True),
    ("log(1 - z/3)", True),
    ("sqrt(1 - z)", True),  # base 0 under a positive constant exponent
    ("z**p", True),  # base positive at 1
    ("log(2)*sign(p) + z", True),  # parts without z are constant
    ("1/sqrt(1 - z)", False),  # base 0 under a negative exponent: a pole
    ("floor(z)", False),
    ("Abs(z - 1)", False),
    ("Piecewise((1, z < 1), (2, True))", False),
    ("log(z + p)", False),  # p + 1 not known to be positive
    ("(z - 2)**(1/3)", False),  # negative base: branch cut
    ("0**(z - 1)", False),
])
def test_continuity_at_one_is_an_allowlist(function, continuous):
    z, p = sympy.Symbol("z", real=True), sympy.Symbol("p", real=True)
    assert _continuous_at_one(sympy.sympify(function, locals={"z": z, "p": p}), z) is continuous


# observe after unif: the unobserved part has a removable singularity at 1 without simplification (#259)
OBSERVE_AFTER_UNIF = [
    pytest.param("z: nat = unif(1, 2);\nobserve(z < 3);\n", "z", id="observe"),
    pytest.param("def f(a: nat) -> nat {\n    observe(a < 3);\n    return a;\n}\n"
                 "z: nat = unif(1, 2);\nx: nat = f(z);\n", "z*x", id="call_in_joint"),
]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("program, variable", OBSERVE_AFTER_UNIF)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_observe_after_unif_is_fast(program, variable, engine, simplification):
    config = ForwardAnalysisConfig(engine=_engine(engine), use_simplification=simplification)
    session = AnalysisSession(parse_pgcl(program, "ppy"), source=program, syntax="ppy", config=config,
                              decisions={}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == Bound.EXACT
    computed = sym(result["distribution"]["closed_form"])
    computed = computed.subs({s: sympy.Symbol(s.name) for s in computed.free_symbols})
    single = sympy.sympify(variable)
    assert sympy.cancel(computed - (single + single ** 2) / 2) == 0


@pytest.mark.parametrize("value", ["nan", "NaN", "zoo", "oo", "-oo", "\\infty", "∞", "x*nan", "2*zoo + x",
                                   "Infinity", sympy.nan, sympy.zoo * sympy.Symbol("x")])
def test_non_finite_values_are_recognised(value):
    assert is_non_finite(value, names=frozenset({"x"}))


@pytest.mark.parametrize("value", ["1/3", "x^2*y/2", "inf*x", "0", sympy.Rational(2, 3), None])
def test_finite_values_are_not_flagged(value):
    # 'inf' is a program variable here, not infinity
    assert not is_non_finite(value, names=frozenset({"x", "y", "inf"}))


def test_infinite_expectation_is_a_value():
    assert not is_non_finite("\\infty", infinity=True)
    assert not is_non_finite(sympy.oo, infinity=True)
    assert is_non_finite("nan", infinity=True)


PROGRAM = "x: nat;\nx = unif(1, 3);\nquery Pr[x == 1];\nquery Ex[x];\n"


def _run_with(monkeypatch, function, queries):
    """Runs a session whose analysis yields the SymPy distribution ``function`` and ``queries``."""
    config = ForwardAnalysisConfig()
    session = AnalysisSession(parse_pgcl(PROGRAM, "ppy"), source=PROGRAM, syntax="ppy", config=config,
                              decisions={}, capture_output=True)

    def fake(*_args, **_kwargs):
        session.queries.extend(queries)
        return GeneratingFunction(function, "x"), GeneratingFunction("0", "x")

    monkeypatch.setattr("prodigy.analysis.analyzer.compute_discrete_distribution", fake)
    return session.run().to_json()


def test_nan_distribution_is_an_error(monkeypatch):
    x = sympy.Symbol("x", real=True)
    result = _run_with(monkeypatch, sympy.nan * x, [])
    assert result["status"] == "error"
    assert result["error"]["type"] == "NonFiniteResult"
    assert "the distribution" in result["error"]["details"]["parts"]
    assert result["bounds"]["kind"] == Bound.UNKNOWN


@pytest.mark.parametrize("kind, value", [("probability", "nan"), ("probability", "\\infty"),
                                         ("expectation", "zoo"), ("probability", "x*nan")])
def test_undefined_query_value_loses_its_guarantee(monkeypatch, kind, value):
    # the query has no guarantee, but the analysis does not fail: the distribution is not affected (#316)
    query = QueryResult(kind=kind, expr="x == 1", value=value, lower=value, upper=value,
                        bound=Bound.EXACT, line=3)
    result = _run_with(monkeypatch, "x/3 + x^2/3 + x^3/3", [query])
    assert result["status"] == "ok", result["error"]
    assert result["distribution"] is not None
    assert any("the query x == 1 (line 3)" in warning for warning in result["warnings"])
    (answer,) = result["queries"]
    assert answer["bound"] == Bound.UNKNOWN
    assert answer["lower"] is None and answer["upper"] is None


def test_infinite_expectation_stays_ok(monkeypatch):
    query = QueryResult(kind="expectation", expr="x", value="\\infty", lower="2", upper="\\infty",
                        bound=Bound.LOWER, line=4)
    result = _run_with(monkeypatch, "x/3 + x^2/3 + x^3/3", [query])
    assert result["status"] == "ok", result["error"]
    assert result["queries"][0]["bound"] == Bound.LOWER
