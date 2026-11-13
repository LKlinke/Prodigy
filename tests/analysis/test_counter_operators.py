"""
Probabilities rational in a counter, applied as operators on the generating function, and EVT invariants that
are exponential polynomials (#323); see :mod:`prodigy.analysis.instructionhandler.counter_operators` and
:mod:`prodigy.analysis.evtinvariants.exp_polynomial`.
"""
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants import exp_polynomial
from prodigy.analysis.evtinvariants.exp_polynomial import (analytic_at_zero, entire_exp_polynomial,
                                                           exp_polynomial_nonneg, integral_from_zero)
from prodigy.analysis.instructionhandler.counter_operators import CounterProbability, counter_probability
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.equality import decide_equal
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.parser import parse_expr

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE,
           ForwardAnalysisConfig.Engine.GINAC]

EXAMPLE = Path(__file__).parents[2] / "pgfexamples/inference/loopy/conditioning/poisson_from_coins_choice.ppy"
INVARIANT = "ok*exp(k/2)*(1 + stop/2) + (k/2 - 1)*exp(k/2) + 1"

k, x, ok, stop = sympy.symbols("k x ok stop")


def _run(source, engine, decisions=None):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    # simplification as in the CLI and the GUI: without it, SymPy may print exp(1) as E (read back as a symbol)
    config = ForwardAnalysisConfig(engine=engine, use_simplification=True)
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                              decisions=decisions or {}, capture_output=True)
    return session.run().to_json()


def _evt(invariant):
    return {"0": {"strategy": "evt_invariant", "params": {"invariant": invariant}}}


def _equal(value, expected) -> bool:
    return decide_equal(sym(value), sympy.S(expected)) is True


# --- the Poisson example --------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_poisson_from_coins_choice_is_exact(engine):
    result = _run(EXAMPLE.read_text(), engine, _evt(INVARIANT))
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == Bound.EXACT
    certificate = loop["certificates"][0]
    assert certificate["nonnegative"] is True
    assert certificate["finite_mass"] is True
    assert _equal(certificate["mass"], 1 + sympy.exp(sympy.Rational(1, 2)))
    half = sympy.Rational(1, 2)
    expected = [sympy.exp(-half), sympy.exp(-half) / 2, sympy.exp(-half) / 8, half]  # Poisson(1/2)
    for query, value in zip(result["queries"], expected):
        assert query["bound"] == Bound.EXACT
        assert _equal(query["value"], value), (query["value"], value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("invariant, verdicts", [
    pytest.param(INVARIANT + " + 1", ("refuted",), id="off-by-a-constant"),
    pytest.param("ok*exp(k/3)*(1 + stop/2) + (k/2 - 1)*exp(k/2) + 1", ("refuted", "unknown"),
                 id="wrong-derivative"),
    # the guard part does not vanish at k = 0 by the right amount: the integral gets a wrong constant
    pytest.param("ok*(exp(k/2) + 1)*(1 + stop/2) + (k/2 - 1)*exp(k/2) + 1", ("refuted",), id="wrong-value-at-0"),
    pytest.param("ok*exp(k/2)*(1 + stop/2) + (k/2 - 1)*exp(k/2)", ("refuted",), id="integral-not-0-at-0"),
])
def test_wrong_candidates_are_rejected(engine, invariant, verdicts):
    result = _run(EXAMPLE.read_text(), engine, _evt(invariant))
    loop = result["loops"][0]
    assert loop["kind"] == Bound.UNKNOWN
    assert loop["certificates"][0]["verdict"] in verdicts
    assert all(query["bound"] == Bound.UNKNOWN for query in result["queries"])


@pytest.mark.parametrize("engine", ENGINES)
def test_unproven_non_negativity_gives_unknown(engine):
    # an independent geometric factor 1/(2 - j): a fixed point, but A·exp(c·k) + B with polynomials A, B is the
    # only non-rational form whose non-negativity is decided
    source = EXAMPLE.read_text().replace("stop: nat = 0;", "stop: nat = 0;\nj: nat = geometric(1/2);")
    result = _run(source, engine, _evt(f"({INVARIANT})/(2 - j)"))
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == Bound.UNKNOWN
    assert loop["certificates"][0]["nonnegative"] is None
    assert all(query["bound"] == Bound.UNKNOWN for query in result["queries"])


# --- loop-free programs ---------------------------------------------------------------------------------------

LOOP_FREE = """k: nat = poisson(1);
x: nat = 0;
{x = 1;} [1/(k + 1)] {x = 2;}
if (k == 0) {x = x + 5;}
query Pr[x == 6];
query Pr[x == 1];
query Pr[x == 2];
"""

LOOP_FREE_SHIFT = """k: nat = poisson(1);
x: nat = 0;
{x = 1;} [1/(k + 2)] {x = 2;}
query Pr[k == 0 and x == 1];
query Pr[k == 1 and x == 1];
query Pr[k == 1];
query Ex[k * x];
"""


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    # Pr[x = 1] = Σ_(n≥1) e^(-1)/n! · 1/(n + 1) = 1 - 2/e
    pytest.param(LOOP_FREE, [sympy.exp(-1), 1 - 2 * sympy.exp(-1), sympy.exp(-1)], id="1/(k+1)"),
    # Ex[k·x] = Σ_n e^(-1)/n! · n · (2 - 1/(n + 2)) = 1 + 2/e
    pytest.param(LOOP_FREE_SHIFT, [sympy.exp(-1) / 2, sympy.exp(-1) / 3, sympy.exp(-1), 1 + 2 * sympy.exp(-1)],
                 id="1/(k+2)"),
])
def test_loop_free_choices_are_exact(engine, source, expected):
    result = _run(source, engine)
    assert result["status"] == "ok", result.get("error")
    for query, value in zip(result["queries"], expected):
        assert query["bound"] == Bound.EXACT
        assert _equal(query["value"], value), (query["value"], value)


@pytest.mark.parametrize("engine", ENGINES)
def test_non_exponential_integrand_is_not_implemented(engine):
    # a geometric k: the integral of 1/(2 - k) is a logarithm, which is not implemented
    result = _run("k: nat = geometric(1/2);\nx: nat = 0;\n{x = 1;} [1/(k + 1)] {x = 2;}\nquery Pr[x == 1];",
                  engine)
    assert result["status"] == "error"
    assert "exp(c·k)" in result["error"]["message"]


# --- units ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("probability, expected", [
    ("1/(k + 1)", CounterProbability("k", 0, 1, 1)),
    ("k/(k + 1)", CounterProbability("k", 1, -1, 1)),
    ("1 - 1/(k + 1)", CounterProbability("k", 1, -1, 1)),
    ("1/(2*k + 2) + 1/2", CounterProbability("k", sympy.Rational(1, 2), sympy.Rational(1, 2), 1)),
    ("1/(k + 3)", CounterProbability("k", 0, 1, 3)),
    ("2/(k + 1)", None),  # 2 at k = 0
    ("1/k", None),  # undefined at k = 0
    ("1/(k - 1 + 2)", None),  # k - 1 is a monus for k = 0
    ("1/(k*k + 1)", None),
    ("1/(k + 1/2)", None),
])
def test_counter_probability_shapes(probability, expected):
    result = counter_probability(parse_expr(probability), {"k"})
    if expected is None:
        assert result is None
    else:
        assert result == CounterProbability(expected.variable, *(sympy.Rational(v) for v in expected[1:3]),
                                            expected.shift)


def test_counter_probability_needs_one_variable():
    assert counter_probability(parse_expr("k/(k + j + 1)"), {"k", "j"}) is None


def test_integral_from_zero():
    variables = [k, x]
    assert decide_equal(integral_from_zero(sympy.exp(k / 2), k, variables), 2 * sympy.exp(k / 2) - 2) is True
    assert decide_equal(integral_from_zero(k ** 2 * x, k, variables), k ** 3 * x / 3) is True
    integral = integral_from_zero(k ** 3 * sympy.exp(-k) / (2 - x), k, variables)
    assert decide_equal(sympy.diff(integral, k), k ** 3 * sympy.exp(-k) / (2 - x)) is True
    assert integral.subs(k, 0) == 0
    assert integral_from_zero(1 / (2 - k), k, variables) is None  # not an exponential polynomial
    assert integral_from_zero(sympy.exp(k * x), k, variables) is None  # the rate is not a number


def test_integral_from_zero_rejects_a_wrong_proposal(monkeypatch):
    # the formula is only a proposal: a wrong antiderivative fails the derivative check
    monkeypatch.setattr(exp_polynomial, "_integral_of_term", lambda n, c, var: 2 * var ** (n + 1))
    assert integral_from_zero(sympy.exp(k / 2), k, [k]) is None


def test_analytic_at_zero():
    assert analytic_at_zero(sympy.exp(k / 2) / (2 - x), [k, x])
    assert analytic_at_zero(sympy.log(1 - k), [k])
    assert not analytic_at_zero(sympy.log(k), [k])
    assert not analytic_at_zero((sympy.exp(k) - 1) / k, [k])  # removable singularity: not by construction
    assert not analytic_at_zero(1 / (1 - sympy.Symbol("p")), [k])  # p = 1 is possible


@pytest.mark.parametrize("expr, expected", [
    ("(k/2 - 1)*exp(k/2) + 1", True),
    ("exp(k/2) - 1", True),
    ("exp(k/2) - k", False),  # coefficient of k: 1/2 - 1
    ("(k - 3)*exp(k)", False),  # coefficient (m - 3)/m!
    ("(k - 3)*exp(k) + 3 + 2*k + k^2/2", True),  # the first three coefficients cancel to 0
    ("(k^2 - 5*k + 7)*exp(k)", False),  # r(m) = m^2 - 6m + 7 < 0 for m = 3, 4: found below Cauchy's bound
    ("ok*exp(k/2)*(1 + stop/2) + (k/2 - 1)*exp(k/2) + 1", True),
    ("ok*exp(k/2) - stop", False),
    ("exp(-k)", None),  # negative rate: not decided
    ("exp(k/2)*exp(x)", None),  # two exponentials
    ("p*exp(k)", None),  # a parameter
    ("exp(k)/(2 - x)", None),  # not a polynomial in x
])
def test_exp_polynomial_nonneg(expr, expected):
    assert exp_polynomial_nonneg(sym(expr), ["k", "x", "ok", "stop"]) is expected


def test_entire_exp_polynomial():
    assert entire_exp_polynomial(sym("(k/2 - 1)*exp(k/2) + 1 + x"), ["k", "x"])
    assert not entire_exp_polynomial(sym("exp(k)/(2 - x)"), ["k", "x"])
    assert not entire_exp_polynomial(sym("1/(1 - k)"), ["k"])


def test_cancelled_rational_difference_is_proven_different():
    assert decide_equal(sympy.exp(k) + 1, sympy.exp(k)) is False
    assert decide_equal(sympy.exp(k) + 1, sympy.exp(k) + 1) is True
