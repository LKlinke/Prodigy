"""
``x = unif(0, n)`` for a program variable n with infinitely many values (#385), computed with the operator
``1/(θ + 1)`` in n (:func:`prodigy.analysis.instructionhandler.counter_operators.uniform_up_to`). The example
``pgfexamples/inference/loop_free/uniform_geometric_bound.ppy`` is also compared against the simulator
(``tests/simulation/test_examples.py``).
"""
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.counter_operators import uniform_up_to
from prodigy.analysis.values import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution import MarginalType
from prodigy.distribution.equality import decide_equal, sympify_distribution
from prodigy.language import Program, parse_program
from prodigy.language.compiler import compile_program

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]

EXAMPLE = Path(__file__).parents[2] / "pgfexamples/inference/loop_free/uniform_geometric_bound.ppy"

n, x, y = sympy.symbols("n x y")


def _distribution(source: str, engine):
    program = compile_program(source, syntax="ppy")
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine, normalize=False)
    with config.activate():
        result, _ = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    return result, config


def _session(source: str, engine):
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_program(source, "ppy"), source=source, config=config,
                              decisions={}, capture_output=True)
    return session.run().to_json()


def _equal(value, expected) -> bool:
    parsed = sym(value)
    parsed = parsed.xreplace({s: sympy.Symbol(s.name) for s in parsed.free_symbols})  # without assumptions
    return decide_equal(parsed, sympy.S(expected)) is True


def _coefficients(expr: sympy.Expr, order: int) -> dict:
    """The coefficients of n^k x^j (k < order) of the power series of ``expr``."""
    series = sympy.series(expr, n, 0, order).removeO()
    return {(k, j): sympy.nsimplify(sympy.series(series.coeff(n, k), x, 0, order).removeO().coeff(x, j))
            for k in range(order) for j in range(order)}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("prefix", [
    pytest.param("n: nat = unif(0, 3);\nx: nat = 0;\n", id="uniform"),
    pytest.param("n: nat = binomial(4, 1/3);\ny: nat = bernoulli(1/4);\nx: nat = 0;\n", id="binomial-and-other-variable"),
    pytest.param("n: nat = unif(1, 2);\nx: nat = 5;\n", id="x-assigned-before"),
])
def test_finite_bound_agrees_with_the_case_split(engine, prefix):
    before, config = _distribution(prefix, engine)
    split, _ = _distribution(prefix + "x = unif(0, n);", engine)
    with config.activate():
        operator = uniform_up_to(before, "x", "n", "unif(0, n)")
    assert operator.get_variables() == split.get_variables()
    assert decide_equal(sympify_distribution(operator), sympify_distribution(split)) is True


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("generating_function, expected", [
    # the second-order PGF of n (the uninitialised input of an equivalence check)
    pytest.param("1/(1-n)", (sympy.log(1 - x * n) - sympy.log(1 - n)) / (n * (1 - x)), id="sum-of-powers"),
    # geometric(1/2)
    pytest.param("1/(2-n)", (sympy.log(2 - x * n) - sympy.log(2 - n)) / (n * (1 - x)), id="geometric"),
])
def test_infinite_bound_has_the_closed_form(engine, generating_function, expected):
    config = ForwardAnalysisConfig(engine=engine)
    with config.activate():
        result = uniform_up_to(config.factory.from_expr(generating_function, "n", "x"), "x", "n", "unif(0, n)")
    computed = sympify_distribution(result)
    assert decide_equal(computed, expected) is True
    # term by term: n^k (1 + x + … + x^k)/(k + 1) times the coefficient of n^k
    weights = _coefficients(sympy.S(generating_function).subs(sympy.Symbol("n"), n), 5)
    for (k, j), coefficient in _coefficients(computed, 5).items():
        assert coefficient == (weights[(k, 0)] / (k + 1) if j <= k else 0), (k, j)


@pytest.mark.parametrize("engine", ENGINES)
def test_poisson_bound_with_another_variable(engine):
    # the integral of an exponential polynomial is given by a formula; y stays untouched
    result, _ = _distribution("n: nat = poisson(2);\ny: nat = geometric(1/3);\nx: nat = unif(0, n);", engine)
    expected = (sympy.exp(2 * n) - sympy.exp(2 * n * x)) * sympy.exp(-2) / (2 * n * (1 - x) * (3 - 2 * y))
    assert decide_equal(sympify_distribution(result), expected) is True


@pytest.mark.parametrize("engine", ENGINES)
def test_the_example_is_exact(engine):
    result = _session(EXAMPLE.read_text(encoding="utf-8"), engine)
    assert result["status"] == "ok", result.get("error")
    expected = [sympy.log(2), sympy.log(2) - sympy.Rational(1, 2), sympy.Rational(1, 2), sympy.Rational(1, 24)]
    for query, value in zip(result["queries"], expected, strict=True):
        assert query["bound"] == "exact"
        assert _equal(query["value"], value), (query["value"], value)


@pytest.mark.parametrize("engine", ENGINES)
def test_bound_zero_gives_zero(engine):
    # unif(0, 0) is x = 0: the division by n is exact at the term n^0 (a removable singularity of the closed form)
    source = ("n: nat = geometric(1/2);\nx: nat = unif(0, n);\n"
              "query Pr[n == 0 and x == 0];\nquery Pr[n == 0 and x == 1];\nquery Pr[n == 0];")
    result = _session(source, engine)
    assert result["status"] == "ok", result.get("error")
    for query, value in zip(result["queries"], ["1/2", "0", "1/2"], strict=True):
        assert query["bound"] == "exact"
        assert _equal(query["value"], value), (query["value"], value)


# The closed form has a removable singularity at x = 1. SymPy computes the value there; GiNaC's
# substitution divides by zero (it cancels removable singularities of rational functions only), so the analysis
# ends with an error, never with a wrong result. The fix belongs in the bindings.
ENGINES_AT_X_ONE = [
    ENGINES[0],
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=[pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed"),
                        pytest.mark.xfail(raises=(ArithmeticError, ValueError), strict=True,
                                          reason="GiNaC divides by zero at a removable singularity of a logarithm")]),
]


@pytest.mark.parametrize("engine", ENGINES_AT_X_ONE)
def test_marginal_of_the_bound_is_unchanged(engine):
    result, config = _distribution("n: nat = geometric(1/2);\nx: nat = unif(0, n);", engine)
    with config.activate():
        marginal = result.marginal("n", method=MarginalType.INCLUDE)
    assert decide_equal(sympify_distribution(marginal), 1 / (2 - n)) is True


@pytest.mark.parametrize("engine", ENGINES_AT_X_ONE)
def test_assignment_to_the_uniform_variable(engine):
    result, _ = _distribution("n: nat = geometric(1/2);\nx: nat = unif(0, n);\nx = 0;", engine)
    assert decide_equal(sympify_distribution(result), 1 / (2 - n)) is True


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, message", [
    pytest.param("x: nat = unif(1, n);", "other bounds", id="lower-bound-1"),
    pytest.param("x: nat = unif(0, n + 1);", "other bounds", id="affine-upper-bound"),
    pytest.param("x: nat = unif(n, n);", "other bounds", id="variable-lower-bound"),
    pytest.param("n = unif(0, n);", "its own upper bound", id="own-bound"),
])
def test_other_forms_are_not_implemented(engine, source, message):
    with pytest.raises(NotImplementedError, match=message):
        _distribution("n: nat = geometric(1/2);\n" + source, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_uncertified_integral_is_not_implemented(engine):
    # with a parameter, SymPy's integrate is not used (it can take very long): no result instead of a guess
    source = "param p: real;\nn: nat = geometric(p);\nx: nat = unif(0, n);"
    with pytest.raises(NotImplementedError, match="needs the integral from 0"):
        _distribution(source, engine)
