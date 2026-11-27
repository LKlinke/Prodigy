"""
Comparisons of two variables (`x < n + c`, `x = n + c`, ...) and monus `n - x` on generating
functions with infinite mass, read as formal power series (Prodigy#366): e.g. the second-order
distribution of the equivalence check, or an uninitialised `n`. The filter is linear on the
coefficients, so it needs no finite mass; the case split over the values of a variable does, and
used to refuse these guards ("not a finite measure"). Every result is checked coefficient by
coefficient against the filtered series (brute force on the truncated series), on all engines.
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution import NonFiniteMarginalError
from prodigy.distribution.pgf.rules.independent_comparison import ComparisonNotImplemented
from prodigy.language.ppy_parser import parse_ppy_expr

Engine = ForwardAnalysisConfig.Engine
ENGINES = [Engine.SYMPY, Engine.GINAC]

x, n, a, b, w, z, t = sympy.symbols("x n a b w z t")
ORDER = 6

# (generating function, variables); every one has infinite mass
FUNCTIONS = [
    # the second-order distribution of the equivalence check for x and n (a, b: second-order variables)
    pytest.param("1/((1 - a*x)*(1 - b*n))", ("x", "n", "a", "b"), id="second-order"),
    # x and n both uninitialised over all values with weight 1
    pytest.param("1/((1 - x)*(1 - n))", ("x", "n"), id="uniform-weights"),
    # after the guard n > 0 and with another variable w of infinite mass
    pytest.param("n/((1 - x/2)*(1 - n)*(1 - w))", ("x", "n", "w"), id="product-other-variable"),
    # a signed measure with infinite mass
    pytest.param("(1 - 2*x)/((1 - x)^2*(1 - n))", ("x", "n"), id="signed"),
    # dependent x and n: the residue method, rational residues only
    pytest.param("1/((1 - x)*(1 - n)*(1 - n*x))", ("x", "n"), id="dependent"),
    pytest.param("1/((1 - x)*(1 - x*n/2 - n/2))", ("x", "n"), id="dependent-linear-kernel"),
]

# guard -> predicate on the exponents (i of x, j of n)
GUARDS = [
    ("x < n", lambda i, j: i < j),
    ("x <= n + 1", lambda i, j: i <= j + 1),
    ("x == n", lambda i, j: i == j),
    ("n + 2 == x", lambda i, j: i == j + 2),
    ("x > n", lambda i, j: i > j),
    ("x >= n + 2", lambda i, j: i >= j + 2),
    ("n < x + 1", lambda i, j: j < i + 1),
    ("not (x < n)", lambda i, j: not i < j),
    ("x > 0 and x < n", lambda i, j: 0 < i < j),
]


def _config(engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine)


def _sympy(dist) -> sympy.Expr:
    return sympy.sympify(str(dist).replace("^", "**"))


def _terms(function: sympy.Expr, variables) -> dict:
    """The coefficients of the monomials of total degree < ORDER (brute force: series in t)."""
    symbols = [sympy.Symbol(v) for v in variables]
    scaled = function.subs({s: t * s for s in symbols}, simultaneous=True)
    series = sympy.series(scaled, t, 0, ORDER).removeO()
    poly = sympy.Poly(sympy.expand(series.subs(t, 1)), *symbols)
    return {monomial: coefficient for monomial, coefficient in poly.terms() if coefficient != 0}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function, variables", FUNCTIONS)
@pytest.mark.parametrize("guard, holds", GUARDS)
def test_filter_coefficients(engine, function, variables, guard, holds):
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(function, *variables)
        filtered = dist.filter(parse_ppy_expr(guard))
    full = _terms(sympy.sympify(function.replace("^", "**")), variables)
    expected = {m: c for m, c in full.items() if holds(m[0], m[1])}
    assert _terms(_sympy(filtered), variables) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function, variables", FUNCTIONS)
def test_monus_coefficients(engine, function, variables):
    # z := n - x (truncated at 0): the term x^i n^j moves to z^max(j - i, 0)
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(function, *variables, "z")
        result = dist.update(parse_ppy_expr("z == n - x"))
    full = _terms(sympy.sympify(function.replace("^", "**")), variables)
    expected: dict = {}
    for monomial, coefficient in full.items():
        key = (*monomial, max(monomial[1] - monomial[0], 0))
        expected[key] = expected.get(key, 0) + coefficient
    # every term of degree < ORDER in the input variables, whatever the power of z
    symbols = [sympy.Symbol(v) for v in variables]
    scaled = _sympy(result).subs({s: t * s for s in symbols}, simultaneous=True)
    series = sympy.series(scaled, t, 0, ORDER).removeO()
    poly = sympy.Poly(sympy.expand(series.subs(t, 1)), *symbols, z)
    assert {m: c for m, c in poly.terms() if c != 0} == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_closed_form_on_the_second_order_distribution(engine):
    # sum over i < j of x^i n^j = n / ((1 - n) (1 - x n))
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("1/((1 - x)*(1 - n))", "x", "n")
        below = dist.filter(parse_ppy_expr("x < n"))
    assert sympy.simplify(_sympy(below) - n / ((1 - n) * (1 - x * n))) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_unproven_branch_stays_not_implemented(engine):
    # dependent x and n with the radicand 1 - 4 x n: the formal identity holds, but the principal
    # square root is not proven to be the right branch where the result is evaluated (|x|, |n| <= 1)
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("1/(1 - x - n)", "x", "n")
        with pytest.raises(ComparisonNotImplemented, match="branch"):
            dist.filter(parse_ppy_expr("x < n"))


@pytest.mark.parametrize("engine", ENGINES)
def test_other_guards_still_need_a_finite_measure(engine):
    # x < 2*n is no comparison of the form x < n + c: the case split over the values of n is still
    # refused, as the marginal of n is not a finite measure
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("1/((1 - x)*(1 - n))", "x", "n")
        with pytest.raises(NonFiniteMarginalError, match="not a finite measure"):
            dist.filter(parse_ppy_expr("x < 2 * n"))
