"""
Exact filters x < y + c and x = y + c for products of rational functions (Prodigy#243), checked
against the power series coefficient by coefficient and on all engines.
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.independent_comparison import ComparisonNotImplemented, filter_difference
from prodigy.pgcl.parser import parse_expr

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

Engine = ForwardAnalysisConfig.Engine
ENGINES = [pytest.param(Engine.SYMPY, id="SymPy"), pytest.param(Engine.SYMENGINE, id="SymEngine"),
           pytest.param(Engine.GINAC, id="GiNaC",
                        marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed"))]

x, y, r, p = sympy.symbols("x y r p")
HALF, THIRD = sympy.Rational(1, 2), sympy.Rational(1, 3)


def _geometric(q, variable):
    """Pr[k] = (1 - q) q^k"""
    return (1 - q) / (1 - q * variable)


FUNCTIONS = [
    pytest.param(_geometric(HALF, x) * _geometric(THIRD, y), id="geometric-geometric"),
    pytest.param(_geometric(HALF, x) * _geometric(THIRD, y) ** 2, id="geometric-negative-binomial"),
    pytest.param((1 + x ** 2) / 2 * _geometric(sympy.Rational(1, 4), y) * (y ** 3 + 1) / 2, id="polynomial-part"),
    pytest.param(_geometric(p, x) * _geometric(THIRD, y), id="parametric"),
    pytest.param(_geometric(HALF, x) * (1 + 2 * y + y ** 3) / 4, id="finite-y"),
    pytest.param((x + x ** 2) / 2 * (y + 3 * y ** 2) / 4, id="finite-both"),
    # the other variable r as a coefficient: u := x, v := y with independent x and y
    pytest.param(1 / ((x * r - 2) * (2 * y - 3)) * (1 + r) / (2 - r * y), id="other-variable"),
    # a signed measure (negative coefficients, as candidate invariants have)
    pytest.param((1 - x) * (1 + y) / ((1 - x / 2) * (1 - y / 3)), id="signed"),
    # a measure with infinite mass
    pytest.param(1 / ((1 - x) * (1 - y)), id="infinite-mass"),
]
POINT = {p: sympy.Rational(2, 5), r: THIRD}


def _coefficients(function, order):
    series = sympy.series(sympy.series(function, x, 0, order).removeO(), y, 0, order).removeO()
    poly = sympy.Poly(sympy.expand(series), x, y)
    return {(i, j): poly.coeff_monomial(x ** i * y ** j) for i in range(order) for j in range(order - i)}


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("relation", ["ge", "eq"])
@pytest.mark.parametrize("shift", [-2, -1, 0, 1, 3])
def test_coefficients(function, relation, shift):
    filtered = filter_difference(function, "x", "y", relation, shift, "guard")
    assert not (filtered.free_symbols - function.free_symbols)
    order = 6
    expected = _coefficients(function.subs(POINT), order)
    actual = _coefficients(filtered.subs(POINT), order)
    for (i, j), coefficient in expected.items():
        keep = j - i >= shift if relation == "ge" else j - i == shift
        assert sympy.simplify(actual[(i, j)] - (coefficient if keep else 0)) == 0, (i, j)


def test_closed_forms():
    # x geometric (Pr[i] = 2^-(i+1)), y geometric (Pr[j] = (2/3) 3^-j): Pr[x < y] = 1/5
    function = _geometric(HALF, x) * _geometric(THIRD, y)
    below = filter_difference(function, "x", "y", "ge", 1, "x < y")
    assert sympy.simplify(below - 2 * y / ((3 - y) * (6 - x * y))) == 0
    assert below.subs({x: 1, y: 1}) == sympy.Rational(1, 5)
    # Pr[x = y] = sum_i 2^-(i+1) (2/3) 3^-i = 2/5
    assert filter_difference(function, "x", "y", "eq", 0, "x = y").subs({x: 1, y: 1}) == sympy.Rational(2, 5)
    # parametric: x with Pr[i] = (1-p) p^i, y with Pr[j] = (1/2)^(j+1): Pr[x < y] = (1-p) / (2-p)
    parametric = _geometric(p, x) * _geometric(HALF, y)
    probability = filter_difference(parametric, "x", "y", "ge", 1, "x < y").subs({x: 1, y: 1})
    assert sympy.simplify(probability - (1 - p) / (2 - p)) == 0


def test_counterexample_of_the_issue_is_not_implemented():
    # the part with x < y of (1/2) / (1 - x/4 - y/4) is algebraic, not rational
    with pytest.raises(ComparisonNotImplemented, match="not independent"):
        filter_difference(HALF / (1 - x / 4 - y / 4), "x", "y", "ge", 1, "x < y")


@pytest.mark.parametrize("function, reason", [
    pytest.param(sympy.exp(x - 1) * _geometric(HALF, y), "not rational", id="poisson"),
    pytest.param(sympy.sqrt(1 - y) * _geometric(HALF, x), "not rational", id="algebraic"),
    pytest.param(1 / (1 - x * y / 2), "not independent", id="equal"),
])
def test_outside_the_class(function, reason):
    with pytest.raises(ComparisonNotImplemented, match=reason):
        filter_difference(function, "x", "y", "ge", 1, "x < y")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("guard, probability", [
    ("x < y", sympy.Rational(1, 5)), ("x <= y", sympy.Rational(3, 5)), ("x > y", sympy.Rational(2, 5)),
    ("x >= y", sympy.Rational(4, 5)), ("x = y", sympy.Rational(2, 5)), ("y + 1 = x", sympy.Rational(1, 5)),
    ("x < y + 2", sympy.Rational(4, 5)), ("2*x < 2*y + 1", sympy.Rational(3, 5)),
    ("2*x = 2*y + 1", 0), ("not (x < y)", sympy.Rational(4, 5)),
])
def test_engines(engine, guard, probability):
    factory = ForwardAnalysisConfig(engine=engine).factory
    dist = factory.from_expr(str(_geometric(HALF, x) * _geometric(THIRD, y)).replace("**", "^"), "x", "y")
    filtered = dist.filter(parse_expr(guard))
    assert sympy.nsimplify(sympy.sympify(str(filtered.get_probability_mass()))) == probability
    # the filtered part and its complement give the distribution back
    rest = dist.filter(parse_expr(f"not ({guard})"))
    assert filtered + rest == dist


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_counterexample(engine):
    factory = ForwardAnalysisConfig(engine=engine).factory
    dist = factory.from_expr("(1/2)/(1 - x/4 - y/4)", "x", "y")
    with pytest.raises(ComparisonNotImplemented, match="not independent"):
        dist.filter(parse_expr("x < y"))
    with pytest.raises(ComparisonNotImplemented, match="of the form"):
        dist.filter(parse_expr("2*x < y"))


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_filtered_part_is_no_product(engine):
    # the part with x < y is no product any more: a second comparison on it is not implemented
    factory = ForwardAnalysisConfig(engine=engine).factory
    dist = factory.from_expr(str(_geometric(HALF, x) * _geometric(THIRD, y)).replace("**", "^"), "x", "y")
    with pytest.raises(ComparisonNotImplemented, match="not independent"):
        dist.filter(parse_expr("x < y || y < x"))
