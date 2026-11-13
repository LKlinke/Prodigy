"""
Exact filters x < y + c and x = y + c (Prodigy#243): for products of rational functions (rational
results) and by the residue method for dependent variables (algebraic results), checked against the
power series coefficient by coefficient and on all engines.
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.algebraic_comparison import filter_difference_algebraic
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


def test_counterexample_is_no_product():
    # the part with x < y of (1/2) / (1 - x/4 - y/4) is algebraic, not rational: no product either
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


# --- dependent variables: the residue method

COUNTEREXAMPLE = HALF / (1 - x / 4 - y / 4)
t = sympy.Symbol("t")


def _check_algebraic(function, relation, shift, variables=("x", "y"), point=None):
    """Compares the series of the filtered part along x = a t, y = b t with the filtered series of `function`."""
    point = point or {}
    filtered = filter_difference_algebraic(function, "x", "y", relation, shift, variables, "guard")
    order = 6
    for a, b in ((1, 1), (THIRD, 2)):
        series = sympy.series(filtered.subs({x: a * t, y: b * t} | point), t, 0, order).removeO()
        full = sympy.series(function.subs({x: t * x, y: t * y} | point), t, 0, order).removeO()
        for n in range(order):
            part = sympy.Poly(sympy.expand(full.coeff(t, n)), x, y)
            expected = sum((c * a ** i * b ** j for (i, j), c in part.terms()
                            if (j - i >= shift if relation == "ge" else j - i == shift)), sympy.Integer(0))
            assert sympy.simplify(series.coeff(t, n) - expected) == 0, (n, a, b)
    return filtered


def test_counterexample_of_the_issue_comment():
    # the part with x < y of (1/2) / (1 - x/4 - y/4) is algebraic: along x = y = t its coefficient of
    # t^(2m) is 1/4^(m+1) - binomial(2m, m) / (4 * 16^m) (the comment on #243)
    below = _check_algebraic(COUNTEREXAMPLE, "ge", 1)
    assert below.has(sympy.sqrt(4 - x * y)) or below.has(sympy.sqrt(1 - x * y / 4))
    series = sympy.series(below.subs({x: t, y: t}), t, 0, 12).removeO()
    for m in range(6):
        expected = sympy.Rational(1, 4 ** (m + 1)) - sympy.binomial(2 * m, m) / (4 * 16 ** m)
        assert sympy.simplify(series.coeff(t, 2 * m) - expected) == 0
    # Pr[x - y = k] = (2 - sqrt(3))^|k| / sqrt(3), so Pr[x < y] = (1 - 1/sqrt(3)) / 2
    assert sympy.simplify(below.subs({x: 1, y: 1}) - (1 - 1 / sympy.sqrt(3)) / 2) == 0


@pytest.mark.parametrize("function, variables, point", [
    pytest.param(COUNTEREXAMPLE, ("x", "y"), None, id="counterexample"),
    pytest.param(THIRD / (1 - x / 3 - y / 3 - x * y / 9), ("x", "y"), None, id="with-xy"),
    # another variable r and a second, rational root of the kernel
    pytest.param((1 - r / 2) / ((1 - x / 4 - y / 4) * (1 - r * x / 2)), ("x", "y", "r"), {r: HALF}, id="other-variable"),
    pytest.param(HALF / (1 - x / 2 - x * y / 4) * (1 + y) / 2, ("x", "y"), None, id="kernel-with-y-factor"),
])
@pytest.mark.parametrize("relation", ["ge", "eq"])
@pytest.mark.parametrize("shift", [-1, 0, 1, 2])
def test_algebraic_coefficients(function, variables, point, relation, shift):
    _check_algebraic(function, relation, shift, variables, point)


@pytest.mark.parametrize("function, reason", [
    # the kernel s - x/4 - y^2 s^3 / 4 is cubic
    pytest.param(HALF / (1 - x / 4 - y ** 2 / 4), "degree", id="cubic-kernel"),
    # infinite mass: the radicand 1 - x*y reaches 0 on the unit polydisk
    pytest.param(1 / (1 - x / 2 - y / 2), "branch", id="branch-not-proven"),
    pytest.param(p / (1 - p * x / 2 - p * y / 2), "parameters", id="parameter"),
    pytest.param(sympy.exp(x * y - 1), "not rational", id="not-rational"),
])
def test_algebraic_outside(function, reason):
    with pytest.raises(ComparisonNotImplemented, match=reason):
        filter_difference_algebraic(function, "x", "y", "ge", 1, ("x", "y"), "x < y")


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_counterexample(engine):
    factory = ForwardAnalysisConfig(engine=engine).factory
    dist = factory.from_expr("(1/2)/(1 - x/4 - y/4)", "x", "y")
    below = dist.filter(parse_expr("x < y"))
    assert not below.is_finite()
    value = sympy.sympify(str(below.get_probability_mass()).replace("^", "**"))
    assert sympy.simplify(value - (1 - 1 / sympy.sqrt(3)) / 2) == 0
    monus = dist.set_variables("x", "y", "z").update(parse_expr("z = x - y"))
    # Ex[z] = sum_k>=1 k (2 - sqrt(3))^k / sqrt(3) = 1 / (2 sqrt(3))
    value = sympy.sympify(str(monus.get_expected_value_of("z")).replace("^", "**"))
    assert abs(float(value) - 1 / (2 * 3 ** 0.5)) < 1e-12
    assert sympy.simplify(sympy.nsimplify(value, [sympy.sqrt(3)]) - 1 / (2 * sympy.sqrt(3))) == 0
    # the algebraic part is not rational: comparing two variables on it is not implemented
    with pytest.raises(ComparisonNotImplemented, match="not rational"):
        below.filter(parse_expr("x + 1 < y"))
    with pytest.raises(ComparisonNotImplemented, match="of the form"):
        dist.filter(parse_expr("2*x < y"))
    with pytest.raises(ComparisonNotImplemented, match="degree"):
        factory.from_expr("(1/2)/(1 - x/4 - y^2/4)", "x", "y").filter(parse_expr("x < y"))


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_second_comparison_on_a_filtered_part(engine):
    # the part with x < y is no product any more: its filter y < x goes through the residue method
    factory = ForwardAnalysisConfig(engine=engine).factory
    dist = factory.from_expr(str(_geometric(HALF, x) * _geometric(THIRD, y)).replace("**", "^"), "x", "y")
    either = dist.filter(parse_expr("x < y || y < x"))
    assert sympy.nsimplify(sympy.sympify(str(either.get_probability_mass()))) == sympy.Rational(3, 5)
