"""
Modulo filters `x % q = r` on distributions with a single variable (issue #124), compared with the series
expansion on SymPy and GiNaC.
"""
import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.multisection import rational_multisection
from prodigy.distribution.roots_of_unity import NotSingleValuedError
from prodigy.language.ppy_parser import parse_ppy_expr

try:
    from prodigy.distribution.fast_generating_function import FPS
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    FPS = None

GINAC = pytest.param(FPS, id="GiNaC",
                     marks=pytest.mark.skipif(FPS is None, reason="prodigy_ginac is not installed"))
ENGINES = [pytest.param(GeneratingFunction, id="SymPy"), GINAC]

X, P = sympy.symbols("x p")
DEGREE = 10
GUARDS = [(2, 0), (2, 1), (3, 0), (3, 2), (4, 1), (5, 3)]
SINGLE_VALUED = [
    "1/(2 - x)",
    "x/(3 - 2*x)^2",
    "(1 + x)^4/16",
    "x^5",
    "(1 + x^2)/((2 - x)*(3 - x))",
    # entire, not rational: the roots of unity are substituted
    "exp(3*(x - 1))",
]
MULTI_VALUED = [
    "sqrt(x^2)",
    "(1 - sqrt(1 - x^2))/x",
]


def _sympy(expression) -> sympy.Expr:
    text = str(expression).replace("**", "^")
    return sympy.parse_expr(text, local_dict={"x": X, "p": P, "I": sympy.I, "Pi": sympy.pi},
                            transformations=standard_transformations + (convert_xor,))


def _assert_section(result, expression, keep):
    """
    Pr[x = i] in `result` is the coefficient of x^i in `expression` if keep(i) and 0 otherwise, for i < DEGREE.
    The probabilities come from the engine itself and are compared to 40 digits: results with roots of unity
    are too slow for exact series expansion in SymPy.
    """
    series = sympy.series(_sympy(expression), X, 0, DEGREE).removeO()
    for i in range(DEGREE):
        expected = series.coeff(X, i) if keep(i) else 0
        actual = _sympy(result.filter(parse_ppy_expr(f"x == {i}")).get_probability_mass())
        assert abs(sympy.N(actual - expected, 40)) < 1e-30, f"Pr[x = {i}] is {actual}, expected {expected}"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", SINGLE_VALUED)
@pytest.mark.parametrize("modulus, remainder", GUARDS)
def test_modulo_guard_matches_series(engine, expression, modulus, remainder):
    result = engine(expression, "x").filter(parse_ppy_expr(f"x % {modulus} == {remainder}"))
    _assert_section(result, expression, lambda i: i % modulus == remainder)


@pytest.mark.parametrize("engine", ENGINES)
def test_modulo_guard_probability(engine):
    # x ~ Geometric(1/2) on 0, 1, 2, ...: x is even with probability 2/3
    result = engine("1/(2 - x)", "x").filter(parse_ppy_expr("x % 2 == 0"))
    assert sympy.nsimplify(_sympy(result.get_probability_mass())) == sympy.Rational(2, 3)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", MULTI_VALUED)
@pytest.mark.parametrize("guard", ["x % 2 == 0", "x % 2 == 1", "x % 3 == 2"])
def test_modulo_guard_on_multi_valued_function_is_rejected(engine, expression, guard):
    if engine is FPS and expression == "sqrt(x^2)":
        pytest.skip("GiNaC parses sqrt(x^2) as x, which is single-valued")
    with pytest.raises(NotSingleValuedError, match="not provably single-valued in 'x'"):
        engine(expression, "x").filter(parse_ppy_expr(guard))


@pytest.mark.parametrize("expression", ["1/(2 - x)", "x/(3 - 2*x)^2", "(1 + x)^4/16", "x^7", "3", "1/x",
                                        "p/(1 - (1 - p)*x)", "(x + p)/((2 - x)*(1 + p*x^2))"])
@pytest.mark.parametrize("modulus", [1, 2, 3, 4, 5])
def test_rational_multisection(expression, modulus):
    function = _sympy(expression)
    sections = rational_multisection(function, X, modulus)
    assert len(sections) == modulus
    assert sympy.cancel(sum(sections) - function) == 0
    for residue, section in enumerate(sections):
        assert section.is_rational_function(X)
        # the section only has exponents congruent to residue: f_r(w x) = w^r f_r(x), checked at a point
        omega = sympy.exp(2 * sympy.pi * sympy.I / modulus)
        point = {X: sympy.Rational(1, 5), P: sympy.Rational(2, 7)}
        shifted = section.subs(X, omega * X).subs(point) - omega ** residue * section.subs(point)
        assert abs(sympy.N(shifted, 40)) < 1e-30


@pytest.mark.parametrize("expression, modulus", [("exp(x)", 2), ("1/(2 - x)", 0)])
def test_rational_multisection_rejects_invalid_input(expression, modulus):
    with pytest.raises(ValueError):
        rational_multisection(_sympy(expression), X, modulus)
