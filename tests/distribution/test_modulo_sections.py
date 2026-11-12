"""
Exact residue classes `x % q = r` of rational generating functions (issue #35): all engines compute them
without roots of unity, also with several variables and with parameters. The coefficients are compared
exactly with a series expansion of the input.
"""
import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.multisection import rational_sections
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl.parser import parse_expr

try:
    from prodigy.distribution.fast_generating_function import FPS
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    FPS = None

GINAC = pytest.param(FPS, id="GiNaC",
                     marks=pytest.mark.skipif(FPS is None, reason="prodigy_ginac is not installed"))
ENGINES = [pytest.param(GeneratingFunction, id="SymPy"), pytest.param(SymengineDist, id="SymEngine"), GINAC]

X, Y, P, L = sympy.symbols("x y p l")
DEGREE = 12
MODULI = [2, 3, 5, 6]
# (generating function, variables); all other symbols are parameters
RATIONAL = [
    ("1/2/(1-1/2*x)", ("x",)),
    ("p/(1 - (1 - p)*x)", ("x",)),
    ("(1 + x^2)/((2 - x)*(3 - x))", ("x",)),
    ("(x + p)/((2 - x)*(1 + p*x^2))/(1 + 1/p)", ("x",)),
    ("x*y/((2 - x)*(3 - y))", ("x", "y")),
    ("1/(4 - x - y - x*y)", ("x", "y")),
    ("p*y/(1 - (1 - p)*x*y)", ("x", "y")),
    # coefficients that are not rational functions of the parameters
    ("(1 - exp(-l))/(1 - exp(-l)*x)", ("x",)),
    ("sqrt(p)*y/(2 - sqrt(p)*x*y)", ("x", "y")),
]


def _sympy(expression) -> sympy.Expr:
    text = str(expression).replace("**", "^")
    return sympy.parse_expr(text, local_dict={"x": X, "y": Y, "p": P, "l": L, "I": sympy.I, "Pi": sympy.pi},
                            transformations=standard_transformations + (convert_xor,))


def _coefficients(expression: sympy.Expr):
    series = sympy.expand(sympy.series(expression, X, 0, DEGREE).removeO())
    return [series.coeff(X, i) for i in range(DEGREE)]


def _assert_rational(expression: sympy.Expr):
    assert not expression.has(sympy.I), expression
    assert not expression.has(sympy.pi), expression
    assert expression.is_rational_function(X), expression


def _assert_equal(actual: sympy.Expr, expected: sympy.Expr):
    difference = sympy.cancel(actual - expected)
    assert difference == 0 or sympy.simplify(difference) == 0, f"{actual} != {expected}"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression, variables", RATIONAL)
@pytest.mark.parametrize("modulus", MODULI)
def test_sections_are_exact_and_rational(engine, expression, variables, modulus):
    function = _sympy(expression)
    expected = _coefficients(function)
    total = 0
    for remainder in range(modulus):
        result = engine(expression, *variables).filter(parse_expr(f"x % {modulus} = {remainder}"))
        assert result.get_variables() == set(variables)
        section = _sympy(result)
        _assert_rational(section)
        for i, coefficient in enumerate(_coefficients(section)):
            _assert_equal(coefficient, expected[i] if i % modulus == remainder else 0)
        total += section
    _assert_equal(total, function)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression, variables", [RATIONAL[1], RATIONAL[4]])
def test_modulo_assignment_is_rational(engine, expression, variables):
    # z := x % 3 splits the infinite marginal of x into its residue classes
    result = engine(expression, *variables, "z").update(parse_expr("z = x % 3"))
    function = _sympy(result)
    assert not function.has(sympy.I) and not function.has(sympy.pi)
    for remainder in range(3):
        probability = result.filter(parse_expr(f"z = {remainder}")).get_probability_mass()
        section = engine(expression, *variables).filter(parse_expr(f"x % 3 = {remainder}"))
        _assert_equal(_sympy(probability), _sympy(section.get_probability_mass()))


@pytest.mark.parametrize("engine", ENGINES)
def test_large_modulus(engine):
    expression = "p/(1 - (1 - p)*x)"
    result = engine(expression, "x").filter(parse_expr("x % 60 = 7"))
    section = _sympy(result)
    _assert_rational(section)
    _assert_equal(section, P * (1 - P) ** 7 * X ** 7 / (1 - (1 - P) ** 60 * X ** 60))


def test_rational_sections_rejects_functions_that_are_not_rational():
    assert rational_sections(sympy.exp(L * (X - 1)), X, 2) is None
    assert rational_sections(sympy.sqrt(1 - X), X, 2) is None
    assert rational_sections(X ** P, X, 2) is None


def test_roots_of_unity_constants_are_not_parameters():
    """After a modulo guard on a variable with infinite support, I and pi are constants, not parameters (#342)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.analyzer import compute_discrete_distribution
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.pgcl import parse_pgcl
    program = parse_pgcl("x: nat;\ny: nat;\nx = poisson(2);\nif (x % 3 == 1) {\n    y = 1;\n}\nquery Ex[x];\n", "ppy")
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.SYMPY)
    result, _ = compute_discrete_distribution(program, config.factory.one(*program.variables), config)
    assert result.get_parameters() == set()
