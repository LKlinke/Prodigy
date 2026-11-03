"""
Guards with constants that are not natural numbers (issue #112) are evaluated exactly and in the
same way on every engine.
"""
from fractions import Fraction
from itertools import product

import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

import prodigy.pgcl as pgcl
from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.fast_generating_function import FPS
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.linear_guards import normalize_guard
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl.parser import parse_expr

ENGINES = [GeneratingFunction, FPS, SymengineDist]
X, Y = sympy.Symbol('x'), sympy.Symbol('y')
DEGREE = 12

# (guard, the values of x for which it holds); every guard is constant for x >= DEGREE
SINGLE_VARIABLE = [
    ("1/2*x < 2", lambda x: x < 4),
    ("2 > 1/2*x", lambda x: x < 4),
    ("x <= 3/2", lambda x: x <= 1),
    ("3/2 >= x", lambda x: x <= 1),
    ("1/2*x = 1", lambda x: x == 2),
    ("1 = x/2", lambda x: x == 2),
    ("x = 3/2", lambda x: False),
    ("2 < 1/2*x", lambda x: x > 4),
    ("x > 3/2", lambda x: x >= 2),
    ("3/2 <= x", lambda x: x >= 2),
    ("x >= 0.5", lambda x: x >= 1),
    ("0.5*x <= 1", lambda x: x <= 2),
    ("x + 1/2 < 3", lambda x: x <= 2),
    ("x/2 < 2", lambda x: x < 4),
    # rational, not floor division: x = 3 does not satisfy it
    ("x/2 <= 1", lambda x: x <= 2),
    ("2*x < 8", lambda x: x < 4),
    ("1/3*x + 1/6*x <= 1", lambda x: x <= 2),
    ("x < 1/2", lambda x: x == 0),
    ("x < 0.5 + 0.5", lambda x: x == 0),
    ("x + 1 > 1/2", lambda x: True),
]


def _to_sympy(distribution) -> sympy.Expr:
    return sympy.parse_expr(str(distribution), local_dict={'x': X, 'y': Y},
                            transformations=standard_transformations + (convert_xor,))


def _coefficients(function: sympy.Expr) -> list:
    series = sympy.series(function, X, 0, DEGREE).removeO()
    return [sympy.nsimplify(sympy.simplify(series.coeff(X, k))) for k in range(DEGREE)]


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard,holds', SINGLE_VARIABLE)
def test_single_variable_on_infinite_distribution(engine, guard, holds):
    # geometric distribution, P(x = k) = 1/2^(k+1)
    result = _to_sympy(engine('1/(2-x)', 'x').filter(parse_expr(guard)))
    expected = [sympy.Rational(1, 2**(k + 1)) if holds(k) else 0 for k in range(DEGREE)]
    assert _coefficients(result) == expected
    tail = sympy.Rational(1, 2**DEGREE) if holds(DEGREE) else 0
    assert sympy.simplify(result.subs(X, 1) - sum(expected) - tail) == 0


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard,holds', SINGLE_VARIABLE)
def test_single_variable_on_finite_distribution(engine, guard, holds):
    values = [0, 1, 2, 3, 5]
    function = " + ".join(f"1/5*x^{k}" for k in values)
    result = sympy.expand(_to_sympy(engine(function, 'x').filter(parse_expr(guard))))
    assert result == sum((sympy.Rational(1, 5) * X**k for k in values if holds(k)), sympy.Integer(0))


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard,holds', [
    ("1/2*x + 1/3*y < 2", lambda x, y: Fraction(x, 2) + Fraction(y, 3) < 2),
    ("x/2 = y/3", lambda x, y: 3 * x == 2 * y),
    ("1/2*x >= 1/3*y + 1/6", lambda x, y: Fraction(x, 2) >= Fraction(y, 3) + Fraction(1, 6)),
    ("0.5*x + 0.5*y <= 1.5", lambda x, y: x + y <= 3),
])
def test_several_variables(engine, guard, holds):
    states = list(product(range(4), range(4)))
    function = " + ".join(f"1/16*x^{x}*y^{y}" for x, y in states)
    result = sympy.expand(_to_sympy(engine(function, 'x', 'y').filter(parse_expr(guard))))
    assert result == sum((sympy.Rational(1, 16) * X**x * Y**y for x, y in states if holds(x, y)),
                         sympy.Integer(0))


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard', ["x*y < 1/2", "x % 2 = 1/2", "x^(1/2) <= 1", "x - 1/2 < 1"])
def test_non_linear_guards_are_rejected(engine, guard):
    with pytest.raises(ValueError, match='only supported in linear guards'):
        engine('1/4*x + 1/4*x^2*y + 1/2*y^3', 'x', 'y').filter(parse_expr(guard))


def test_normalized_guards():
    variables = {'x', 'y'}
    assert str(normalize_guard(parse_expr("1/2*x < 2"), variables)) == "x < 4"
    assert str(normalize_guard(parse_expr("2 < 1/2*x"), variables)) == "x > 4"
    assert str(normalize_guard(parse_expr("x = 3/2"), variables)) == "false"
    assert str(normalize_guard(parse_expr("1/2*x + 1/3*y < 1"), variables)) == "((3 * x) + (2 * y)) < 6"
    # nothing to do for guards with natural numbers and several variables, or non-linear ones
    assert normalize_guard(parse_expr("x + 1 < y"), variables) is None
    assert normalize_guard(parse_expr("x % 2 = 0"), variables) is None
    assert normalize_guard(parse_expr("x * y < 2"), variables) is None


@pytest.mark.parametrize('engine', list(ForwardAnalysisConfig.Engine))
def test_program_from_issue(engine):
    program = pgcl.parse_pgcl("""
        nat x;
        x := geometric(1/2);
        if (1/2*x < 2) { x := 0 } else { x := 1 }
    """)
    config = ForwardAnalysisConfig(engine=engine)
    result, error_prob = compute_discrete_distribution(program, config.factory.one('x'), config)
    assert sympy.simplify(_to_sympy(result) - (sympy.Rational(15, 16) + X / 16)) == 0
    assert sympy.S(str(error_prob)) == 0
