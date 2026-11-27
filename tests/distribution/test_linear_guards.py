"""
Guards with constants that are not natural numbers (issue #112) are evaluated exactly and in the
same way on every engine.
"""
from fractions import Fraction
from itertools import product

import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

import prodigy.language as language
from prodigy.analysis.session import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.pgf.ginac.engine import GinacPGF
from prodigy.distribution.pgf.sympy.engine import SympyPGF
from prodigy.distribution.pgf.rules.linear_guards import linear_form, normalize_guard
from prodigy.language.ppy_parser import parse_ppy_expr

ENGINES = [SympyPGF, GinacPGF]
X, Y = sympy.Symbol('x'), sympy.Symbol('y')
DEGREE = 12

# (guard, the values of x for which it holds); every guard is constant for x >= DEGREE
SINGLE_VARIABLE = [
    ("1/2*x < 2", lambda x: x < 4),
    ("2 > 1/2*x", lambda x: x < 4),
    ("x <= 3/2", lambda x: x <= 1),
    ("3/2 >= x", lambda x: x <= 1),
    ("1/2 * x == 1", lambda x: x == 2),
    ("1 == x / 2", lambda x: x == 2),
    ("x == 3/2", lambda x: False),
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
    result = _to_sympy(engine('1/(2-x)', 'x').filter(parse_ppy_expr(guard)))
    expected = [sympy.Rational(1, 2**(k + 1)) if holds(k) else 0 for k in range(DEGREE)]
    assert _coefficients(result) == expected
    tail = sympy.Rational(1, 2**DEGREE) if holds(DEGREE) else 0
    assert sympy.simplify(result.subs(X, 1) - sum(expected) - tail) == 0


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard,holds', SINGLE_VARIABLE)
def test_single_variable_on_finite_distribution(engine, guard, holds):
    values = [0, 1, 2, 3, 5]
    function = " + ".join(f"1/5*x^{k}" for k in values)
    result = sympy.expand(_to_sympy(engine(function, 'x').filter(parse_ppy_expr(guard))))
    assert result == sum((sympy.Rational(1, 5) * X**k for k in values if holds(k)), sympy.Integer(0))


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard,holds', [
    ("1/2*x + 1/3*y < 2", lambda x, y: Fraction(x, 2) + Fraction(y, 3) < 2),
    ("x / 2 == y / 3", lambda x, y: 3 * x == 2 * y),
    ("1/2*x >= 1/3*y + 1/6", lambda x, y: Fraction(x, 2) >= Fraction(y, 3) + Fraction(1, 6)),
    ("0.5*x + 0.5*y <= 1.5", lambda x, y: x + y <= 3),
])
def test_several_variables(engine, guard, holds):
    states = list(product(range(4), range(4)))
    function = " + ".join(f"1/16*x^{x}*y^{y}" for x, y in states)
    result = sympy.expand(_to_sympy(engine(function, 'x', 'y').filter(parse_ppy_expr(guard))))
    assert result == sum((sympy.Rational(1, 16) * X**x * Y**y for x, y in states if holds(x, y)),
                         sympy.Integer(0))


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('guard', ["x*y < 1/2", "x % 2 == 1/2", "x**(1/2) <= 1", "x - 1/2 < 1"])
def test_non_linear_guards_are_rejected(engine, guard):
    with pytest.raises(ValueError, match='only supported in linear guards'):
        engine('1/4*x + 1/4*x^2*y + 1/2*y^3', 'x', 'y').filter(parse_ppy_expr(guard))


def test_normalized_guards():
    variables = {'x', 'y'}
    assert str(normalize_guard(parse_ppy_expr("1/2 * x < 2"), variables)) == "x < 4"
    assert str(normalize_guard(parse_ppy_expr("2 < 1/2 * x"), variables)) == "x > 4"
    assert str(normalize_guard(parse_ppy_expr("x == 3/2"), variables)) == "false"
    assert str(normalize_guard(parse_ppy_expr("1/2 * x + 1/3 * y < 1"), variables)) == "3 * x + 2 * y < 6"
    # nothing to do for guards with natural numbers and several variables, or non-linear ones
    assert normalize_guard(parse_ppy_expr("x + 1 < y"), variables) is None
    assert normalize_guard(parse_ppy_expr("x % 2 == 0"), variables) is None
    assert normalize_guard(parse_ppy_expr("x * y < 2"), variables) is None


@pytest.mark.parametrize('engine', list(ForwardAnalysisConfig.Engine))
def test_program_from_issue(engine):
    program = language.parse_program("""
x: nat;
x = geometric(1/2);
if (1/2 * x < 2) {
    x = 0;
} else {
    x = 1;
}""")
    config = ForwardAnalysisConfig(engine=engine)
    result, error_prob = compute_discrete_distribution(program, config.factory.one('x'), config)
    assert sympy.simplify(_to_sympy(result) - (sympy.Rational(15, 16) + X / 16)) == 0
    assert sympy.S(str(error_prob)) == 0


# (expression, its linear form without monus_safe, with monus_safe); None: not linear
LINEAR_FORMS = [
    ("3", {"": Fraction(3)}, {"": Fraction(3)}),
    ("0", {}, {}),
    ("x/2 + 1/3*y", {"x": Fraction(1, 2), "y": Fraction(1, 3)}, {"x": Fraction(1, 2), "y": Fraction(1, 3)}),
    ("2*(x + 1) / 4", {"x": Fraction(1, 2), "": Fraction(1, 2)}, {"x": Fraction(1, 2), "": Fraction(1, 2)}),
    ("5 - 2", {"": Fraction(3)}, {"": Fraction(3)}),
    ("2 - 5", None, None),  # truncated at 0
    ("(2*x + 3) - x", None, {"x": Fraction(1), "": Fraction(3)}),
    ("(x + y) - y", None, {"x": Fraction(1)}),
    ("x - x", None, {}),
    ("x - 1", None, None),  # 0 for x = 0, not -1
    ("x - y", None, None),
    ("(x + 1) - 2*x", None, None),
    ("x*y", None, None),
    ("x / y", None, None),
    ("z + 1", None, None),  # not a variable
]


@pytest.mark.parametrize("expression, plain, monus_safe", LINEAR_FORMS)
def test_linear_form(expression, plain, monus_safe):
    """The linear form of an expression over the natural variables x and y, with and without monus_safe."""
    expr = parse_ppy_expr(expression)
    assert linear_form(expr, {"x", "y"}) == plain
    assert linear_form(expr, {"x", "y"}, monus_safe=True) == monus_safe


def _monus_value(expr, values):
    """The value of a linear expression on natural numbers, with subtraction truncated at 0."""
    if isinstance(expr, language.VarExpr):
        return Fraction(values[expr.var])
    if isinstance(expr, language.NatLitExpr):
        return Fraction(expr.value)
    if isinstance(expr, language.RealLitExpr):
        return expr.to_fraction()
    left, right = _monus_value(expr.lhs, values), _monus_value(expr.rhs, values)
    return {language.Binop.PLUS: lambda: left + right, language.Binop.MINUS: lambda: max(left - right, 0),
            language.Binop.TIMES: lambda: left * right, language.Binop.DIVIDE: lambda: left / right}[expr.operator]()


@pytest.mark.parametrize("expression, _plain, form", [case for case in LINEAR_FORMS if case[2] is not None])
def test_monus_safe_linear_form_agrees_with_monus(expression, _plain, form):
    """Where monus_safe accepts a subtraction, the form has the value of the expression with monus for all
    natural values."""
    expr = parse_ppy_expr(expression)
    for x, y in product(range(5), repeat=2):
        point = {"x": x, "y": y, "": 1}
        assert sum((coefficient * point[key] for key, coefficient in form.items()), Fraction(0)) \
            == _monus_value(expr, {"x": x, "y": y}), (expression, x, y)
