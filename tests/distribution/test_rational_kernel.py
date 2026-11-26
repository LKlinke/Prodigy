"""
The FLINT rational kernel (#417): unit tests, and property tests that every operation agrees with SymPy on
random rational functions in program variables and a parameter, and that SymPy -> kernel -> SymPy is the
identity.
"""
from fractions import Fraction

import pytest
import sympy
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from prodigy.distribution.kernel import NotRepresentableError, PoleAtZeroError, RationalGF
from prodigy.distribution.pgf.rules.rational_series import rational_taylor_coefficients

X, Y, P = sympy.symbols("x y p")
PROPERTY = settings(max_examples=60, deadline=None, derandomize=True,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


def _same(first: sympy.Expr, second: sympy.Expr) -> bool:
    return sympy.cancel(sympy.together(first - second)) == 0


@st.composite
def polynomials(draw: st.DrawFn, symbols: tuple = (X, Y, P), max_terms: int = 4) -> sympy.Expr:
    """A random polynomial with small rational coefficients and exponents up to 3."""
    terms = []
    for _ in range(draw(st.integers(0, max_terms))):
        coefficient = sympy.Rational(draw(st.integers(-5, 5)), draw(st.integers(1, 4)))
        monomial = sympy.Mul(*[s ** draw(st.integers(0, 3)) for s in symbols])
        terms.append(coefficient * monomial)
    return sympy.Add(*terms)


@st.composite
def rationals(draw: st.DrawFn, symbols: tuple = (X, Y, P)) -> sympy.Expr:
    """A random rational function: a polynomial over a non-zero polynomial, unexpanded as SymPy builds it."""
    numerator = draw(polynomials(symbols))
    denominator = draw(polynomials(symbols)) + draw(st.sampled_from([0, 1, 1 - X / 2, 1 - P * Y]))
    assume(sympy.expand(denominator) != 0)
    return numerator / denominator


def _kernel(expression: sympy.Expr) -> RationalGF:
    return RationalGF.from_sympy(expression, ("x", "y", "p"))


# --- properties --------------------------------------------------------------------------------------------------

@PROPERTY
@given(rationals())
def test_round_trip_is_the_identity(expression):
    assert _same(_kernel(expression).to_sympy(), expression)


@PROPERTY
@given(rationals())
def test_representation_is_canonical(expression):
    """Two forms of the same function give the same reduced pair (cancelled, monic denominator)."""
    other = sympy.expand(expression.as_numer_denom()[0] * (3 - X * P)) / sympy.expand(
        expression.as_numer_denom()[1] * (3 - X * P))
    first, second = _kernel(expression), _kernel(other)
    assert first.numerator == second.numerator
    assert first.denominator == second.denominator
    assert first.denominator.is_zero() is False
    assert first.numerator.gcd(first.denominator).is_constant()


@PROPERTY
@given(rationals(), rationals())
def test_arithmetic_agrees_with_sympy(first, second):
    f, g = _kernel(first), _kernel(second)
    assert _same((f + g).to_sympy(), first + second)
    assert _same((f - g).to_sympy(), first - second)
    assert _same((f * g).to_sympy(), first * second)
    assert f + g == g + f and f * g == g * f
    if not g.is_zero():
        assert _same((f / g).to_sympy(), first / second)
    for exponent in (0, 1, 3):
        assert _same((f ** exponent).to_sympy(), first ** exponent)
    if not f.is_zero():
        assert _same((f ** -2).to_sympy(), first ** -2)


@PROPERTY
@given(rationals(), rationals((X, Y)), st.sampled_from(["x", "y", "p"]))
def test_substitution_agrees_with_sympy(expression, value, name):
    symbol = sympy.Symbol(name)
    function = _kernel(expression)
    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(expression)))
    if sympy.cancel(sympy.together(denominator.subs(symbol, value))) == 0:
        with pytest.raises(ZeroDivisionError):
            function.subs(name, _kernel(value))
        return
    assert _same(function.subs(name, _kernel(value)).to_sympy(), numerator.subs(symbol, value) /
                 denominator.subs(symbol, value))


@PROPERTY
@given(rationals(), st.sampled_from(["x", "y"]))
def test_coefficients_agree_with_the_sympy_recursion(expression, name):
    symbol = sympy.Symbol(name)
    reference = rational_taylor_coefficients(expression, symbol, 8)
    function = _kernel(expression)
    if reference is None:
        with pytest.raises(PoleAtZeroError):
            function.coefficient_list(name, 8)
        return
    reference += [0] * (8 - len(reference))
    for kernel, expected in zip(function.coefficient_list(name, 8), reference):
        assert _same(kernel.to_sympy(), expected)
        assert name not in kernel.free_names()


# --- units -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("expression", [
    "exp(x)", "log(1 - x)", "sqrt(1 - x)", "x**(1/2)", "x**y", "0.5*x", "pi*x", "I*x", "E", "1/(1 - exp(p)*x)",
    "zoo", "nan", "oo",
])
def test_not_representable(expression):
    with pytest.raises(NotRepresentableError):
        RationalGF.from_sympy(sympy.sympify(expression))


def test_symbols_with_the_same_name_are_rejected():
    with pytest.raises(NotRepresentableError):
        RationalGF.from_sympy(X + sympy.Symbol("x", positive=True))


def test_geometric_coefficients_with_a_parameter():
    function = RationalGF.from_sympy((1 - P) / (1 - P * X))
    coefficients = function.coefficient_list("x", 5)
    assert [c.to_sympy() for c in coefficients] == [sympy.expand((1 - P) * P ** n) for n in range(5)]
    assert function.subs("x", 1) == 1


def test_coefficients_of_a_polynomial_end_after_the_degree():
    function = RationalGF.from_sympy((1 + X) ** 3 / 8)
    assert [c.constant_value() for c in function.coefficients("x")] == [Fraction(1, 8), Fraction(3, 8),
                                                                         Fraction(3, 8), Fraction(1, 8)]
    assert len(function.coefficient_list("x", 6)) == 6
    assert not list(RationalGF.constant(0, ["x"]).coefficients("x"))
    assert [c.constant_value() for c in RationalGF.constant(2).coefficients("x")] == [2]


def test_coefficients_with_a_non_constant_q0_are_rational():
    """1/(1 - y - x y): q_0 = 1 - y depends on y, so s_n = y^n / (1 - y)^(n+1)."""
    function = RationalGF.from_sympy(1 / (1 - Y - X * Y))
    for n, coefficient in enumerate(function.coefficient_list("x", 6)):
        assert _same(coefficient.to_sympy(), Y ** n / (1 - Y) ** (n + 1))


def test_pole_at_zero():
    with pytest.raises(PoleAtZeroError):
        next(RationalGF.from_sympy(1 / (X * (1 - X))).coefficients("x"))


def test_division_by_zero():
    function = RationalGF.from_sympy(1 / (1 - X))
    with pytest.raises(ZeroDivisionError):
        function.subs("x", 1)
    with pytest.raises(ZeroDivisionError):
        function / 0
    with pytest.raises(ZeroDivisionError):
        RationalGF.constant(0).inverse()


def test_substitution_by_a_power_with_a_huge_exponent():
    """x := 2*x in the St. Petersburg loop maps g(x) to g(x^2); FLINT keeps exponents like 2^40 exact."""
    function = RationalGF.from_sympy(X / 2)
    for _ in range(40):
        function = function.subs("x", RationalGF.variable("x") ** 2)
    assert function.numerator.degrees() == (2 ** 40,)
    assert function.subs("x", 1).constant_value() == Fraction(1, 2)


def test_rings_are_merged_by_name():
    first, second = RationalGF.variable("x"), RationalGF.variable("p", ["c"])
    total = first + second
    assert total.names == ("c", "p", "x")
    assert total.free_names() == ("p", "x")
    assert total.subs("q", 3) is total
    assert first.in_ring(["x"]) is first


def test_scalars_mix_with_functions():
    function = RationalGF.variable("x")
    assert (1 - function) * Fraction(1, 2) + function / 2 == Fraction(1, 2)
    assert 2 / (function + 1) == RationalGF.from_sympy(2 / (X + 1))
    assert (function != "x") is True


def test_to_sympy_uses_the_given_symbols():
    n = sympy.Symbol("n", nonnegative=True, integer=True)
    expression = RationalGF.from_sympy((1 - X) / (1 - n * X)).to_sympy({"n": n})
    assert n in expression.free_symbols
    assert str(RationalGF.from_sympy(X / 3)) == "1/3*x"
    assert "ring=" in repr(RationalGF.constant(1))
