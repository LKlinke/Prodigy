"""
The three proof rules for classes of generating functions of #431, on generic members of each class and on
near-misses that the rules must not prove: substitution of a monomial ``F(r*x^m)``, the shift ``G / x^k``, and
univariate rational functions with a unique dominant simple pole (a decision procedure).
"""
import random
from fractions import Fraction

import pytest
import sympy

from prodigy.distribution.pgf.rules import dominant_pole
from prodigy.distribution.pgf.rules.dominant_pole import dominant_pole_verdict
from prodigy.distribution.pgf.rules.finite_measure import check_input_expression, check_series
from prodigy.distribution.pgf.sympy.engine import SympyPGF
from prodigy.distribution.pgf.rules.rational_series import rational_taylor_coefficients
from prodigy.distribution.pgf.rules.series_transforms import low_coefficients_vanish, monomial_substitution, shift

x, y = sympy.symbols("x y")


def _text(expr):
    return str(expr).replace("**", "^")


def _coefficients(expr, count):
    series = sympy.series(expr, x, 0, count).removeO()
    return [series.coeff(x, n) for n in range(count)]


# ---- substitution of a monomial ----------------------------------------------------------------------------

def test_monomial_substitution_shape():
    assert monomial_substitution(1 - sympy.sqrt(1 - x ** 6), x) == (1 - sympy.sqrt(1 - x), 6)
    assert monomial_substitution(x ** 4 / (1 - x ** 6), x) == (x ** 2 / (1 - x ** 3), 2)
    assert monomial_substitution(x ** 2 + x, x) is None  # a bare x: exponent gcd 1
    assert monomial_substitution(sympy.sqrt(x) + x ** 2, x) is None  # not a power with integer exponent
    assert monomial_substitution(y ** 2, x) is None  # x does not occur


def _random_fractional(rng):
    """``A - B*(1 - r*u)^a`` with ``0 < a < 1``, ``0 < r <= 1``, ``B > 0`` and ``A >= B``: non-negative
    coefficients and finite mass (the existing rule for fractional powers)."""
    a = Fraction(rng.randint(1, 6), 7)
    r = Fraction(rng.randint(1, 9), 9)
    b = Fraction(rng.randint(1, 5), 5)
    return sympy.Rational(b) + sympy.Rational(rng.randint(0, 3), 4), sympy.Rational(b), sympy.Rational(a), \
        sympy.Rational(r)


@pytest.mark.parametrize("seed", range(8))
def test_monomial_substitution_generic_members(seed):
    rng = random.Random(seed)
    offset, scale, exponent, ratio = _random_fractional(rng)
    m = rng.randint(2, 5)
    expr = offset - scale * (1 - ratio * x ** m) ** exponent
    assert check_input_expression(_text(expr), ["x"]) == (True, "")
    assert all(c >= 0 for c in _coefficients(expr, 3 * m + 1))


@pytest.mark.parametrize("seed", range(4))
def test_monomial_substitution_of_rational_functions(seed):
    """``F(x^m)`` for an ``F`` with two dominant poles ``±rho`` in x becomes one dominant pole in ``u = x^2``."""
    rng = random.Random(seed)
    rho = rng.randint(2, 5)
    other = rng.randint(rho + 1, 9)
    expr = (1 + x ** 2) / ((rho - x ** 2) * (other + x ** 2))
    assert check_input_expression(_text(expr), ["x"]) == (True, "")


@pytest.mark.parametrize("text", [
    "1 - (1 + c^2)^(1/2)",  # F(-c^2): a negative monomial
    "2 - (1 - c^2)^(1/2) - c",  # c occurs as c and c^2
    "1 - (1 - 2*c^2)^(1/2)",  # r > 1: the series diverges at 1
])
def test_monomial_substitution_near_misses(text):
    assert check_input_expression(text, ["c"])[0] is not True


# ---- shift -------------------------------------------------------------------------------------------------

def test_shift_shape():
    assert shift((1 - sympy.sqrt(1 - x ** 2)) / x ** 2, x) == (1 - sympy.sqrt(1 - x ** 2), 2)
    assert shift(1 - x, x) is None
    assert low_coefficients_vanish(1 - sympy.sqrt(1 - x ** 2), x, 2)
    assert not low_coefficients_vanish(1 - sympy.sqrt(1 - x ** 2), x, 3)
    assert not low_coefficients_vanish(sympy.sqrt(x), x, 1)


@pytest.mark.parametrize("seed", range(8))
def test_shift_generic_members(seed):
    rng = random.Random(seed)
    offset, scale, exponent, ratio = _random_fractional(rng)
    m = rng.randint(1, 4)
    k = rng.randint(1, m)
    inner = scale - scale * (1 - ratio * x ** m) ** exponent  # no constant term, first power x^m
    expr = inner * y / x ** k
    assert check_input_expression(_text(expr), ["x", "y"]) == (True, "")
    too_far = inner / x ** (m + 1)  # a pole at 0: never proven
    assert check_input_expression(_text(too_far), ["x"])[0] is not True
    del offset


def test_library_forms_are_probability_generating_functions():
    assert check_input_expression("(1 - (1 - c * c) ^ (1/2)) / c", ["c"], probability=True) == (True, "")
    assert check_input_expression("7/13 * (3 * i ^ 3 - 16 * i) / (i ^ 2 + 8 * i - 16)", ["i"],
                                  probability=True) == (True, "")
    assert check_input_expression("7/3 * (i ^ 2 - 4 * i) / (i ^ 2 + 8 * i - 16)", ["i"],
                                  probability=True) == (True, "")


# ---- univariate rational functions with a unique dominant simple pole -------------------------------------

def _check_against_coefficients(expr, verdict, count=300):
    coefficients = rational_taylor_coefficients(expr, x, count) + [sympy.S.Zero] * count
    if verdict.holds is True:
        assert all(c >= 0 for c in coefficients)
    elif verdict.holds is False:
        index, value = verdict.witness
        assert value < 0 and coefficients[index] == sympy.Rational(value.numerator, value.denominator)


@pytest.mark.parametrize("seed", range(12))
def test_dominant_pole_generic_members(seed):
    """``(a + b*x + c*x^2) / ((1 - x/rho) * Q1)`` with a positive rational or irrational ``rho`` and the roots of
    ``Q1`` farther out: decided, and the decision agrees with the exact coefficients."""
    rng = random.Random(seed)
    root = sympy.Rational(rng.randint(2, 6), rng.randint(1, 3))
    if seed % 2:
        dominant = x ** 2 + rng.randint(1, 4) * x - root  # one positive root < sqrt(root), one negative farther
    else:
        dominant = root - x
    far = rng.randint(30, 60)
    q1 = far ** 2 + rng.randint(-far, far) * x + x ** 2  # roots of modulus far
    numerator = sum(rng.randint(-3, 5) * x ** i for i in range(3)) or sympy.S.One
    expr = numerator / sympy.expand(dominant * q1)
    verdict = dominant_pole_verdict(expr, x)
    assert verdict.holds is not None, verdict.reason
    _check_against_coefficients(expr, verdict)


@pytest.mark.parametrize("seed", range(20))
def test_dominant_pole_random_rational_functions(seed):
    """Random rational functions: whatever is decided agrees with the exact coefficients."""
    rng = random.Random(1000 + seed)
    denominator = sum(rng.randint(-6, 6) * x ** i for i in range(1, rng.randint(2, 5))) + rng.choice([-5, 5, 7])
    numerator = sum(rng.randint(-4, 6) * x ** i for i in range(rng.randint(1, 4))) or sympy.S.One
    expr = sympy.cancel(numerator / denominator)
    _check_against_coefficients(expr, dominant_pole_verdict(expr, x))


@pytest.mark.parametrize("expr, holds", [
    (1 / (1 - x) ** 2, None),  # a double dominant pole: outside the class
    (1 / (1 - x ** 2), None),  # two dominant poles 1 and -1
    (1 / (1 - x + x ** 2), None),  # a complex pair of dominant poles
    (1 / ((2 - x) * (2 + x) * (5 - x)), None),  # two dominant poles 2 and -2
    (1 / (1 + x), False),  # a negative dominant pole
    ((1 - 3 * x) / ((2 - x) * (5 + x)), False),  # a negative residue at the dominant pole beyond x^0
    (-1 / (2 - x) + 3 / (3 - x), False),  # a negative dominant term, positive first coefficients
    (1 / (2 - x) - 1 / (3 - x) ** 2, True),  # the other pole is double
    (x ** 5 / (1 - x / 3) + 1 / (1 - x / 2) - 1 / (1 - x / 3), True),  # polynomial part and a cancellation
    (sympy.sqrt(2) / (2 - x), None),  # irrational coefficients: outside the class
])
def test_dominant_pole_near_misses(expr, holds):
    verdict = dominant_pole_verdict(expr, x)
    assert verdict.holds is holds, verdict
    _check_against_coefficients(expr, verdict)


def test_dominant_pole_beyond_the_coefficient_limit_is_undecided(monkeypatch):
    """If the index from which on the dominant pole decides the signs exceeds the limit, nothing is claimed."""
    monkeypatch.setattr(dominant_pole, "MAX_EXACT_COEFFICIENTS", 0)
    verdict = dominant_pole.dominant_pole_verdict(1 / ((2 - x) * (1 + x / 3)), x)
    assert verdict.holds is None and "exceeds 0" in verdict.reason


def test_dominant_pole_refutation_reaches_the_input_check():
    verdict = check_input_expression("(1 - 3*x)/((2 - x)*(5 + x))", ["x"])
    assert verdict == (False, "the coefficient of x is -27/100, which is negative")


# ---- the order uses the rules ----------------------------------------------------------------------------

@pytest.mark.parametrize("lower, upper", [
    ("(1 - (1 - x^2)^(1/2)) / x / 2", "(1 - (1 - x^2)^(1/2)) / x"),
    ("1/(x^2 + 8*x - 16) * (-1)", "2/(16 - 8*x - x^2) + x/(16 - 8*x - x^2)"),
])
def test_order_uses_the_class_rules(lower, upper):
    a = SympyPGF.from_expr(lower, "x")
    b = SympyPGF.from_expr(upper, "x")
    assert a.decide_le_verdict(b).holds is True


def test_check_series_on_sympy_expressions():
    assert check_series((1 - sympy.sqrt(1 - x ** 2)) / x, ["x"]) == (True, "")
