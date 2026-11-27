import sympy

from prodigy.distribution.pgf.rules.equality import _numerically_nonzero, decide_equal

Y = sympy.Symbol("y")
OMEGA = sympy.exp(2 * sympy.I * sympy.pi / 3)


def test_numeric_precheck_sees_a_nonzero_sum_of_exponentials():
    value = sympy.Rational(2, 3) * sympy.exp(Y - 1) - OMEGA * sympy.exp(Y / OMEGA - 1) / 3 \
        - sympy.exp(Y * OMEGA - 1) / (3 * OMEGA)
    assert _numerically_nonzero(value)
    assert decide_equal(value, sympy.Integer(0)) is None


def test_numeric_precheck_never_calls_a_zero_nonzero():
    assert not _numerically_nonzero(sympy.exp(Y) * sympy.exp(-Y) - 1)
    assert not _numerically_nonzero(sympy.Integer(0))
    assert decide_equal(sympy.exp(Y) * sympy.exp(-Y), sympy.Integer(1)) is True


def test_roots_of_unity_in_two_notations_cancel():
    # the roots of unity as exp(2*I*pi/3) and as -1/2 + sqrt(3)*I/2
    left = sympy.exp(Y * (-sympy.Rational(1, 2) + sympy.sqrt(3) * sympy.I / 2)) * (OMEGA + 1)
    right = sympy.exp(Y * OMEGA) * (OMEGA + 1)
    assert decide_equal(left, right, use_simplify=False) is True


def test_a_huge_constant_difference_does_not_hang():
    # a sum of exp of complex roots of unity, about 49000 operations in normal form: only the cheap proof is tried
    constant = sum(sympy.exp(k * OMEGA + sympy.Rational(1, k)) * (OMEGA ** k) for k in range(1, 6))
    result = decide_equal(constant ** 3, constant ** 3 + sympy.exp(OMEGA) - sympy.exp(OMEGA), use_simplify=False)
    assert result is True


def test_nonzero_difference_with_irrational_constants_is_not_cancelled():
    # nothing to decide: not zero, and not rational over Q; the answer comes without a cancel over the extension
    value = sympy.exp(Y * OMEGA) * (OMEGA + 2) - sympy.exp(Y)
    assert decide_equal(value, sympy.Integer(0), use_simplify=False) is None
