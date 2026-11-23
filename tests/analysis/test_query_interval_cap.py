"""A ``?Pr`` query interval of a lower bound is capped at 1 like the table rows (``upper_end``); #468."""
import sympy

from prodigy.analysis.result import Bound, make_interval


def test_probability_interval_is_capped_at_one():
    lower, upper, bound, _ = make_interval(sympy.Rational(3, 5), Bound.LOWER, sympy.Rational(1, 2),
                                           monotone=True, probability=True)
    assert (lower, bound) == (sympy.Rational(3, 5), Bound.LOWER)
    assert upper == 1


def test_probability_interval_below_one_keeps_the_sum():
    _, upper, _, _ = make_interval(sympy.Rational(1, 5), Bound.LOWER, sympy.Rational(1, 2),
                                   monotone=True, probability=True)
    assert upper == sympy.Rational(7, 10)


def test_probability_interval_with_parameter_keeps_the_sum():
    p = sympy.Symbol("p")
    _, upper, _, _ = make_interval(p, Bound.LOWER, sympy.Rational(1, 2), monotone=True, probability=True)
    assert sympy.simplify(upper - (p + sympy.Rational(1, 2))) == 0
