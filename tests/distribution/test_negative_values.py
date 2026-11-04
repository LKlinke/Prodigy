"""The check for negative values of a subtraction when a marginal diverges (issue #143)."""
import pytest
import sympy

from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.negative_values import has_negative_powers
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl.parser import parse_expr

ENGINES = [pytest.param(GeneratingFunction, id="sympy"), pytest.param(SymengineDist, id="symengine")]

# `y` has infinite mass, so the marginal of `x` diverges and cannot decide the subtraction.
INFINITE_MASS = "1/((1-y)*(1-x/2))"


@pytest.mark.parametrize("function, expected", [
    ("1/x", True),
    ("(1 + x)/x", True),
    ("x/(x*(1-y))", False),
    ("1/((1-y)*(1-x))", False),
    ("1/((1-y)*(1-x)) / x", True),
    ("(1/((1-y)*(1-x)) - 1/(1-y)) / x", False),
    ("(1/((1-y)*(1-x)) - 1/(1-y) - x/(1-y)) / x**2", False),
    ("(1/((1-y)*(1-x)) - 1/(1-y)) / x**2", True),
    ("(1/((1-p*y)*(1-x)) - 1/(1-p*y)) / x", False),
])
def test_has_negative_powers(function, expected):
    assert has_negative_powers(sympy.S(function), sympy.Symbol("x")) is expected


def test_has_negative_powers_undecided():
    assert has_negative_powers(sympy.S("exp(x - 1)/(x*(1 - y))"), sympy.Symbol("x")) is None


@pytest.mark.parametrize("engine", ENGINES)
def test_subtraction_under_infinite_mass_is_rejected(engine):
    dist = engine(INFINITE_MASS, "x", "y")
    with pytest.raises(ValueError, match="because it can be negative"):
        dist.update(parse_expr("x = x - 1"))


@pytest.mark.parametrize("engine", ENGINES)
def test_subtraction_under_infinite_mass_is_accepted(engine):
    dist = engine(INFINITE_MASS, "x", "y").filter(parse_expr("x > 0"))
    result = dist.update(parse_expr("x = x - 1"))
    expected = sympy.S("1/((1-y)*(1-x/2)) / 2")
    assert sympy.simplify(sympy.S(str(result)) - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_subtraction_under_infinite_mass_undecided(engine):
    dist = engine("exp(x - 1)/(1-y)", "x", "y")
    with pytest.raises(ValueError, match="cannot be proven that it is never negative"):
        dist.update(parse_expr("x = x - 1"))


@pytest.mark.parametrize("function, expected", [
    ("exp(3*(x - 1))/x**2", True),
    ("(exp(3*(x - 1)) - exp(-3))/x", False),
    ("(exp(3*(x - 1)) - exp(-3))/x**2", True),
])
def test_has_negative_powers_univariate(function, expected):
    assert has_negative_powers(sympy.S(function), sympy.Symbol("x")) is expected
