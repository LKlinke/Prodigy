"""
The coefficient-wise order ``<=``/``<`` of distributions, shared by the engines (#441): proven True, refuted
with a witness state, or ``UndecidedComparisonError``, never a guess.
"""
import pytest

from prodigy.distribution import State, UndecidedComparisonError
from prodigy.distribution.generating_function import SympyPGF
from tests.distribution.engines import ginac_factory


def _pair(factory, lhs, rhs, variables=("x",), parameters=()):
    return tuple(factory.from_expr(text, *variables).set_parameters(*parameters) for text in (lhs, rhs))


def test_issue_441_reproduction(factory):
    a, b = _pair(factory, "1/(2-x) + 1/2", "1/(2-x)")
    assert not a <= b
    assert b <= a
    assert b < a
    assert not a < b
    assert a.decide_le_verdict(b).witness == State({"x": 0})


def test_equal_infinite(factory):
    a, b = _pair(factory, "1/(2-x)", "1/(2-x)")
    assert a <= b
    assert b <= a
    assert not a < b


@pytest.mark.parametrize("lhs, rhs, holds, witness", [
    ("x^2*y^3/2", "x^2*y^3", True, None),
    ("x^2*y^3", "x^2*y^3/2", False, {"x": 2, "y": 3}),
    ("x*y/2 + y/2", "x*y/2 + y/2", True, None),
    ("x/2", "y/2", False, {"x": 1, "y": 0}),
])
def test_finite(factory, lhs, rhs, holds, witness):
    a, b = _pair(factory, lhs, rhs, ("x", "y"))
    verdict = a.decide_le_verdict(b)
    assert verdict.holds is holds
    assert verdict.witness == (None if witness is None else State(witness))
    assert (a <= b) is holds


@pytest.mark.parametrize("lhs, rhs, holds, witness", [
    ("x/4", "1/(2-x)", True, None),               # finite below infinite
    ("x", "1/(2-x)", False, {"x": 1}),
    ("1/(2-x)", "x/4", False, {"x": 0}),          # infinite below finite
    ("1/(2-x)", "1/(2-x) + 1/(3-x)", True, None),  # non-negative by construction
    ("1/(2-x)", "1/(2-x) - x^30", False, {"x": 30}),
    ("(1-sqrt(1-x^2))/x", "2/(2-x)-1", False, {"x": 5}),  # found by the coefficient search
])
def test_infinite(factory, lhs, rhs, holds, witness):
    a, b = _pair(factory, lhs, rhs)
    verdict = a.decide_le_verdict(b)
    assert verdict.holds is holds
    assert verdict.witness == (None if witness is None else State(witness))


def test_multivariate_infinite(factory):
    a, b = _pair(factory, "x*y/4", "1/((2-x)*(2-y))", ("x", "y"))
    assert a.decide_le_verdict(b).witness == State({"x": 1, "y": 1})
    a, b = _pair(factory, "x*y/16", "1/((2-x)*(2-y))", ("x", "y"))
    assert a <= b


@pytest.mark.parametrize("lhs, rhs", [
    ("p*x", "x"),             # (1 - p)*x: the sign depends on p
    ("x/4", "p/(2-x)"),
    ("p*x", "1/(2-x)"),
])
def test_parametric_undecided(factory, lhs, rhs):
    a, b = _pair(factory, lhs, rhs, parameters=("p",))
    assert a.decide_le(b) is None
    with pytest.raises(UndecidedComparisonError, match="Cannot decide whether"):
        assert a <= b
    with pytest.raises(UndecidedComparisonError):
        assert a < b


def test_parametric_decided(factory):
    a, b = _pair(factory, "p*x", "p*x + 1/2", parameters=("p",))
    assert a <= b
    assert a < b
    a, b = _pair(factory, "p*x + 1", "p*x + 1/2", parameters=("p",))
    assert not a <= b


def test_undecided_infinite(factory):
    # exp(x) - 1 - x has positive coefficients, but no rule proves it
    a, b = _pair(factory, "1/(2-x)", "1/(2-x) + exp(x) - 1 - x")
    with pytest.raises(UndecidedComparisonError):
        assert a <= b


def test_fractional_power_difference(factory):
    # 1 - sqrt(1 - x) has positive coefficients: proven by the rule for fractional powers (#431 uses the proofs
    # of finite measures in the order)
    a, b = _pair(factory, "1/(2-x)", "1/(2-x) + 1 - sqrt(1-x)")
    assert a <= b


def test_incomparable(factory):
    a = factory.from_expr("x", "x")
    with pytest.raises(TypeError, match="Incomparable types"):
        assert a <= 1
    with pytest.raises(TypeError, match="Incomparable distributions"):
        assert a <= factory.from_expr("x", "x", "y")
    with pytest.raises(TypeError, match="Incomparable distributions"):
        assert a <= factory.from_expr("x", "x").set_parameters("p")


def test_engines_incomparable():
    ginac = ginac_factory()
    with pytest.raises(TypeError, match="Incomparable types"):
        assert SympyPGF.from_expr("x", "x") <= ginac.from_expr("x", "x")
