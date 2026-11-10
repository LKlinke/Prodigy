"""
Regression tests for #242: a marginal of a measure with infinite mass is not a finite measure (`oo`,
`zoo`, `nan`, a division by zero). Code that unfolds a marginal into its states (filters on guards
like `a < b`, updates like `x := a * b`) must then raise `NonFiniteMarginalError` instead of reading
states off the non-finite value (SymEngine took `zoo` for the single state 0 and filtered to 0).

Filters and case-split updates need only the values of the unfolded variables, not their mass: if
these take finitely many values, their finite support is unfolded instead and the result is exact
(#249). GiNaC raises the same error as the other engines (#250).
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution import Distribution, NonFiniteMarginalError
from prodigy.pgcl import Binop, BinopExpr, VarExpr
from prodigy.pgcl.parser import parse_expr

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE]
ALL_ENGINES = [*ENGINES, ForwardAnalysisConfig.Engine.GINAC]
SIMPLIFICATION = [True, False]

# mass sum_{b, w} 2^(-b-1) = oo: summing out w diverges for every value of b
INFINITE = "1/((1 - w)*(2 - b))"
# infinite mass as well, but a takes the two values 1 and 2 (#249)
TWO_VALUES = "(a + a^2*w)/((1 - w)*(2 - b))"
# neither a nor b has a finite support
BOTH_INFINITE = "1/((1 - w)*(2 - a)*(2 - b))"
A_LESS_B = BinopExpr(Binop.LT, VarExpr("a"), VarExpr("b"))
A, B, W, X = sympy.symbols("a b w x")


def _config(engine, simplification):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine, use_simplification=simplification)


def _sympy(dist) -> sympy.Expr:
    return sympy.sympify(str(dist).replace("^", "**"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_marginal_of_infinite_measure_is_rejected(engine, simplification):
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w")
        with pytest.raises(NonFiniteMarginalError, match="not a finite measure"):
            dist.checked_marginal("b")
        with pytest.raises(NonFiniteMarginalError, match="not a finite measure"):
            dist.checked_marginal("a")
        # summing out the other variables converges here
        assert sympy.simplify(_sympy(dist.checked_marginal("w")) - 1 / (1 - sympy.Symbol("w"))) == 0


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_filter_on_infinite_measure_raises_instead_of_a_wrong_result(engine, simplification):
    # both sides take infinitely many values: there is nothing finite to unfold
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(BOTH_INFINITE, "a", "b", "w")
        with pytest.raises(NonFiniteMarginalError):
            dist.filter(A_LESS_B)


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment", ["x = b*a", "x = a*b", "x = a % b", "x = a ^ b", "x = 2 ^ b",
                                        "x = b / 2", "x = a / b"])
def test_case_split_updates_on_infinite_measure_raise(engine, simplification, assignment):
    # GiNaC raised the bindings' ValueError (division by zero) here (#250)
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(BOTH_INFINITE, "a", "b", "w", "x")
        with pytest.raises(NonFiniteMarginalError):
            dist.update(parse_expr(assignment))


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment", ["x = a ^ b", "x = 2 ^ b", "x = b ^ a"])
def test_updates_needing_an_infinite_support_still_raise(engine, simplification, assignment):
    # a has a finite support, but the power still needs the values of b (#249 only fixes finite ones)
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(TWO_VALUES, "a", "b", "w", "x")
        with pytest.raises(NonFiniteMarginalError):
            dist.update(parse_expr(assignment))


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("condition, expected", [
    # a = 0 everywhere: a < b means b >= 1 (SymPy computed this by luck before #242)
    ("a < b", 1 / ((1 - W) * (2 - B)) - 1 / (2 * (1 - W))),
    ("b < a", sympy.Integer(0)),
    ("a = b", 1 / (2 * (1 - W))),
    ("b <= a", 1 / (2 * (1 - W))),
])
def test_filter_unfolds_the_finite_support_on_infinite_measure(engine, simplification, condition, expected):
    # #249: the marginal of a is not finite (w diverges), but its support {0} is
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w")
        result = dist.filter(parse_expr(condition))
    assert sympy.simplify(_sympy(result) - expected) == 0


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_filter_unfolds_a_support_with_two_values(engine, simplification):
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(TWO_VALUES, "a", "b", "w")
        result = dist.filter(A_LESS_B)
    # a = 1: b >= 2; a = 2: b >= 3 (the terms of 1/(2 - b) are b^k / 2^(k+1))
    geometric = 1 / (2 - B)
    expected = (A / (1 - W) * (geometric - sympy.Rational(1, 2) - B / 4)
                + A ** 2 * W / (1 - W) * (geometric - sympy.Rational(1, 2) - B / 4 - B ** 2 / 8))
    assert sympy.simplify(_sympy(result) - expected) == 0


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment, expected", [
    # a = 1: x := b; a = 2: x := 2*b
    ("x = a*b", A / ((1 - W) * (2 - B * X)) + A ** 2 * W / ((1 - W) * (2 - B * X ** 2))),
    ("x = b*a", A / ((1 - W) * (2 - B * X)) + A ** 2 * W / ((1 - W) * (2 - B * X ** 2))),
    # a = 1: x := 0; a = 2: x := b % 2
    ("x = b % a", A / ((1 - W) * (2 - B)) + A ** 2 * W / (1 - W) * (2 + B * X) / (4 - B ** 2)),
    # a = 1: x := 2*b; a = 2: x := b
    ("x = (2*b) / a", A / ((1 - W) * (2 - B * X ** 2)) + A ** 2 * W / ((1 - W) * (2 - B * X))),
    # the copies of a for both operands take the values (1, 1) and (2, 2) jointly
    ("x = a ^ a", A * X / ((1 - W) * (2 - B)) + A ** 2 * W * X ** 4 / ((1 - W) * (2 - B))),
])
def test_case_split_updates_unfold_the_finite_support(engine, simplification, assignment, expected):
    # #249: the update is split over the values {1, 2} of a, the only operand with a finite support
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(TWO_VALUES, "a", "b", "w", "x")
        result = dist.update(parse_expr(assignment))
    assert sympy.simplify(_sympy(result) - expected) == 0


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_finite_support(engine):
    config = _config(engine, True)
    with config.activate():
        dist = config.factory.from_expr(TWO_VALUES, "a", "b", "w")
        support = dist.finite_support("a")
        assert support is not None
        assert sorted(state["a"] for _, state in support) == [1, 2]
        assert dist.finite_support("b") is None
        assert dist.finite_support("a", "b") is None
        # cancelling terms in the other variables do not hide a value
        signed = config.factory.from_expr("a*(w - b) + a^2*(1 - w)", "a", "b", "w")
        signed_support = signed.finite_support("a")
        assert signed_support is not None
        assert sorted(state["a"] for _, state in signed_support) == [1, 2]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment, expected", [
    # x := b % 2 does not unfold the marginal of b: the arithmetic progression is exact for any measure
    ("x = b % 2", (2 + B * X) / ((4 - B ** 2) * (1 - W))),
    # an exact division by a constant needs no marginal either
    ("x = (2*b) / 2", 1 / ((1 - W) * (2 - B * X))),
])
def test_constant_modulus_and_divisor_stay_exact_on_infinite_measure(engine, simplification, assignment,
                                                                      expected):
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w", "x")
        result = dist.update(parse_expr(assignment))
    assert sympy.simplify(_sympy(result) - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_state_unfolding_on_finite_mass_stays_exact(engine, simplification):
    # b has infinite support, so a < b is unfolded over the two values of a
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr("(1 + a)/(2*(2 - b))", "a", "b")
        result = dist.filter(A_LESS_B)
    a, b = sympy.symbols("a b")
    # a = 0: all b >= 1; a = 1: all b >= 2
    expected = (1 / (2 - b) - sympy.Rational(1, 2)) / 2 + a * (1 / (2 - b) - sympy.Rational(1, 2) - b / 4) / 2
    assert sympy.simplify(_sympy(result) - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_non_finite_values_are_detected(engine):
    config = _config(engine, True)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w")
        finite = dist.marginal("w")
        assert not finite.has_non_finite_value()
        assert not Distribution.has_non_finite_value(finite)
        for variable in ("a", "b"):
            # the engine's check and the default (string) check
            marginal = dist.marginal(variable)
            assert marginal.has_non_finite_value(), variable
            assert Distribution.has_non_finite_value(marginal), variable


def test_variable_named_like_a_non_finite_value_is_not_one():
    config = _config(ForwardAnalysisConfig.Engine.SYMPY, True)
    with config.activate():
        dist = config.factory.from_expr("inf/2", "inf")
    # the default (string) check excludes the program's variables and parameters
    assert not Distribution.has_non_finite_value(dist)


def test_ginac_marginal_division_by_zero_is_a_non_finite_marginal():
    pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.GINAC)
    with config.activate():
        dist = config.factory.from_expr(BOTH_INFINITE, "a", "b", "w")
        # GiNaC's marginal divides by zero (or, once the bindings handle it, returns a non-finite
        # value); either way the filter must not return a result
        with pytest.raises(NonFiniteMarginalError):
            dist.filter(A_LESS_B)
