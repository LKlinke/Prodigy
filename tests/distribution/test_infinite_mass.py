"""
Regression tests for #242: a marginal of a measure with infinite mass is not a finite measure (`oo`,
`zoo`, `nan`, a division by zero). Code that unfolds a marginal into its states (filters on guards
like `a < b`, updates like `x := a * b`) must then raise `NonFiniteMarginalError` instead of reading
states off the non-finite value (SymEngine took `zoo` for the single state 0 and filtered to 0).
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution import Distribution, NonFiniteMarginalError
from prodigy.pgcl import Binop, BinopExpr, VarExpr
from prodigy.pgcl.parser import parse_expr

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE]
SIMPLIFICATION = [True, False]

# mass sum_{b, w} 2^(-b-1) = oo: summing out w diverges for every value of b
INFINITE = "1/((1 - w)*(2 - b))"
A_LESS_B = BinopExpr(Binop.LT, VarExpr("a"), VarExpr("b"))


def _config(engine, simplification):
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


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_filter_on_infinite_measure_raises_instead_of_a_wrong_result(engine, simplification):
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w")
        with pytest.raises(NonFiniteMarginalError):
            dist.filter(A_LESS_B)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment", ["x = b*a", "x = a*b", "x = a % b", "x = a ^ b", "x = 2 ^ b",
                                        "x = b / 2", "x = a / b"])
def test_case_split_updates_on_infinite_measure_raise(engine, simplification, assignment):
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w", "x")
        with pytest.raises(NonFiniteMarginalError):
            dist.update(parse_expr(assignment))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_modulo_by_constant_stays_exact_on_infinite_measure(engine, simplification):
    # x := b % 2 does not unfold the marginal of b: the arithmetic progression is exact for any measure
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w", "x")
        result = dist.update(parse_expr("x = b % 2"))
    b, w, x = sympy.symbols("b w x")
    assert sympy.simplify(_sympy(result) - (2 + b * x) / ((4 - b ** 2) * (1 - w))) == 0


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
        dist = config.factory.from_expr(INFINITE, "a", "b", "w")
        # GiNaC's marginal divides by zero (or, once the bindings handle it, returns a non-finite
        # value); either way the filter must not return a result
        with pytest.raises(NonFiniteMarginalError):
            dist.filter(A_LESS_B)
