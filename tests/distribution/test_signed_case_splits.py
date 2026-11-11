"""
Regression tests for #256: updates that do a case split over the values of an operand (`*`, `^`,
`%`, `/`, state-dependent probabilities and arguments, `dist` values) split over the support of the
operand, not its marginal. In the marginal of a signed measure (e.g. a candidate measure of a loop
strategy) the mass of a value can cancel to 0, and the part with this value was lost.

In the measures below, the coefficient of `a` is `w - b^2`, which is 0 at w = b = 1: the marginal of
`a` has no term `a`, but the measure has states with a = 1 (with mass 1 and -1).

Also #257: a literal modulus or divisor 0 gives the same error on all engines (GiNaC reported a
non-finite marginal on a measure with infinite mass).
"""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.state_dependent import split_states
from prodigy.analysis.nested_inference import _values
from prodigy.pgcl.parser import parse_expr

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE,
           ForwardAnalysisConfig.Engine.GINAC]
SIMPLIFICATION = [True, False]

# the example of #256: a = 1 with w - b^2, a = 2 with b^3/(2 - b)
SIGNED = "a*(w - b^2) + a^2*b^3/(2 - b)"
# a polynomial (finite) signed measure
SIGNED_POLYNOMIAL = "a*(w - b^2) + a^2*b"
# infinite mass (sum over w diverges)
INFINITE = "1/((1 - w)*(2 - b))"
A, B, W, X = sympy.symbols("a b w x")
CANCELLING = A * (W - B ** 2)


def _config(engine, simplification=True):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    return ForwardAnalysisConfig(engine=engine, use_simplification=simplification)


def _sympy(dist) -> sympy.Expr:
    return sympy.sympify(str(dist).replace("^", "**"))


def _update(engine, simplification, function, assignment):
    config = _config(engine, simplification)
    with config.activate():
        dist = config.factory.from_expr(function, "a", "b", "w", "x")
        return _sympy(dist.update(parse_expr(assignment)))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
def test_marginal_cancels_the_value(engine, simplification):
    # the premise of these tests: the marginal of a has no state a = 1
    config = _config(engine, simplification)
    with config.activate():
        marginal = config.factory.from_expr(SIGNED, "a", "b", "w").marginal("a")
    assert sympy.simplify(_sympy(marginal) - A ** 2) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment, expected", [
    # the example of #256: a = 1 gives x := b, a = 2 gives x := 2*b
    ("x = a*b", A * (W - B ** 2 * X ** 2) + A ** 2 * B ** 3 * X ** 6 / (2 - B * X ** 2)),
    ("x = b*a", A * (W - B ** 2 * X ** 2) + A ** 2 * B ** 3 * X ** 6 / (2 - B * X ** 2)),
    ("x = a^2", CANCELLING * X + A ** 2 * B ** 3 * X ** 4 / (2 - B)),
    ("x = 2^a", CANCELLING * X ** 2 + A ** 2 * B ** 3 * X ** 4 / (2 - B)),
    ("x = a^a", CANCELLING * X + A ** 2 * B ** 3 * X ** 4 / (2 - B)),
    # a = 1: x := 0; a = 2: x := b % 2 (b^3/(2 - b) has the even part b^4/(4 - b^2))
    ("x = b % a", CANCELLING + A ** 2 * B ** 3 * (B + 2 * X) / (4 - B ** 2)),
    ("x = a % 2", CANCELLING * X + A ** 2 * B ** 3 / (2 - B)),
    ("x = a / 1", CANCELLING * X + A ** 2 * B ** 3 * X ** 2 / (2 - B)),
    # a = 1: x := 2*b; a = 2: x := b
    ("x = (2*b) / a", A * (W - B ** 2 * X ** 4) + A ** 2 * B ** 3 * X ** 3 / (2 - B * X)),
])
def test_case_split_updates_keep_values_whose_mass_cancels(engine, simplification, assignment, expected):
    result = _update(engine, simplification, SIGNED, assignment)
    assert sympy.simplify(result - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("assignment, expected", [
    ("x = a*b", A * W - A * B ** 2 * X ** 2 + A ** 2 * B * X ** 2),
    ("x = a^2", CANCELLING * X + A ** 2 * B * X ** 4),
    ("x = 2^a", CANCELLING * X ** 2 + A ** 2 * B * X ** 4),
    ("x = a^b", CANCELLING * X + A ** 2 * B * X ** 2),
    ("x = b % a", CANCELLING + A ** 2 * B * X),
    ("x = a % 2", CANCELLING * X + A ** 2 * B),
    ("x = a / 1", CANCELLING * X + A ** 2 * B * X ** 2),
])
def test_case_split_updates_on_signed_polynomials(engine, simplification, assignment, expected):
    # a polynomial is unfolded as well: power and division iterated the marginal on every engine
    result = _update(engine, simplification, SIGNED_POLYNOMIAL, assignment)
    assert sympy.simplify(result - expected) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_state_dependent_split_keeps_values_whose_mass_cancels(engine):
    # split_states is the case split of `{...} [p] {...}` and `x = bernoulli(p)` with p depending on a
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(SIGNED, "a", "b", "w")
        parts = list(split_states(dist, {"a"}, "test"))
        assert sorted(part.state["a"] for part in parts) == [1, 2]
        total = sum((_sympy(part.distribution) for part in parts), sympy.Integer(0))
        assert sympy.simplify(total - _sympy(dist)) == 0
        # the values of `dist` variables are split the same way
        assert sorted(_values(dist, ["a"])) == [(1,), (2,)]


@pytest.mark.parametrize("engine", ENGINES)
def test_state_dependent_split_on_infinite_mass(engine):
    # a takes the single value 0: the split is exact although the marginal of a is not finite
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(INFINITE, "a", "b", "w")
        parts = list(split_states(dist, {"a"}, "test"))
        assert [part.state["a"] for part in parts] == [0]
        assert sympy.simplify(_sympy(parts[0].distribution) - _sympy(dist)) == 0
        assert _values(dist, ["a"]) == [(0,)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("function", [INFINITE, "1/(2 - b)", "b^2 + b"])
@pytest.mark.parametrize("assignment, message", [
    ("x = b % 0", "modulus has to be positive, got 0"),
    ("x = b / 0", "Division by zero"),
])
def test_literal_zero_modulus_or_divisor(engine, function, assignment, message):
    # #257: GiNaC raised NonFiniteMarginalError on the measure with infinite mass, and the engines
    # raised different errors (ZeroDivisionError, RuntimeError) on the polynomial
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr(function, "a", "b", "w", "x")
        with pytest.raises(ValueError, match=message):
            dist.update(parse_expr(assignment))


@pytest.mark.parametrize("engine", ENGINES)
def test_divisor_variable_with_value_zero_on_infinite_measure(engine):
    # the split over the values {0, 1} of a passes the divisor 0 as a literal: the same error
    config = _config(engine)
    with config.activate():
        dist = config.factory.from_expr("(a/2 + b/2)/(2 - w)", "a", "b", "w", "x")
        with pytest.raises(ValueError, match="modulus has to be positive, got 0"):
            dist.update(parse_expr("x = b % a"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("function, assignment, expected", [
    # found by the fuzzer: GiNaC lists the support 1/2 + x/2 of a as the state a = 0 twice (once per
    # term), and the split counted the part twice (mass 2)
    ("1/2 + x/2", "x = a / 2", sympy.Integer(1)),
    ("1/2 + x/2", "x = a ^ 2", sympy.Integer(1)),
    ("1/2 + x/2", "x = 2 ^ a", X),
    # infinite mass, the value a = 1 with two terms (1 and b) in its coefficient
    ("(a + a*b)/((1 - w)*(2 - b))", "x = a*b", A * (1 + B * X) / ((1 - W) * (2 - B * X))),
    ("(a + a*b)/((1 - w)*(2 - b))", "x = b % a", A * (1 + B) / ((1 - W) * (2 - B))),
])
def test_case_split_counts_every_value_once(engine, simplification, function, assignment, expected):
    result = _update(engine, simplification, function, assignment)
    assert sympy.simplify(result - expected) == 0
