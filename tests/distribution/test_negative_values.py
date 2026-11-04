"""The exact check whether a subtraction can be negative (issues #143 and #151)."""
import pytest
import sympy

import prodigy.pgcl as pgcl
from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.negative_values import can_be_negative
from prodigy.distribution.symengine_distribution import SymenginePGF
from prodigy.pgcl.parser import parse_expr


def _ginac_factory():
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import ProdigyPGF
    return ProdigyPGF


FACTORIES = [
    pytest.param(lambda: SympyPGF, id="sympy"),
    pytest.param(_ginac_factory, id="ginac"),
    pytest.param(lambda: SymenginePGF, id="symengine"),
]

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, lambda: SympyPGF, id="sympy"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, _ginac_factory, id="ginac"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, lambda: SymenginePGF, id="symengine"),
]

X, Y, Z = sympy.symbols("x y z")


@pytest.mark.parametrize("function, minuend, subtrahend, expected", [
    # a constant from a variable
    ("1/(2-x)", X, 1, True),
    ("1/(2-x) - 1/2", X, 1, False),
    ("1/(2-x) - 1/2", X, 2, True),
    ("1/(2-x) - 1/2 - x/4", X, 2, False),
    ("1/((1-y)*(1-x))", X, 0, False),
    ("(1/((1-y)*(1-x)) - 1/(1-y))", X, 1, False),
    ("1/((1-y)*(1-x))", X, 1, True),
    # coefficients that cancel in the marginal of `x` (signed coefficients, parameters of unknown sign)
    ("1 - y + x", X, 1, True),
    ("p - p*y + x", X, 1, True),
    ("(1 - y)/(2 - y) + x/(2 - y)", X, 1, True),
    ("exp(3*(x - 1))", X, 1, True),
    ("exp(3*(x - 1)) - exp(-3)", X, 1, False),
    ("exp(l*(x - 1)) - exp(-l)", X, 2, True),
    ("exp(x - 1)/(1 - y)", X, 1, True),
    # a variable from a constant
    ("1 + y + y**2", 2, Y, False),
    ("1 + y**3 - y**3*z", 2, Y, True),
    ("1/(2 - y)", 3, Y, True),
    ("1/(2 - y)", 0, Y, True),
    ("1 + 0*y", 0, Y, False),
    # a variable from a variable
    ("1/(2-y)", X, Y, True),
    ("1/((1-z)*(2-y))", X, Y, True),
    ("1 + x*y**2 - x**2*y**3", X, Y, True),
    ("p - p*x + x*y", X, Y, False),
    ("1/(2 - x*y)", X, Y, False),
    ("x/(2 - x*y)", X, Y, False),
    ("y/(2 - x*y)", X, Y, True),
    ("1/(2 - x)", X, X, False),
    ("x*exp(2*(x*y - 1))", X, Y, False),
    ("exp(x - 1)", X, Y, False),
    # floats on a finite support need no zero test
    ("0.4*x**5*y**13 + 0.6*x**7*y**28", Y, X, False),
    ("0.4*x**5*y**13 + 0.6*x**7*y**2", Y, X, True),
    ("x/2 + 0.5", X, 1, True),
    # two constants
    (1, 3, 1, False),
    (1, 1, 3, True),
])
def test_can_be_negative(function, minuend, subtrahend, expected):
    variables = {X, Y, Z}
    assert can_be_negative(sympy.S(function), variables, sympy.S(minuend), sympy.S(subtrahend)) is expected


@pytest.mark.parametrize("function, minuend, subtrahend", [
    ("exp(2*(x - 1) + 3*(y - 1))", X, Y),
    ("1/(2-x)", X, sympy.Symbol("n")),
    ("1", sympy.Symbol("n"), 1),
    ("x/(2 - y) + 0.5", X, 1),
])
def test_can_be_negative_undecided(function, minuend, subtrahend):
    assert can_be_negative(sympy.S(function), {X, Y, Z}, minuend, sympy.S(subtrahend)) is None


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("function, variables, parameters, assignment", [
    # signed coefficients cancel in the marginal of `x`, e.g. for templates in invariant synthesis
    ("1 - y + x", ["x", "y"], [], "x = x - 1"),
    ("p - p*y + x", ["x", "y"], ["p"], "x = x - 1"),
    ("(1 - y)/(2 - y) + x/(2 - y)", ["x", "y"], [], "x = x - 1"),
    ("1 + y^3 - y^3*z", ["x", "y", "z"], [], "x = 2 - y"),
    ("1 + x*y^2 - x^2*y^3", ["x", "y", "z"], [], "z = x - y"),
    ("(1 + x*y^2 - x^2*y^3)/(2 - w)", ["x", "y", "z", "w"], [], "z = x - y"),
    # the subtrahend has infinite support, so the marginal of the result is expanded at infinity
    ("1/(2 - y)", ["x", "y", "z"], [], "z = x - y"),
    ("1/((1 - w)*(2 - y))", ["x", "y", "z", "w"], [], "z = x - y"),
    ("1/(2 - y)", ["x", "y"], [], "x = 3 - y"),
    ("1/(2 - x)", ["x"], [], "x = 3 - x"),
    # a diverging marginal (issue #143)
    ("1/((1-y)*(1-x/2))", ["x", "y"], [], "x = x - 1"),
    ("exp(x - 1)/(1-y)", ["x", "y"], [], "x = x - 1"),
])
def test_negative_subtraction_is_rejected(factory, function, variables, parameters, assignment):
    dist = factory().from_expr(function, *variables)
    if parameters:
        dist = dist.set_parameters(*parameters)
    with pytest.raises(ValueError, match="can be negative"):
        dist.update(parse_expr(assignment))


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("function, variables, parameters, assignment, expected", [
    ("p - p*x + x*y", ["x", "y", "z"], ["p"], "z = x - y", "p - p*x*z + x*y"),
    ("1/(2 - x*y)", ["x", "y", "z"], [], "z = x - y", "1/(2 - x*y)"),
    ("x/(1 - w)", ["x", "y", "z", "w"], [], "z = x - y", "x*z/(1 - w)"),
    ("exp(l*(x - 1)) - exp(-l)", ["x", "y"], ["l"], "x = x - 1", "(exp(l*(x - 1)) - exp(-l))/x"),
    ("(exp(2*x - 2) - exp(-2))/(2 - y)", ["x", "y"], [], "x = x - 1", "(exp(2*x - 2) - exp(-2))/((2 - y)*x)"),
    ("1/((1-y)*(1-x/2)) - 1/(1-y)", ["x", "y"], [], "x = x - 1", "1/((1-y)*(1-x/2)) / 2"),
])
def test_safe_subtraction_is_accepted(factory, function, variables, parameters, assignment, expected):
    dist = factory().from_expr(function, *variables)
    if parameters:
        dist = dist.set_parameters(*parameters)
    result = dist.update(parse_expr(assignment))
    assert sympy.simplify(sympy.S(str(result).replace("^", "**")) - sympy.S(expected)) == 0


@pytest.mark.parametrize("factory", FACTORIES)
def test_undecided_subtraction_is_rejected(factory):
    # `y` is Poisson distributed: its exponential becomes `exp(2*(y/t - 1))`, which the check cannot decide
    dist = factory().from_expr("exp(2*(y - 1) + x - 1)", "x", "y", "z")
    with pytest.raises(ValueError, match="cannot be proven that it is never negative"):
        dist.update(parse_expr("z = x - y"))


@pytest.mark.parametrize("engine, factory", ENGINES)
@pytest.mark.parametrize("program, negative", [
    ("nat x; nat y; y := geometric(1/2); x := x - y;", True),
    ("nat x; nat y; nat z; y := geometric(1/2); z := x - y;", True),
    ("nat x; x := geometric(1/2); x := 3 - x;", True),
    ("nat x; x := geometric(1/2); if (x <= 3) { x := 3 - x } else { x := 0 }", False),
    ("nat x; nat y; y := geometric(1/2); x := y; x := y - x;", False),
    ("nat x; nat y; nat z; y := poisson(2); x := y + 1; z := x - y;", False),
    ("nat x; rparam p; x := geometric(p); if (x > 0) { x := x - 1 } else { skip }", False),
])
def test_subtraction_in_programs(engine, factory, program, negative):
    prog = pgcl.parse_pgcl(program)
    config = ForwardAnalysisConfig(engine=engine)
    initial = factory().from_expr("1")
    if negative:
        with pytest.raises(ValueError, match="can be negative"):
            compute_discrete_distribution(prog, initial, config)
    else:
        compute_discrete_distribution(prog, initial, config)
