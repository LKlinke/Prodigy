"""
Substitutions with roots of unity (modulo guards, arithmetic progressions, `x % q` with an infinite
marginal of x) are only sound for generating functions that are single-valued in x, e.g. rational
(issue #116).
"""
import pytest
import sympy
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.fast_generating_function import FPS
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.distribution.roots_of_unity import NotSingleValuedError, is_single_valued_in
from prodigy.distribution.symengine_distribution import SymengineDist
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.parser import parse_expr

ENGINES = [GeneratingFunction, FPS, SymengineDist]
VARIABLES = ("c", "y", "z")
C, Y, Z = sympy.symbols("c y z")
DEGREE = 6

# Not single-valued in c. `sqrt(c^2*y^2)` denotes the series c*y, but the engines evaluate it at -c with
# the principal branch (giving sqrt(c^2*y^2) again), so all of it ended up in the even part.
MULTI_VALUED = [
    "sqrt(c^2*y^2)",
    "(1 - sqrt(1 - c^2*y^2))/sqrt(c^2*y^2)",
    "(1 - sqrt(1 - c^2*y^2))/(c*y)",
]
SINGLE_VALUED = [
    "y/(2 - c)",
    "(1 + c*y)/((2 - c)*(3 - y))",
    # roots of other variables are allowed
    "c*(1 - sqrt(1 - y^2))/(y*(2 - c))",
    # entire functions are single-valued as well, e.g. a Poisson distribution
    "exp(2*(c*y - 1))",
]


def _sympy(expression) -> sympy.Expr:
    text = str(expression).replace("**", "^")
    return sympy.parse_expr(text, local_dict={"c": C, "y": Y, "z": Z, "I": sympy.I, "Pi": sympy.pi},
                            transformations=standard_transformations + (convert_xor,))


def _coefficients(expression, keep=lambda i: True):
    """The coefficients of c^i y^j z^k for i, j, k < DEGREE (with c^i dropped unless keep(i))."""
    function = _sympy(expression)
    series = sympy.series(function, C, 0, DEGREE).removeO()
    result = {}
    for i in range(DEGREE):
        coefficient = sympy.expand(series.coeff(C, i))
        for j in range(DEGREE):
            row = sympy.series(coefficient, Y, 0, DEGREE).removeO().coeff(Y, j)
            for k in range(DEGREE):
                value = sympy.nsimplify(sympy.simplify(
                    sympy.series(row, Z, 0, DEGREE).removeO().coeff(Z, k)))
                if value != 0 and keep(i):
                    result[(i, j, k)] = value
    return result


@pytest.mark.parametrize("expression, expected", [
    ("sqrt(x^2*y^2)", False),
    ("(1 - sqrt(1 - x^2))/x", False),
    ("x^p/(1 - x)", False),
    ("log(1 - x/2)", False),
    ("exp(x)", True),
    ("exp(4*(x^2 - 1))*cosh(y*x)/(2 - x)", True),
    ("exp(sqrt(x))", False),
    ("x*sqrt(1 - y)/(1 - x*y)", True),
    ("(1 + x)^3/(2 - x)^2", True),
    ("sqrt(1 - y)", True),
])
def test_is_single_valued_in(expression, expected):
    assert is_single_valued_in(sympy.parse_expr(expression.replace("^", "**")), "x") is expected


def test_absolute_value_is_not_single_valued():
    x = sympy.Symbol("x", real=True)
    assert not is_single_valued_in(sympy.Abs(x) / (2 - x), "x")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", MULTI_VALUED)
@pytest.mark.parametrize("guard", ["c % 2 = 0", "c % 2 = 1", "c % 3 = 2"])
def test_modulo_guard_on_multi_valued_function_is_rejected(engine, expression, guard):
    with pytest.raises(NotSingleValuedError, match="not provably single-valued in 'c'"):
        engine(expression, *VARIABLES).filter(parse_expr(guard))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", MULTI_VALUED)
def test_modulo_update_on_multi_valued_function_is_rejected(engine, expression):
    with pytest.raises(NotSingleValuedError, match="not provably single-valued in 'c'"):
        engine(f"({expression})*z/(2 - z)", *VARIABLES)._update_modulo("z", "c", "2", None)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", MULTI_VALUED)
def test_arithmetic_progression_on_multi_valued_function_is_rejected(engine, expression):
    with pytest.raises(NotSingleValuedError):
        engine(expression, *VARIABLES)._arithmetic_progression("c", "2")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("expression", SINGLE_VALUED)
@pytest.mark.parametrize("modulus, remainder", [(2, 0), (2, 1), (3, 2)])
def test_modulo_guard_on_single_valued_function_matches_series(engine, expression, modulus, remainder):
    result = engine(expression, *VARIABLES).filter(parse_expr(f"c % {modulus} = {remainder}"))
    assert _coefficients(result) == _coefficients(expression, lambda i: i % modulus == remainder)


@pytest.mark.parametrize("engine", ENGINES)
def test_modulo_update_on_single_valued_function_matches_series(engine):
    result = engine("y/(2 - c)", *VARIABLES)._update_modulo("z", "c", "2", None)
    # c ~ Geometric(1/2) on 0, 1, 2, ... (times y): z = c % 2 is 0 with probability 2/3
    assert _coefficients(result) == _coefficients("y/(2 - c)*(2 + c*z)/(2 + c)")


# A symmetric random walk from 1; c and y count the steps. The EVT invariant is
# f + 2x(x - f)/(2x - u - u x^2) with u = c*y and f = (1 - sqrt(1 - u^2))/u, the generating function of
# the number of steps (always odd). The second form writes u as sqrt(c^2*y^2) in f, which denotes
# the same series; GiNaC accepts it and used to report Pr[c % 2 = 0] = 1 (the truth is 0).
WALK = """nat x; nat c; nat y;
x := 1
while (x > 0) { {x := x - 1} [1/2] {x := x + 1}; c := c + 1; y := y + 1 }
?Pr[c % 2 = 0]
"""


def _walk_invariant(f):
    return f"({f}) + 2*x*(x - ({f}))/(2*x - c*y - c*y*x^2)"


@pytest.mark.parametrize("engine, f", [
    (ForwardAnalysisConfig.Engine.SYMPY, "(1 - sqrt(1 - c^2*y^2))/(c*y)"),
    (ForwardAnalysisConfig.Engine.GINAC, "(1 - sqrt(1 - c^2*y^2))/(c*y)"),
    (ForwardAnalysisConfig.Engine.GINAC, "(1 - sqrt(1 - c^2*y^2))/sqrt(c^2*y^2)"),
])
def test_modulo_guard_after_loop_with_root_invariant_is_rejected(engine, f):
    config = ForwardAnalysisConfig(use_simplification=True, engine=engine)
    decisions = {"0": {"strategy": "evt_invariant", "params": {"invariant": _walk_invariant(f)}}}
    session = AnalysisSession(parse_pgcl(WALK), source=WALK, config=config, decisions=decisions,
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "error"
    assert "not provably single-valued in 'c'" in result["error"]["message"]
    assert not result["queries"]
