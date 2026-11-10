"""
``x = categorical({v1: p1, …})`` (issue #233): the choice between the assignments ``x = v_i`` with
the probabilities ``p_i``, on every engine, with and without simplification. The expected
generating functions are computed by hand; the comparisons with nested choices check the same
semantics on programs where the values depend on the state, in functions and with observations.
"""
import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl import Program
from prodigy.pgcl.compiler import compile_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

Engine = ForwardAnalysisConfig.Engine
GINAC = pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")
ENGINES = [pytest.param(Engine.SYMPY, id="SymPy"), pytest.param(Engine.SYMENGINE, id="SymEngine"),
           pytest.param(Engine.GINAC, id="GiNaC", marks=GINAC)]
SIMPLIFICATION = [pytest.param(True, id="simplified"), pytest.param(False, id="unsimplified")]


def _distribution(source: str, engine, simplification: bool, syntax: str = "ppy"):
    program = compile_pgcl(source, syntax=syntax)
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine, use_simplification=simplification)
    with config.activate():
        result, error = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    return result, error, config.factory


def _equal(result, factory, expected: str) -> bool:
    return result == factory.from_expr(expected, *result.get_variables())


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    pytest.param("x: nat;\nx = categorical({1: 1/2, 2: 1/2});", "x/2 + x^2/2", id="issue"),
    pytest.param("x: nat = categorical({1: 1/6, 4: 5/6});", "x/6 + 5/6*x^4", id="initial-value"),
    pytest.param("x: nat = categorical(1/4, 0, 3/4);", "1/4 + 3/4*x^2", id="short-form"),
    pytest.param("x: nat = categorical({3: 1});", "x^3", id="one-value"),
    pytest.param("x: nat = categorical({2: 0.5, 2: 0.5});", "x^2", id="repeated-value"),
    pytest.param("x: nat = 7;\nx = categorical({0: 1/3, 1: 2/3});", "1/3 + 2/3*x", id="overwrites"),
    pytest.param("y: nat = unif(0, 1);\nx: nat = categorical({y: 1/2, y + 2: 1/2});",
                 "1/4 + x*y/4 + x^2/4 + x^3*y/4", id="state-dependent-values"),
    pytest.param("x: nat = unif(0, 1);\nx = categorical({x + 1: 1/2, 0: 1/2});",
                 "1/2 + x/4 + x^2/4", id="value-reads-the-variable"),
    pytest.param("b: bool = categorical({0: 1/3, 1: 2/3});", "1/3 + 2/3*b", id="bool"),
    pytest.param("x: nat = geometric(1/2);\ny: nat = categorical({1: 1/2, 2: 1/2});",
                 "(y/2 + y^2/2) / (2 - x)", id="infinite-support"),
    pytest.param("x: nat;\nloop(2) { x = categorical({x: 1/2, x + 1: 1/2}); }",
                 "1/4 + x/2 + x^2/4", id="loop"),
])
def test_assignments(engine, simplification, source, expected):
    result, error, factory = _distribution(source, engine, simplification)
    assert _equal(result, factory, expected)
    assert error.is_zero_dist()


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
def test_pgcl_syntax(engine, simplification):
    source = "nat x;\nnat y;\ny := 2 : 1/2 + 5 : 1/2;\nx := y : 1/3 + 7 : 2/3;\n"
    result, _, factory = _distribution(source, engine, simplification, syntax="pgcl")
    assert _equal(result, factory, "x^2*y^2/6 + x^5*y^5/6 + x^7*y^2/3 + x^7*y^5/3")


FUNCTIONS = "def f(a: nat) -> nat {\n    observe(a < 3);\n    return categorical({a: 1/2, a + 1: 1/2});\n}\n"


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("categorical, choice", [
    pytest.param("x: nat = categorical({y + 1: 1/3, 0: 1/6, 2 * y: 1/2});",
                 "x: nat;\n{ x = y + 1; } [1/3] { { x = 0; } [1/4] { x = 2 * y; } }", id="assignment"),
    pytest.param(FUNCTIONS + "x: nat = f(categorical({1: 1/2, 3: 1/2}));",
                 FUNCTIONS + "x: nat;\n{ x = f(1); } [1/2] { x = f(3); }", id="argument-and-return"),
    pytest.param("x: nat = categorical({y: 1/2, 1: 1/2});\nobserve(x == 1);",
                 "x: nat;\n{ x = y; } [1/2] { x = 1; }\nobserve(x == 1);", id="observe"),
])
def test_same_as_nested_choice(engine, simplification, categorical, choice):
    prefix = "y: nat = unif(0, 2);\n"
    result, error, factory = _distribution(prefix + categorical, engine, simplification)
    expected, expected_error, _ = _distribution(prefix + choice, engine, simplification)
    variables = result.get_variables() | expected.get_variables()
    assert result == factory.from_expr(str(expected), *variables)
    assert error == factory.from_expr(str(expected_error), *variables)
