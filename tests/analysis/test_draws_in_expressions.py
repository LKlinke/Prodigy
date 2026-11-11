"""
Draws inside expressions (issue #178): ``x = geometric(1/2) + 1`` draws into a fresh helper
variable, assigns the expression and marginalises the helper (:mod:`prodigy.pgcl.draws`), on every
engine and in both syntaxes. Every occurrence is a draw of its own. The expected generating
functions are computed by hand. A draw inside a guard, an observation or a query ends with a clear
"not implemented" message that names it.
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


def _distribution(source: str, engine, syntax: str):
    program = compile_pgcl(source, syntax=syntax)
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine)
    with config.activate():
        result, error = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    return result, error, config.factory


# (.ppy, .pgcl, expected generating function)
CASES = [
    pytest.param("x: nat;\nx = geometric(1/2) + 1;",
                 "nat x;\nx := geometric(1/2) + 1;",
                 "x / (2 - x)", id="issue"),
    pytest.param("x: nat;\nx = geometric(1/2) + geometric(1/2);",
                 "nat x;\nx := geometric(1/2) + geometric(1/2);",
                 "1 / (2 - x)^2", id="sum-of-two-independent-draws"),
    pytest.param("x: nat;\nx = bernoulli(1/2) * binomial(3, 1/2) * 2;",
                 "nat x;\nx := bernoulli(1/2) * binomial(3, 1/2) * 2;",
                 "9/16 + 3/16*x^2 + 3/16*x^4 + 1/16*x^6", id="product"),
    pytest.param("x: nat;\nn: nat;\nn = unif(1, 3);\nx = binomial(n, 1/2) + n;",
                 "nat x;\nnat n;\nn := unif(1, 3);\nx := binomial(n, 1/2) + n;",
                 "1/3*n*x*(1/2 + x/2) + 1/3*n^2*x^2*(1/2 + x/2)^2 + 1/3*n^3*x^3*(1/2 + x/2)^3",
                 id="program-variable-argument"),
    pytest.param("x: nat;\nx = binomial(unif(1, 3), 1/2) + 1;",
                 "nat x;\nx := binomial(unif(1, 3), 1/2) + 1;",
                 "x * (1/3*(1/2 + x/2) + 1/3*(1/2 + x/2)^2 + 1/3*(1/2 + x/2)^3)",
                 id="draw-as-argument"),
    pytest.param("x: nat;\nx = binomial(unif(1, 3), 1/2);",
                 "nat x;\nx := binomial(unif(1, 3), 1/2);",
                 "1/3*(1/2 + x/2) + 1/3*(1/2 + x/2)^2 + 1/3*(1/2 + x/2)^3",
                 id="draw-as-argument-of-the-right-hand-side"),
    pytest.param("x: nat;\nx = iid(bernoulli(1/2), geometric(1/2));",
                 "nat x;\nx := iid(bernoulli(1/2), geometric(1/2));",
                 "2 / (3 - x)", id="iid-count"),
    pytest.param("x: nat;\nx = sample_pgf(1/2 + x/2, x) + 1;",
                 "nat x;\nx := sample_pgf(1/2 + x/2, x) + 1;",
                 "x/2 + x^2/2", id="sample-pgf"),
    pytest.param("x: nat;\nx = 2;\nx = geometric(1/2) + x;",
                 "nat x;\nx := 2;\nx := geometric(1/2) + x;",
                 "x^2 / (2 - x)", id="reads-the-assigned-variable"),
    pytest.param("x: nat;\nloop(2) { x = x + 2 * bernoulli(1/2); }",
                 "nat x;\nloop(2) { x := x + 2 * bernoulli(1/2) }",
                 "(1/2 + x^2/2)^2", id="in-a-loop"),
    pytest.param("x: nat;\ny: nat;\ny = 1;\nx = categorical({geometric(1/2) + y: 1/2, 0: 1/2});",
                 "nat x;\nnat y;\ny := 1;\nx := (geometric(1/2) + y) : 1/2 + 0 : 1/2;",
                 "y/2 + y * x / (2 * (2 - x))", id="inside-categorical"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
@pytest.mark.parametrize("ppy, pgcl, expected", CASES)
def test_draws_in_assignments(engine, syntax, ppy, pgcl, expected):
    result, error, factory = _distribution(ppy if syntax == "ppy" else pgcl, engine, syntax)
    assert error.is_zero_dist()
    # the helper variables are marginalised
    assert result.get_variables() == set(compile_pgcl(ppy, syntax="ppy").variables)
    assert result == factory.from_expr(expected, *result.get_variables()), str(result)


@pytest.mark.parametrize("engine", ENGINES)
def test_function_call_inside_an_expression(engine):
    source = ("def f(a: nat) -> nat {\n    return 2 * a;\n}\n"
              "x: nat;\nx = f(geometric(1/2)) + f(1);")
    result, _, factory = _distribution(source, engine, "ppy")
    assert result == factory.from_expr("x^2 / (2 - x^2)", "x"), str(result)


@pytest.mark.parametrize("engine", ENGINES)
def test_draw_in_a_function_body(engine):
    source = ("def f(a: nat) -> nat {\n    return a + bernoulli(1/2);\n}\n"
              "x: nat;\nx = f(1);")
    result, _, factory = _distribution(source, engine, "ppy")
    assert result == factory.from_expr("x/2 + x^2/2", "x"), str(result)


@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
@pytest.mark.parametrize("ppy, pgcl, where", [
    pytest.param("x: nat;\nif (geometric(1/2) > 2) { x = 1; }",
                 "nat x;\nif (geometric(1/2) > 2) { x := 1 } else { skip }", "the guard of if", id="if"),
    pytest.param("x: nat;\nwhile (x + geometric(1/2) < 2) { x = x + 1; }",
                 "nat x;\nwhile (x + geometric(1/2) < 2) { x := x + 1 }", "the guard of while",
                 id="while"),
    pytest.param("x: nat;\nobserve(geometric(1/2) > 2);",
                 "nat x;\nobserve(geometric(1/2) > 2);", "observe", id="observe"),
    pytest.param("x: nat;\nquery Pr[geometric(1/2) > x];",
                 "nat x;\n?Pr[geometric(1/2) > x];", "a query", id="query"),
])
def test_draws_in_conditions_are_not_implemented(syntax, ppy, pgcl, where):
    program = compile_pgcl(ppy if syntax == "ppy" else pgcl, syntax=syntax)
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=Engine.SYMPY)
    with pytest.raises(NotImplementedError) as info:
        with config.activate():
            compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    message = str(info.value)
    assert f"Sampling inside {where}" in message
    assert "geometric(1/2)" in message
    assert "not implemented" in message


def test_message_about_a_helper_names_the_draw():
    program = compile_pgcl("x: nat;\nx = binomial(geometric(1/2), 1/2);", syntax="ppy")
    config = ForwardAnalysisConfig(engine=Engine.SYMPY)
    with pytest.raises(NotImplementedError) as info:
        with config.activate():
            compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    message = str(info.value)
    assert "binomial(geometric(1/2), 1/2)" in message
    assert "x_0" not in message
