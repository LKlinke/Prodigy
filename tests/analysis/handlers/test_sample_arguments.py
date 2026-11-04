"""
The arguments of ``iid`` and ``sample_pgf`` (issue #140): a natural number of samples is summed up
exactly on every engine; counts and variables the analysis cannot handle are rejected by the type
checker instead of crashing the analysis or giving a wrong result.
"""
import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl.check import CheckFail
from prodigy.pgcl.compiler import compile_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]


def _run(source: str, engine: ForwardAnalysisConfig.Engine):
    program = compile_pgcl(source)
    assert not isinstance(program, CheckFail), program
    config = ForwardAnalysisConfig(engine=engine)
    result, error_prob = compute_discrete_distribution(
        program, config.factory.one(*program.variables.keys()), config)
    return result, error_prob, config.factory


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    # the sum of three geometric samples is negative binomial
    ("nat x; x := iid(geometric(1/2), 3)", "(1/(2-x))^3"),
    ("nat x; x := iid(bernoulli(1/3), 2)", "(2/3 + x/3)^2"),
    ("nat x; x := iid(unif(1, 2), 2)", "(x/2 + x^2/2)^2"),
    # the distribution as a generating function in the assigned variable
    ("nat x; x := iid(1/2*x + 1/2, 3)", "(1/2 + x/2)^3"),
    # constants are substituted before the analysis
    ("nat x; const N := 3; x := iid(bernoulli(1/2), N)", "(1/2 + x/2)^3"),
    # zero samples sum up to zero
    ("nat x; x := iid(geometric(1/2), 0)", "1"),
    # x is overwritten, y is kept
    ("nat x; nat y; x := 5; y := 1; x := iid(bernoulli(1/2), 2)", "y * (1/2 + x/2)^2"),
    ("nat x; x := 7; x := sample_pgf(1/2*x + 1/2, x)", "1/2 + x/2"),
])
def test_accepted_forms_are_exact(engine, source, expected):
    result, error_prob, factory = _run(source, engine)
    variables = [str(var) for var in result.get_variables()]
    assert result == factory.from_expr(expected, *variables)
    assert error_prob.is_zero_dist()


@pytest.mark.parametrize("engine", ENGINES)
def test_literal_count_agrees_with_variable_count(engine):
    literal, _, _ = _run("nat x; nat n; n := 3; x := iid(geometric(1/3), 3); n := 0", engine)
    variable, _, _ = _run("nat x; nat n; n := 3; x := iid(geometric(1/3), n); n := 0", engine)
    assert literal == variable


@pytest.mark.parametrize("source, message", [
    ("nat x; x := sample_pgf(1/2, 3)", "must be the program variable"),
    ("nat x; x := sample_pgf(1/2*z + 1/2, z)", "must be the program variable"),
    ("nat x; nat y; x := sample_pgf(1/2*y + 1/2, y)", "only sampling in the assigned variable x"),
    ("nat x; nat y; x := iid(geometric(1/2), y + 1)", "must be a program variable or a natural number"),
    # a parameter count used to give the point mass at 0 on every engine
    ("nat x; nparam n; x := iid(bernoulli(1/2), n)", "must be a program variable or a natural number"),
    ("nat x; x := iid(geometric(1/2))", "Too few parameters"),
    ("nat x; x := sample_pgf(1/2)", "Too few parameters"),
    ("nat x; x := geometric()", "Too few parameters"),
])
def test_rejected_forms(source, message):
    result = compile_pgcl(source)
    assert isinstance(result, CheckFail)
    assert message in result.message
