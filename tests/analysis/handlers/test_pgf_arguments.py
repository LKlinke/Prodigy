"""
The generating functions given to ``sample_pgf`` and ``iid`` must be probability generating functions:
non-negative coefficients and mass 1 (#424). Before, ``sample_pgf(2*y, y)`` gave the "distribution" ``2*y``
of mass 2 as an exact result. A refuted argument is a type error if the type checker can tell, else an error of
the analysis; an argument that is not proven gives a warning, and without parameters a result without guarantee.
"""
import pytest
from click.testing import CliRunner

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.finite_measure import check_probability_generating_function
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.language.check import CheckFail
from prodigy.language.compiler import compile_program, parse_program

Engine = ForwardAnalysisConfig.Engine

ENGINES = [
    pytest.param(Engine.SYMPY, id="SymPy"),
    pytest.param(Engine.GINAC, id="GiNaC"),
    # feature-frozen (#416), but must not report a wrong result either
]

# the square of 7/13 * (16*i - 3*i^3) / (16 - 8*i - i^2) (from the invariant of
# pgfexamples/equivalence/loopy/bit_flip_conditioning.ppy, proven by its unique dominant simple pole, #431) has
# non-negative coefficients and mass 1, but its dominant pole is double, so Prodigy cannot prove it
UNPROVEN = "(7/13 * (3 * i ** 3 - 16 * i) / (i ** 2 + 8 * i - 16)) ** 2"


def _analyse(source: str, engine: Engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_program(source, "ppy"), source=source, syntax="ppy", config=config,
                              capture_output=True)
    return session.run().to_json()


@pytest.mark.parametrize("text, variables, verdict, reason", [
    ("1/2 + x/2", ("x",), True, ""),
    ("x / (2 - x)", ("x",), True, ""),
    ("exp(3 * (x - 1))", ("x",), True, ""),
    ("1 - (1 - x)^(1/2)", ("x",), True, ""),
    # a removable singularity at 1: the mass is the limit
    ("(1 - x^3) / (1 - x) / 3", ("x",), True, ""),
    ("2*y", ("y",), False, "its mass (its value at 1) is 2, not 1"),
    ("1/2", ("x",), False, "its mass (its value at 1) is 1/2, not 1"),
    ("-y + 2*y^2", ("y",), False, "the coefficient of y is -1, which is negative"),
    ("1/(1 - x)", ("x",), False, "infinite"),
    # not proven to have non-negative coefficients, but the value at 1 already refutes it (Abel's theorem)
    ("1/(1 + x)", ("x",), False, "the coefficient of x is -1, which is negative"),
    ("1/(3 - x)", ("x",), False, "its mass (its value at 1) is 1/2, not 1"),
    (UNPROVEN.replace("**", "^"), ("i",), None, "non-negativity of its coefficients could not be proven"),
    ("p*y + 1 - p", ("y",), None, "parameter p"),
])
def test_check_probability_generating_function(text, variables, verdict, reason):
    actual, message = check_probability_generating_function(text, variables)
    assert actual is verdict, message
    assert reason in message


@pytest.mark.parametrize("source, expression", [
    ("y: nat;\ny = sample_pgf(2*y, y);", "2 * y"),
    ("y: nat;\ny = iid(2*y, 3);", "2 * y"),
    ("x: nat;\nx = sample_pgf(1/2, x);", "1/2"),
    ("y: nat;\ny = sample_pgf(-y + 2*y*y, y);", "y"),
    ("x: nat;\nx = iid(1/(1 + x), 2);", "x"),
    # a draw inside an expression and the distribution of iid are checked as well
    ("x: nat;\nx = sample_pgf(2*x, x) + 1;", "2"),
    ("x: nat;\ny: nat;\nx = 2;\ny = iid(sample_pgf(-y + 2*y*y, y), x);", "y"),
    # a refutation holds for every value of a parameter
    ("param p: real;\ny: nat;\ny = sample_pgf(p*y - y*y + 1, y);", "y"),
])
def test_refuted_generating_functions_are_type_errors(source, expression):
    result = compile_program(source, syntax="ppy")
    assert isinstance(result, CheckFail), result
    assert "needs a probability generating function" in result.message
    assert "is not one" in result.message and expression in result.message


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, reason", [
    ("""y: nat;
y = sample_pgf(2 * y, y);""", "its mass (its value at 1) is 2, not 1"),
    ("""y: nat;
y = iid(2 * y, 3);""", "its mass (its value at 1) is 2, not 1"),
    ("""y: nat;
y = sample_pgf(2 * y * y - y, y);""", "which is negative"),
])
def test_refuted_generating_functions_without_type_check(engine, source, reason):
    """Programs that did not pass the type checker (Python API) are checked by the analysis, on every engine."""
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    program = parse_program(source)
    config = ForwardAnalysisConfig(engine=engine)
    with pytest.raises(ValueError, match="needs a probability generating function") as error:
        compute_discrete_distribution(program, config.factory.one(*program.variables.keys()), config)
    assert reason in str(error.value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("call", [f"sample_pgf({UNPROVEN}, i)", f"iid({UNPROVEN}, 2)"])
def test_unproven_generating_function_has_no_guarantee(engine, call):
    result = _analyse(f"i: nat;\ni = {call};", engine)
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "unknown"
    assert any("could not prove that" in warning and "no guarantee" in warning for warning in result["warnings"])


@pytest.mark.parametrize("engine", ENGINES)
def test_parametric_generating_function_holds_for_its_parameter_values(engine):
    """As for geometric(p), a parametric result holds for the parameter values for which the arguments are
    admissible; parameters have no declared range yet (#303), so this is not proven, and a warning says so."""
    result = _analyse("param p: real;\ny: nat;\ny = sample_pgf(p*y + 1 - p, y);", engine)
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "exact"
    assert any("could not prove that" in warning and "the parameter p for which it is one" in warning
               for warning in result["warnings"])


@pytest.mark.parametrize("engine", ENGINES)
def test_proven_generating_function_has_no_warning(engine):
    result = _analyse("x: nat;\nn: nat;\nn = 3;\nx = iid(x / (2 - x), n);\nx = sample_pgf(1/2 + x/2, x) + x;",
                      engine)
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "exact"
    assert not result["warnings"]


def test_main_prints_the_warning(tmp_path):
    program = tmp_path / "unproven.ppy"
    program.write_text(f"i: nat;\ni = sample_pgf({UNPROVEN}, i);\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["main", str(program)])
    assert result.exit_code == 0, result.output
    assert "Warning:" in result.output and "no guarantee" in result.output
