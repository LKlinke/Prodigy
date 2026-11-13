"""Engines report the same error message, in terms of the program (#349)."""
import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover
    HAS_GINAC = False

ENGINES = [
    ForwardAnalysisConfig.Engine.SYMPY,
    ForwardAnalysisConfig.Engine.SYMENGINE,
    pytest.param(ForwardAnalysisConfig.Engine.GINAC,
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]

VARIABLE_DIVISOR = "a: nat;\nb: nat;\nx: nat;\na = 2;\nb = poisson(3);\nx = b // a;\nquery Ex[x];\n"
OBSERVE_ZERO = "x: nat;\ny: nat;\nx = poisson(2);\ny = x % 3;\nobserve(y == 5);\nquery Ex[x];\n"


def _message(source: str, engine) -> str:
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              config=ForwardAnalysisConfig(engine=engine))
    result = session.run().to_json()
    assert result["status"] == "error", result
    return result["error"]["message"]


@pytest.mark.parametrize("source", [VARIABLE_DIVISOR, OBSERVE_ZERO], ids=["variable-divisor", "observe-zero"])
def test_engines_give_the_same_message(source):
    messages = {str(engine): _message(source, engine) for engine in
                [p.values[0] if hasattr(p, "values") else p for p in ENGINES]
                if engine != ForwardAnalysisConfig.Engine.GINAC or HAS_GINAC}
    assert len(set(messages.values())) == 1, messages


def test_variable_divisor_passes_the_type_check_and_is_not_implemented():
    # syntax and semantics are separate (#356): the analysis ends with a clear "not implemented"
    message = _message(VARIABLE_DIVISOR, ForwardAnalysisConfig.Engine.SYMPY)
    assert "Cannot compute b // a" in message and "not implemented" in message, message


@pytest.mark.parametrize("source, construct", [
    ("x: nat;\nx = 7;\nx = x // 1.5;\nquery Ex[x];\n", "x // 1.5"),
    ("x: nat;\nparam p: real;\nx = poisson(3);\nx = p // 2;\nquery Ex[x];\n", "p // 2"),
    ("x: nat;\nparam p: real;\nx = poisson(3);\nx = x // p;\nquery Ex[x];\n", "x // p"),
], ids=["real-divisor", "real-dividend", "real-parameter-divisor"])
@pytest.mark.parametrize("engine", ENGINES)
def test_real_floor_division_passes_the_type_check_and_is_not_implemented(source, construct, engine):
    # #372: accepted by the type checker, "not implemented" in the analysis
    message = _message(source, engine)
    assert f"Cannot compute {construct}" in message and "not implemented" in message, message


def test_variable_divisor_is_named_as_written_by_the_engine():
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.SYMPY)
    program = parse_pgcl(VARIABLE_DIVISOR, "ppy")
    with AnalysisSession(program, source=VARIABLE_DIVISOR, syntax="ppy", config=config).activate(), \
            config.activate(), pytest.raises(Exception, match="Cannot compute b // a:"):
        compute_discrete_distribution(program, config.factory.one(*program.variables), config)


def test_cli_analyze_reports_type_errors(tmp_path):
    # pylint: disable=import-outside-toplevel
    from click.testing import CliRunner
    from prodigy.cli import cli
    program = tmp_path / "bad.ppy"
    program.write_text("x: nat;\nx = z + 1;\nquery Ex[x];\n")
    result = CliRunner().invoke(cli, ["analyze", str(program)])
    assert result.exit_code != 0
    assert "type error" in result.output and "bad.ppy" in result.output, result.output
