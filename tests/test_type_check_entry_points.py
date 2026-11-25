"""Every entry point type-checks the program, not only `prodigy analyze` (#351)."""
import pytest
from click.testing import CliRunner

from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.language import parse_program
from prodigy.language.compiler import compile_program

# `y` is no declared variable
BAD = """x: nat;
x = y;
"""


def test_cli_commands_reject_type_errors(tmp_path):
    path = tmp_path / "bad.ppy"
    path.write_text(BAD)
    for command in ("outline", "main"):
        result = CliRunner().invoke(cli, [command, str(path)])
        assert result.exit_code != 0, command
        assert "type error" in result.output, command


def test_check_equality_rejects_type_errors(tmp_path):
    bad = tmp_path / "bad.ppy"
    bad.write_text(BAD)
    good = tmp_path / "good.ppy"
    good.write_text("""x: nat;
x = 1;
""")
    result = CliRunner().invoke(cli, ["check_equality", str(good), str(bad)])
    assert result.exit_code != 0
    assert "type error" in result.output


def test_session_reports_type_errors_in_the_result():
    result = AnalysisSession(parse_program(BAD), source=BAD).run()
    assert result.status == "error"
    assert "Type error" in result.error["message"]


# the probability 5/k is 5/3 for k = 3 (#425)
BAD_PROBABILITY_PPY = "x: nat;\nk: nat;\nk = 3;\n{ x = 1; } [5 / k] { x = 2; }\n"


@pytest.mark.parametrize("engine", ["sympy", "ginac"])
@pytest.mark.parametrize("command", ["main", "analyze"])
def test_invalid_choice_probability_is_a_message_not_a_traceback(tmp_path, engine, command):
    if engine == "ginac":
        pytest.importorskip("prodigy_ginac")
    path = tmp_path / "bad.ppy"
    path.write_text(BAD_PROBABILITY_PPY)
    result = CliRunner().invoke(cli, ["--engine", engine, command, str(path)])
    assert result.exit_code != 0
    assert "not a probability" in result.output
    assert not isinstance(result.exception, ValueError)  # no traceback


def test_type_errors_spell_expressions_in_the_syntax_of_the_program():
    ppy = "x: nat;\ny: nat;\nx = iid(1/2, y * 2);\n"
    failure = compile_program(ppy, syntax="ppy")
    assert "y * 2" in failure.message and "iid(1/2, y * 2)" in failure.message
    assert "(y) * (2)" not in failure.message


# a refuted generating function (#424) is rejected by the check of the analysis that every entry path runs after
# the type checker (T3 of docs/source/architecture.md), in the program and in a function
REFUTED_PGF = ["x: nat;\nx = sample_pgf(2*x, x);\n",
               "def f() -> nat {\n  y: nat;\n  y = sample_pgf(2*y, y);\n  return y;\n}\nx: nat;\nx = f();\n"]
REFUTED_MESSAGE = "needs a probability generating function"


@pytest.mark.parametrize("source", REFUTED_PGF)
@pytest.mark.parametrize("command", ["outline", "main", "analyze"])
def test_cli_commands_reject_refuted_generating_functions(tmp_path, source, command):
    path = tmp_path / "bad.ppy"
    path.write_text(source)
    result = CliRunner().invoke(cli, [command, str(path)])
    assert result.exit_code != 0, command
    assert "type error" in result.output and REFUTED_MESSAGE in result.output, result.output


@pytest.mark.parametrize("source", REFUTED_PGF)
def test_session_rejects_refuted_generating_functions(source):
    result = AnalysisSession(parse_program(source), source=source).run()
    assert result.status == "error"
    assert "Type error" in result.error["message"] and REFUTED_MESSAGE in result.error["message"]


@pytest.mark.parametrize("source", REFUTED_PGF)
def test_gui_check_rejects_refuted_generating_functions(source):
    from prodigy.web.diagnostics import check_source  # pylint: disable=import-outside-toplevel
    diagnostics = check_source(source)["diagnostics"]
    assert any(REFUTED_MESSAGE in d["message"] for d in diagnostics), diagnostics
