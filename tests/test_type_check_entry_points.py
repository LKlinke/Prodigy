"""Every entry point type-checks the program, not only `prodigy analyze` (#351)."""
from click.testing import CliRunner

from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl

# `y` is no declared variable
BAD = "nat x;\nx := y\n"


def test_cli_commands_reject_type_errors(tmp_path):
    path = tmp_path / "bad.pgcl"
    path.write_text(BAD)
    for command in ("convert", "outline", "main"):
        result = CliRunner().invoke(cli, [command, str(path)])
        assert result.exit_code != 0, command
        assert "type error" in result.output, command


def test_check_equality_rejects_type_errors(tmp_path):
    bad = tmp_path / "bad.pgcl"
    bad.write_text(BAD)
    good = tmp_path / "good.pgcl"
    good.write_text("nat x;\nx := 1\n")
    result = CliRunner().invoke(cli, ["check_equality", str(good), str(bad)])
    assert result.exit_code != 0
    assert "type error" in result.output


def test_session_reports_type_errors_in_the_result():
    result = AnalysisSession(parse_pgcl(BAD), source=BAD).run()
    assert result.status == "error"
    assert "Type error" in result.error["message"]
