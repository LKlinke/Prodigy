"""``.pgcl`` was deprecated in v0.5.0 (#432) and removed in v0.6.0 (#433); the SymEngine engine, deprecated in
v0.5.0 as well, was removed in v0.6.0 (#416). The old Z3 solver went with the ``.pgcl`` parser (#474)."""
import re

import pytest
from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.deprecations import PGCL_REMOVED, REMOVED_ENGINES, PgclRemovedError
from prodigy.analysis.exceptions import ConfigurationError
from prodigy.analysis.session import AnalysisSession
from prodigy.analysis.solver.solver_type import SolverType
from prodigy.cli import cli
from prodigy.language.check import CheckFail
from prodigy.language.compiler import compile_program
from prodigy.language.parser import check_syntax, parse_program
from prodigy.web.app import create_app
from prodigy.web.requests import make_config, validate_request

PGCL = "nat x;\nx := 1;\n"
PPY = "x: nat;\nx = 1;\n"
SYMENGINE_REMOVED = "the SymEngine engine was removed in v0.6.0; use sympy or ginac"
Z3_REBUILT = "the Z3 solver is being rebuilt (#474); use sympy"


def test_the_message_names_the_release_and_the_migration():
    assert "v0.6.0" in PGCL_REMOVED and "v0.5.x" in PGCL_REMOVED and "prodigy convert" in PGCL_REMOVED
    for kwargs in ({}, {"syntax": "ppy"}, {"path": "a.ppy"}):
        check_syntax(**kwargs)
    for kwargs in ({"syntax": "pgcl"}, {"path": "a.pgcl"}, {"syntax": "ppy", "path": "a.pgcl"}):
        with pytest.raises(PgclRemovedError, match="removed in v0.6.0"):
            check_syntax(**kwargs)
    with pytest.raises(ValueError, match="Unknown syntax"):
        check_syntax("java")


@pytest.mark.parametrize("command", ["main", "analyze", "outline"])
def test_cli_rejects_a_pgcl_file(tmp_path, command):
    path = tmp_path / "a.pgcl"
    path.write_text(PGCL)
    result = CliRunner().invoke(cli, [command, str(path)])
    assert result.exit_code != 0
    assert PGCL_REMOVED in " ".join(result.output.split())
    assert "Traceback" not in result.output


def test_cli_rejects_the_pgcl_syntax_option(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(PPY)
    result = CliRunner().invoke(cli, ["main", "--syntax", "pgcl", str(path)])
    assert result.exit_code != 0 and "removed in v0.6.0" in result.output
    assert CliRunner().invoke(cli, ["--engine", "sympy", "main", str(path)]).exit_code == 0


def test_cli_has_no_convert_command(tmp_path):
    path = tmp_path / "a.pgcl"
    path.write_text(PGCL)
    result = CliRunner().invoke(cli, ["convert", str(path)])
    assert result.exit_code == 2 and "No such command" in result.output


def test_gui_requests_reject_pgcl():
    for request in ({"source": PGCL, "syntax": "pgcl"}, {"source": PGCL, "file_name": "a.pgcl"}):
        assert PGCL_REMOVED in validate_request(request)
    assert validate_request({"source": PPY}) == []


def test_gui_check_shows_the_message_as_an_error():
    client = create_app(time_limit=30, max_jobs=1).test_client()
    data = client.post("/api/check", json={"source": PGCL, "syntax": "pgcl"},
                       headers={"Host": "localhost"}).get_json()
    assert not data["ok"] and data["diagnostics"][0]["severity"] == "error"
    assert data["diagnostics"][0]["message"] == PGCL_REMOVED
    assert client.post("/api/convert", json={"source": PGCL}, headers={"Host": "localhost"}).status_code in (404, 405)


def test_choosing_z3_ends_with_the_rebuild_message(tmp_path):
    with pytest.raises(ValueError, match=re.escape(Z3_REBUILT)):
        SolverType.named("Z3")
    assert not hasattr(SolverType, "Z3")
    path = tmp_path / "a.ppy"
    path.write_text(PPY)
    result = CliRunner().invoke(cli, ["--solver", "z3", "main", str(path)])
    assert result.exit_code == 2 and Z3_REBUILT in " ".join(result.output.split())
    assert any(Z3_REBUILT in problem for problem in validate_request({"source": PPY, "solver": "z3"}))


@pytest.mark.parametrize("name", ["symengine", "SymEngine"])
def test_the_removed_engine_says_so_in_the_api(name):
    assert REMOVED_ENGINES["symengine"] == SYMENGINE_REMOVED
    with pytest.raises(ConfigurationError) as error:
        ForwardAnalysisConfig.Engine.from_name(name)
    assert str(error.value) == SYMENGINE_REMOVED
    assert not hasattr(ForwardAnalysisConfig.Engine, "SYMENGINE")


@pytest.mark.parametrize("name, engine", [("", "SYMPY"), ("sympy", "SYMPY"), ("GiNaC", "GINAC")])
def test_the_engines_by_name(name, engine):
    assert ForwardAnalysisConfig.Engine.from_name(name) == ForwardAnalysisConfig.Engine[engine]


def test_an_unknown_engine_is_an_error_not_sympy():
    with pytest.raises(ConfigurationError, match="Unknown engine 'maple'; use sympy or ginac"):
        ForwardAnalysisConfig.Engine.from_name("maple")


@pytest.mark.parametrize("command", ["main", "analyze"])
def test_cli_engine_symengine_ends_with_the_removal_message(tmp_path, command):
    path = tmp_path / "a.ppy"
    path.write_text(PPY)
    result = CliRunner().invoke(cli, ["--engine", "symengine", command, str(path)])
    assert result.exit_code == 2
    assert SYMENGINE_REMOVED in result.output
    assert "Traceback" not in result.output


def test_gui_request_with_symengine_is_refused_with_the_removal_message():
    assert SYMENGINE_REMOVED in validate_request({"source": PPY, "syntax": "ppy", "engine": "symengine"})
    with pytest.raises(ConfigurationError, match="removed in v0.6.0"):
        make_config({"engine": "symengine"})


def test_direct_access_to_a_removed_engine_gives_the_removal_message():
    # #464: the enum has no member SYMENGINE any more
    with pytest.raises(AttributeError, match="SymEngine engine was removed"):
        _ = ForwardAnalysisConfig.Engine.SYMENGINE
    with pytest.raises(AttributeError, match="^FOO$"):
        _ = ForwardAnalysisConfig.Engine.FOO


def test_session_syntax_argument_is_deprecated():
    program = parse_program(PPY)
    with pytest.warns(DeprecationWarning, match="removed in v0.8.0"):
        AnalysisSession(program, source=PPY, syntax="ppy")
    with pytest.warns(DeprecationWarning), pytest.raises(PgclRemovedError):
        AnalysisSession(program, source=PPY, syntax="pgcl")


def test_messages_outside_an_analysis_are_ppy():
    # the type checker spells expressions in .ppy also when no analysis runs (#489)
    failure = compile_program("x: nat;\ny: nat;\nx = iid(bernoulli(1/2), y == 1);")
    assert isinstance(failure, CheckFail)
    assert "iid(bernoulli(1/2), y == 1)" in failure.message and "got y == 1." in failure.message
