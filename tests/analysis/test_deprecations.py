"""``.pgcl`` is deprecated in v0.5.0 and removed in v0.6.0 (#432); the SymEngine engine, deprecated in v0.5.0
as well, was removed in v0.6.0 (#416)."""
import json

import pytest
from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.deprecations import PGCL_DEPRECATION, REMOVED_ENGINES, deprecation_warnings
from prodigy.analysis.exceptions import ConfigurationError
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl
from prodigy.web.requests import make_config, validate_request

PGCL = "nat x;\nx := 1;\n"
PPY = "x: nat;\nx = 1;\n"
SYMENGINE_REMOVED = "the SymEngine engine was removed in v0.6.0; use sympy or ginac"


def test_the_warnings_name_the_release_and_the_migration():
    assert "v0.6.0" in PGCL_DEPRECATION and "prodigy convert" in PGCL_DEPRECATION
    assert deprecation_warnings("ppy") == []
    assert deprecation_warnings("pgcl") == [PGCL_DEPRECATION]


@pytest.mark.parametrize("source, syntax, expected", [
    (PGCL, "pgcl", [PGCL_DEPRECATION]),
    (PPY, "ppy", []),
])
def test_the_result_carries_the_deprecation_warnings(source, syntax, expected):
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.SYMPY)
    result = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config).run()
    assert result.status == "ok"
    assert [w for w in result.to_json()["warnings"] if "deprecated" in w] == expected


@pytest.mark.parametrize("name, source, expected", [
    ("a.pgcl", PGCL, True),
    ("a.ppy", PPY, False),
])
def test_cli_main_warns_on_stderr(tmp_path, name, source, expected):
    path = tmp_path / name
    path.write_text(source)
    result = CliRunner().invoke(cli, ["--engine", "sympy", "main", str(path)])
    assert result.exit_code == 0
    assert ("The .pgcl syntax is deprecated" in result.output) == expected


def test_cli_analyze_json_stays_clean_and_has_the_warning(tmp_path):
    path = tmp_path / "a.pgcl"
    path.write_text(PGCL)
    result = CliRunner().invoke(cli, ["analyze", str(path), "--json"])
    assert result.exit_code == 0
    assert PGCL_DEPRECATION in json.loads(result.output)["warnings"]


def test_convert_does_not_warn_about_the_syntax_it_converts(tmp_path):
    path = tmp_path / "a.pgcl"
    path.write_text(PGCL)
    result = CliRunner().invoke(cli, ["convert", str(path)])
    assert result.exit_code == 0
    assert "deprecated" not in result.output


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
