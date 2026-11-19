"""``.pgcl`` and SymEngine are deprecated in v0.5.0 and removed in v0.6.0 (#432, #416)."""
import json

import pytest
from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.deprecations import PGCL_DEPRECATION, SYMENGINE_DEPRECATION, deprecation_warnings
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl

PGCL = "nat x;\nx := 1;\n"
PPY = "x: nat;\nx = 1;\n"


def test_the_warnings_name_the_release_and_the_migration():
    assert "v0.6.0" in PGCL_DEPRECATION and "prodigy convert" in PGCL_DEPRECATION
    assert "v0.6.0" in SYMENGINE_DEPRECATION and "sympy" in SYMENGINE_DEPRECATION.lower()
    assert deprecation_warnings("ppy", "SYMPY") == []
    assert deprecation_warnings("pgcl", "GINAC") == [PGCL_DEPRECATION]
    assert deprecation_warnings("ppy", "SYMENGINE") == [SYMENGINE_DEPRECATION]


@pytest.mark.parametrize("source, syntax, engine, expected", [
    (PGCL, "pgcl", ForwardAnalysisConfig.Engine.SYMPY, [PGCL_DEPRECATION]),
    (PPY, "ppy", ForwardAnalysisConfig.Engine.SYMPY, []),
    (PPY, "ppy", ForwardAnalysisConfig.Engine.SYMENGINE, [SYMENGINE_DEPRECATION]),
    (PGCL, "pgcl", ForwardAnalysisConfig.Engine.SYMENGINE, [PGCL_DEPRECATION, SYMENGINE_DEPRECATION]),
])
def test_the_result_carries_the_deprecation_warnings(source, syntax, engine, expected):
    config = ForwardAnalysisConfig(engine=engine)
    result = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config).run()
    assert result.status == "ok"
    assert [w for w in result.to_json()["warnings"] if "deprecated" in w] == expected


@pytest.mark.parametrize("name, source, engine, expected", [
    ("a.pgcl", PGCL, "sympy", ["pgcl"]),
    ("a.ppy", PPY, "sympy", []),
    ("a.ppy", PPY, "symengine", ["symengine"]),
])
def test_cli_main_warns_on_stderr(tmp_path, name, source, engine, expected):
    path = tmp_path / name
    path.write_text(source)
    result = CliRunner().invoke(cli, ["--engine", engine, "main", str(path)])
    assert result.exit_code == 0
    assert ("v0.6.0" in result.output) == bool(expected)
    assert ("The .pgcl syntax is deprecated" in result.output) == ("pgcl" in expected)
    assert ("The SymEngine engine is deprecated" in result.output) == ("symengine" in expected)


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
