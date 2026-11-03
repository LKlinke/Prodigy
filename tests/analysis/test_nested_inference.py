"""Programs with nested inference (dist, infer, sample) are rejected until they have semantics."""
import json

import pytest
from click.testing import CliRunner
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.compiler import compile_pgcl
from prodigy.pgcl.pgcl_checks import find_nested_inference

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import UnsupportedFeatureError
from prodigy.analysis.session import AnalysisSession
from prodigy.web.diagnostics import check_source
from prodigy.web.requests import run_analysis

NESTED = """dist d;
nat x;
fun f := {
    nat y;
    y := bernoulli(1/2);
    return y;
}
d := infer { f() };
x := sample { d };
?Pr[x = 1]
"""

MESSAGE = "is not supported by the analysis yet"


def test_program_still_compiles():
    # parser and type checker keep accepting the syntax
    assert not hasattr(compile_pgcl(NESTED), "message")


@pytest.mark.parametrize("source, found", [
    (NESTED, "dist d"),
    ("nat x; fun f := { nat y; return y; } x := infer { f() };", "infer { f() }"),
    ("nat x; nat d; x := sample { d };", "sample { d }"),
    ("nat x; fun f := { dist e; return 1; } x := f();", "dist e"),
    ("nat x; fun f := { nat y; return 1; } "
     "fun g := { nat z; z := infer { f() }; return 1; } x := g();", "infer { f() }"),
    ("nat x; x := 1; while (x = 1) { {x := sample { x }} [1/2] {x := 0} }", "sample { x }"),
])
def test_find_nested_inference(source, found):
    assert find_nested_inference(parse_pgcl(source)) == found


def test_find_nested_inference_ignores_other_programs():
    program = parse_pgcl("nat x; fun f := { nat y; y := bernoulli(1/2); return y; } "
                         "x := f(); {x := x + 1} [1/2] {skip}")
    assert find_nested_inference(program) is None


def test_analysis_rejects_before_running():
    program = parse_pgcl(NESTED)
    config = ForwardAnalysisConfig()
    with pytest.raises(UnsupportedFeatureError, match=MESSAGE):
        compute_discrete_distribution(program, config.factory.one(*program.variables.keys()), config)


def test_session_reports_an_analysis_error():
    result = AnalysisSession(parse_pgcl(NESTED), source=NESTED).run().to_json()
    assert result["status"] == "error"
    assert result["error"]["type"] == "UnsupportedFeatureError"
    assert MESSAGE in result["error"]["message"] and "dist d" in result["error"]["message"]
    assert "traceback" not in result["error"]


def test_gui_request_reports_an_analysis_error():
    result = run_analysis({"source": NESTED})
    assert result["status"] == "error"
    assert MESSAGE in result["error"]["message"]
    assert "traceback" not in result["error"]


def test_gui_check_marks_the_program():
    result = check_source(NESTED)
    assert not result["ok"]
    error = next(d for d in result["diagnostics"] if MESSAGE in d["message"])
    assert (error["severity"], error["line"], error["column"], error["end_column"]) == ("error", 1, 1, 5)
    inside = check_source("nat x;\nfun f := { nat y; return y; }\nx := infer { f() };\n")
    error = next(d for d in inside["diagnostics"] if MESSAGE in d["message"])
    assert (error["line"], error["column"]) == (3, 6)


@pytest.mark.parametrize("args", [["main"], ["analyze"], ["analyze", "--json"]])
def test_cli_shows_the_message_without_traceback(tmp_path, args):
    from prodigy.cli import cli  # pylint: disable=import-outside-toplevel
    program = tmp_path / "nested.pgcl"
    program.write_text(NESTED)
    result = CliRunner().invoke(cli, [args[0], str(program), *args[1:]])
    assert result.exit_code == 1
    assert MESSAGE in result.output
    assert "Traceback" not in result.output
    if "--json" in args:
        assert json.loads(result.output)["status"] == "error"
