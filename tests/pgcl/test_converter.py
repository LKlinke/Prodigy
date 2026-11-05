"""Tests for the converter from .pgcl to .ppy and for the .ppy support of the CLI."""
import json

import pytest
from click.testing import CliRunner

from prodigy.cli import cli
from prodigy.pgcl.converter import convert_pgcl, keyword_renamings, rename
from prodigy.pgcl.parser import parse_pgcl
from prodigy.pgcl.ppy_parser import parse_ppy


def _round_trip(source: str):
    conversion = convert_pgcl(source)
    assert parse_ppy(conversion.text) == rename(parse_pgcl(source), conversion.renamings)
    return conversion


# --- renaming -----------------------------------------------------------------


def test_renaming_avoids_keywords_and_other_names():
    program = parse_pgcl("nat switch; nat switch_; nat print; switch := print + switch_")
    assert keyword_renamings(program) == {"print": "print_", "switch": "switch__"}


def test_renaming_does_not_depend_on_declaration_order():
    first = keyword_renamings(parse_pgcl("nat case; nat default; case := default"))
    second = keyword_renamings(parse_pgcl("nat default; nat case; case := default"))
    assert first == second == {"case": "case_", "default": "default_"}


def test_renaming_functions_parameters_and_named_arguments():
    source = """
fun def := {
    nat case;
    return case + 1;
}
nat param;
param := def(case := 2);
?Pr[param = 3]
"""
    conversion = _round_trip(source)
    assert conversion.renamings == {"case": "case_", "def": "def_", "param": "param_"}
    assert "def def_(case_: nat) -> nat {" in conversion.text
    assert "param_ = def_(case_ = 2);" in conversion.text
    assert len(conversion.messages()) == 3


def test_parameters_constants_and_queries_are_renamed():
    source = "nat x; rparam sample; const plot := 2; x := plot; ?Opt[x, sample, MAX]"
    conversion = _round_trip(source)
    assert conversion.renamings == {"plot": "plot_", "sample": "sample_"}
    assert "optimize(x, sample_, MAX);" in conversion.text


def test_rename_leaves_the_original_unchanged():
    program = parse_pgcl("nat switch; switch := switch + 1")
    renamed = rename(program, {"switch": "switch_"})
    assert program.variables == {"switch": program.variables["switch"]}
    assert list(renamed.variables) == ["switch_"]


# --- comments -----------------------------------------------------------------


def test_comments_are_kept():
    source = """# header
nat x; // trailing on a declaration

// before the loop
while (x < 3) { // after the opening brace
    x := x + 1 # trailing
}
// at the end
"""
    conversion = _round_trip(source)
    assert conversion.comments == 6
    assert conversion.text == """# header
x: nat;  # trailing on a declaration

# before the loop
while (x < 3) {
    # after the opening brace
    x = x + 1;  # trailing
}
# at the end
"""


def test_comment_before_else_if_keeps_the_ast():
    source = """nat x;
if (x = 0) { x := 1 } else {
    // second case
    if (x = 1) { x := 2 } else { skip }
}
"""
    conversion = _round_trip(source)
    assert "# second case" in conversion.text


def test_comments_in_functions_and_query_blocks():
    source = """fun f := {
    nat a; // the argument
    // body
    a := a + 1
    return a; // result
}
nat x;
query {
    // inside
    x := f(1)
}
?Ex[x] // the expectation
"""
    conversion = _round_trip(source)
    for text in ("# the argument", "# body", "# result", "# inside", "query Ex[x];  # the expectation"):
        assert text in conversion.text


# --- command line -------------------------------------------------------------

GEOMETRIC_PPY = """c: nat;
x: nat = 1;
while (x == 1) {
    {x = 0;} [1/2] {c = c + 1;}
}
query Pr[c == 2];
"""


def test_cli_convert(tmp_path):
    program = tmp_path / "monty.pgcl"
    program.write_text("nat switch; // the door\nswitch := unif(1, 3)\n?Pr[switch = 1]\n")
    output = tmp_path / "monty.ppy"
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", str(program), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert "renamed 'switch' to 'switch_'" in result.output
    assert output.read_text() == "switch_: nat;  # the door\nswitch_ = unif(1, 3);\nquery Pr[switch_ == 1];\n"

    printed = runner.invoke(cli, ["convert", str(program)])
    assert printed.exit_code == 0
    assert "query Pr[switch_ == 1];" in printed.output

    again = runner.invoke(cli, ["convert", str(output)])
    assert again.exit_code != 0 and "already a .ppy program" in again.output


def test_cli_convert_reports_syntax_errors(tmp_path):
    program = tmp_path / "broken.pgcl"
    program.write_text("nat x;\nx := := 1\n")
    result = CliRunner().invoke(cli, ["convert", str(program)])
    assert result.exit_code == 1
    assert "line 2, column 6" in result.output


def test_cli_analyze_ppy(tmp_path):
    program = tmp_path / "geometric.ppy"
    program.write_text(GEOMETRIC_PPY)
    result = CliRunner().invoke(cli, ["analyze", str(program), "--loop", "0=unroll:iterations=20", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["loops"][0]["line"] == 3
    assert data["queries"][0]["line"] == 6


def test_cli_syntax_option_overrides_the_extension(tmp_path):
    program = tmp_path / "geometric.txt"
    program.write_text(GEOMETRIC_PPY)
    runner = CliRunner()
    assert runner.invoke(cli, ["outline", str(program)]).exit_code == 1
    result = runner.invoke(cli, ["outline", "--syntax", "ppy", str(program)])
    assert result.exit_code == 0, result.output
    assert "loop 0 (line 3, depth 0): while" in result.output
    assert "query 0 (line 6): query Pr[c == 2];" in result.output


def test_cli_ppy_syntax_error_has_line_and_column(tmp_path):
    program = tmp_path / "broken.ppy"
    program.write_text("x: nat;\nx := 3;\n")
    result = CliRunner().invoke(cli, ["analyze", str(program)])
    assert result.exit_code == 1
    assert "broken.ppy: line 2, column 4:" in result.output
    assert "assignments in .ppy are written 'x = e'" in result.output
    assert "    x := 3;\n       ^" in result.output


@pytest.mark.parametrize("suffix", [".ppy", ".pgcl"])
def test_cli_check_equality_reads_each_file_in_its_syntax(tmp_path, suffix):
    program = tmp_path / "program.ppy"
    program.write_text("x: nat;\nx = unif(1, 3);\n")
    other = tmp_path / f"other{suffix}"
    other.write_text("x: nat;\n{x = 1;} [1/3] {{x = 2;} [1/2] {x = 3;}}\n" if suffix == ".ppy"
                     else "nat x;\n{x := 1} [1/3] {{x := 2} [1/2] {x := 3}}\n")
    result = CliRunner().invoke(cli, ["check_equality", str(program), str(other)])
    assert result.exit_code == 0, result.output
    assert "is equivalent" in result.output


def test_cli_invariant_uses_the_program_syntax(tmp_path):
    program = tmp_path / "geometric.ppy"
    program.write_text("x, c, temp: nat;\nwhile (x == 1) {\n    {x = 0;} [1/2] {c = c + 1;}\n    temp = 0;\n}\n")
    invariant = tmp_path / "invariant.ppy"
    invariant.write_text("x, c, temp: nat;\n"
                         "if (x == 1) {\n    temp = geometric(1/2);\n    c = c + temp;\n    x = 0;\n    temp = 0;\n}\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(program), "--json",
                                 "--loop", f"0=invariant:invariant_file={invariant}"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["loops"][0]["strategy"] == "invariant"
    assert data["bounds"]["kind"] in ("exact", "upper")

    pgcl_invariant = tmp_path / "invariant.pgcl"
    pgcl_invariant.write_text("nat x;\nnat c;\nnat temp;\n"
                              "if (x = 1) { temp := geometric(1/2); c := c + temp; x := 0; temp := 0 } else { skip }\n")
    wrong = runner.invoke(cli, ["analyze", str(program), "--json",
                                "--loop", f"0=invariant:invariant_file={pgcl_invariant}"])
    assert wrong.exit_code == 1
    assert "an invariant uses the syntax of the program" in wrong.output


def test_loop_lines_with_a_function_after_a_loop():
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import LoopRegistry
    source = """x: nat;
while (x > 0) { x = x - 1; }  # while in a comment
def f(a: nat) -> nat {
    while (a > 0) { a = a - 1; }
    return a;
}
x = f(2);
"""
    loops = list(LoopRegistry(parse_ppy(source), source, "ppy"))
    assert [(loop.function, loop.line) for loop in loops] == [("f", 4), (None, 2)]
