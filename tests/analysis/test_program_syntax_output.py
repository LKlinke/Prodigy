"""Results speak the syntax of the program: queries, loop guards, certificates and steps are
written in .ppy for a .ppy program and in pGCL for a pGCL program (issue #106)."""
import json

from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl.ppy_printer import spell
from prodigy.cli import EXIT_UNDECIDED, cli
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.ppy_parser import parse_ppy_expr

PPY = """x: nat;
c: nat;
x = 1;
while (x == 1 and c < 5) {
    {x = 0;} [1/2] {c = c + 1;}
}
if (c == 0) { x = 1; }
query Pr[x == 0 and not c == 1];
query Ex[c];
query Pr[c];
"""

PGCL = """x: nat;
c: nat;
x = 1;
while (x == 1 and c < 5) {
    {
        x = 0;
    } [1/2] {
        c = c + 1;
    }
}
if (c == 0) {
    x = 1;
} else {
    skip;
}
query Pr[x == 0 and not c == 1];
query Ex[c];
query Pr[c];
"""


def _analyse(source, syntax, decision="unroll:iterations=5"):
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=ForwardAnalysisConfig(),
                              decisions={"0": decision}, record_steps=True, capture_output=True)
    return session.run().to_json()


def test_ppy_result_uses_ppy_spelling():
    result = _analyse(PPY, "ppy")
    assert result["status"] == "ok", result["error"]
    assert result["syntax"] == "ppy"
    assert [q["expr"] for q in result["queries"]] == ["x == 0 and not c == 1", "c", "c"]
    assert [q["text"] for q in result["program"]["queries"]] == ["query Pr[x == 0 and not c == 1];",
                                                                  "query Ex[c];", "query Pr[c];"]
    assert result["program"]["loops"][0]["cond"] == "x == 1 and c < 5"
    assert result["loops"][0]["cond"] == "x == 1 and c < 5"
    steps = "\n".join(step["instruction"] for step in result["steps"])
    assert "if (c == 0) {" in steps and ":=" not in steps


def test_both_syntaxes_give_the_same_values():
    ppy, pgcl = _analyse(PPY, "ppy"), _analyse(PGCL, "ppy")
    assert [q["value"] for q in ppy["queries"]] == [q["value"] for q in pgcl["queries"]]
    assert ppy["distribution"]["closed_form"] == pgcl["distribution"]["closed_form"]


def test_certificate_equation_uses_the_program_syntax():
    source = "x: nat;\nc: nat;\nx = 1;\nwhile (x == 1) {\n    {x = 0;} [1/2] {c = c + 1;}\n}\nquery Ex[c];\n"
    result = _analyse(source, "ppy", "evt_synthesis")
    assert result["status"] == "ok", result["error"]
    assert "⟨I⟩_(x == 1)" in result["loops"][0]["certificates"][0]["equation"]


def test_cli_summary_uses_the_program_syntax(tmp_path):
    program = tmp_path / "program.ppy"
    program.write_text(PPY)
    output = CliRunner().invoke(cli, ["analyze", str(program), "--loop", "0=unroll:iterations=5"]).output
    assert "Pr[x == 0 and not c == 1] =" in output
    assert "marginal of c:" in output


def test_cli_main_prints_queries_in_the_program_syntax(tmp_path):
    program = tmp_path / "program.ppy"
    program.write_text(PPY)
    result = CliRunner().invoke(cli, ["main", str(program), "--loop", "0=unroll:iterations=5"])
    assert result.exit_code == 0, result.output
    assert "Probability of x == 0 and not c == 1:" in result.output


def test_cli_check_equality_reads_invariant_files_in_the_program_syntax(tmp_path):
    program = tmp_path / "geometric.ppy"
    program.write_text("x, c, temp: nat;\nwhile (x == 1) {\n    {x = 0;} [1/2] {c = c + 1;}\n    temp = 0;\n}\n")
    invariant = tmp_path / "invariant.ppy"
    invariant.write_text("x, c, temp: nat;\n"
                         "if (x == 1) {\n    temp = geometric(1/2);\n    c = c + temp;\n    x = 0;\n    temp = 0;\n}\n")
    # the loop asks for its strategy (1: invariant) and the invariant file
    result = CliRunner().invoke(cli, ["check_equality", str(program), str(invariant)], input=f"1\n{invariant}\n")
    # the invariant (Park's lemma) gives an upper bound: one direction is proven (#395)
    assert result.exit_code == EXIT_UNDECIDED, result.output
    assert "one direction proven: first_below_second" in result.output


def test_result_json_states_the_syntax(tmp_path):
    program = tmp_path / "program.ppy"
    program.write_text(PPY)
    data = json.loads(CliRunner().invoke(cli, ["analyze", str(program), "--json", "--loop",
                                               "0=unroll:iterations=5"]).output)
    assert data["syntax"] == "ppy" and data["queries"][0]["expr"] == "x == 0 and not c == 1"
