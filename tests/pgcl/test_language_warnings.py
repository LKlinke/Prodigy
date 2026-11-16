"""Language warnings (#377 monus, #335 implicit cast): the check, the CLI and the GUI's endpoint."""
import pytest
from click.testing import CliRunner

from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.language_warnings import language_warnings
from prodigy.web.app import create_app

DECLS = "nat x; nat y; nat z;\n"
PPY_DECLS = "x: nat;\ny: nat;\nz: nat;\n"


def kinds(source, syntax="pgcl"):
    return [(w.kind, w.message) for w in language_warnings(parse_pgcl(source, syntax), syntax)]


@pytest.mark.parametrize("expr", ["x - y + z", "x - y + 1", "x - (y - z)", "x + 1 - (y - z)"])
def test_monus_warns(expr):
    found = kinds(DECLS + f"x := {expr}")
    assert [kind for kind, _ in found] == ["monus"]


@pytest.mark.parametrize("expr", ["x - y - z", "(x - y) * z", "x + z - y", "5 - 3 + z", "x - (y + z)", "z + (x - y)"])
def test_monus_does_not_warn(expr):
    assert kinds(DECLS + f"x := {expr}") == []


def test_monus_skips_constants_and_reals():
    assert kinds("const N := 5;\nnat z;\nz := N - 2 + z") == []
    assert kinds("real r; nat x;\nr := r - 1/2 + x") == []


def test_monus_message_in_the_programs_syntax():
    assert kinds(PPY_DECLS + "x = x - y + z;", "ppy") == [
        ("monus", "`x - y + z`: `-` is truncated at 0 before `+` is applied; write `x + z - y` for "
                  "integer arithmetic.")]
    [(_, message)] = kinds(PPY_DECLS + "x = x - (y - z);", "ppy")
    assert "`x - (y - z)`" in message and "`x + z - y`" in message


def test_monus_in_guards_and_functions():
    assert len(kinds(DECLS + "while (x - y + z > 1) { x := x - 1 }")) == 1
    function = "def f(n: nat) -> nat {\n    m: nat;\n    m = n - 1 + 2;\n    return m;\n}\n"
    assert [kind for kind, _ in kinds(function + PPY_DECLS + "x = f(y);", "ppy")] == ["monus"]


def test_implicit_cast_bool_to_nat():
    assert kinds(DECLS + "x := x < 2") == [("cast", "implicit cast from bool to nat (true → 1, false → 0)")]
    assert [kind for kind, _ in kinds(DECLS + "x := true")] == ["cast"]
    assert kinds("bool b; nat x;\nx := b")[0][0] == "cast"


def test_no_cast_warning_otherwise():
    assert kinds("bool b; nat x;\nb := x < 2") == []
    assert kinds(DECLS + "x := y") == []
    assert kinds("bool b;\nb := true") == []


@pytest.fixture(name="client")
def client_fixture():
    return create_app().test_client()


HOST = {"Host": "localhost:8080"}


def diagnostics(client, source, syntax="ppy"):
    body = {"source": source, "syntax": syntax}
    return client.post("/api/check", json=body, headers=HOST).get_json()


def test_endpoint_delivers_warnings_with_positions(client):
    source = PPY_DECLS + "x = x - y + z;\nx = x < 2;\nquery Ex[x];\n"
    result = diagnostics(client, source)
    assert result["ok"]
    warnings = [d for d in result["diagnostics"] if d["severity"] == "warning"]
    assert [(d["line"], d["column"], d["end_column"]) for d in warnings] == [(4, 5, 14), (5, 5, 10)]
    assert "truncated at 0" in warnings[0]["message"]
    assert "implicit cast" in warnings[1]["message"]


def test_endpoint_marks_the_whole_expression_with_redundant_parentheses(client):
    result = diagnostics(client, PPY_DECLS + "x = (x - y) + z;\n")
    warning = next(d for d in result["diagnostics"] if d["severity"] == "warning")
    assert warning["line"] == 4  # not found as printed: the instruction is marked


def test_endpoint_no_warning_for_clean_program(client):
    result = diagnostics(client, PPY_DECLS + "x = x - y - z;\nquery Ex[x];\n")
    assert [d for d in result["diagnostics"] if d["severity"] == "warning"] == []


@pytest.mark.parametrize("flag", [True, False])
def test_cli_prints_warnings_on_stderr(tmp_path, flag):
    program = tmp_path / "w.ppy"
    program.write_text(PPY_DECLS + "x = x - y + z;\nx = x < 2;\nquery Ex[x];\n")
    runner = CliRunner()
    args = ["check_equality", str(program), str(program)] if flag else ["main", str(program)]
    result = runner.invoke(cli, args)
    assert "w.ppy:4:5:" in result.stderr and "truncated at 0" in result.stderr
    assert "w.ppy:5:5:" in result.stderr and "implicit cast from bool to nat" in result.stderr
    assert "warning" not in result.stdout
