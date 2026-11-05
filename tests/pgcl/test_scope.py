"""Scope rules and lowering of declarations inside blocks (`.ppy`, #106)."""
import pytest

from prodigy.pgcl.ast import AsgnInstr, NatLitExpr
from prodigy.pgcl.check import CheckFail, check_program
from prodigy.pgcl.parser import parse_pgcl
from prodigy.pgcl.ppy_parser import PpySyntaxError, parse_ppy
from prodigy.pgcl.scope import ScopeError, lower_block_scope


@pytest.mark.parametrize("source, message", [
    ("n: nat;\nif (n > 0) {\n  c: nat = 1;\n}\nn = c;",
     "line 3, column 3: 'c' is used outside the block that declares it"),
    ("n: nat;\nif (n > 0) { n = c; c: nat; }",
     "line 2, column 21: 'c' is used outside the block that declares it (or before its declaration)"),
    ("n: nat;\nif (n > 0) { c: nat = c + 1; }",
     "line 2, column 14: 'c' is used outside the block"),
    ("n: nat;\nif (n > 0) { c: nat = 1; }\nquery Pr[c == 1];",
     "line 2, column 14: 'c' is used outside the block"),
    ("n: nat;\nif (n > 0) { n: nat = 1; }",
     "line 2, column 14: 'n' is already declared; a declaration inside a block cannot hide another name"),
    ("const N: nat = 2;\nx: nat;\nif (x > 0) { N: nat; }", "'N' is already declared"),
    ("param p: real;\nx: nat;\nif (x > 0) { p: nat; }", "'p' is already declared"),
    ("n: nat;\nwhile (n > 0) { c: nat; if (c == 0) { c: nat; } n = n - 1; }",
     "line 2, column 39: 'c' is already declared"),
    ("n: nat;\nif (n > 0) { c: nat; c: nat; }", "line 2, column 22: 'c' is already declared"),
    ("n: nat;\nif (n > 0) { c: nat; } else { c: bool; }",
     "line 2, column 31: 'c' is declared in several blocks with different types"),
    ("n: nat;\nif (n > 0) { r: real; }", "Only nat and bool variables can be declared inside a block"),
    ("def f(a: nat) -> nat { if (a > 0) { a: nat = 1; } return a; }\nx: nat;",
     "line 1, column 37: 'a' is already declared"),
    ("def f(a: nat) -> nat { if (a > 0) { t: nat = 1; } return t; }\nx: nat;",
     "'t' is used outside the block"),
])
def test_scope_errors(source, message):
    with pytest.raises(PpySyntaxError) as info:
        parse_ppy(source)
    assert message in str(info.value)


def test_names_of_sibling_blocks_share_a_variable():
    program = parse_pgcl("n: nat;\nif (n > 0) { c: nat = 1; n = c; } else { c: nat = 2; n = c; }", "ppy")
    assert program.variables == {"n": parse_ppy("n: nat;").variables["n"], "c": program.variables["c"]}
    assert program.block_variables == {"c"}
    assert program.instructions[0].true[-1] == AsgnInstr("c", NatLitExpr(0))
    assert check_program(program) is None


def test_lowering_leaves_programs_without_block_declarations_alone():
    program = parse_ppy("x: nat; while (x < 3) { x = x + 1; }")
    assert lower_block_scope(program) is program
    assert program.block_variables == set()


def test_lowering_does_not_change_its_input():
    raw = parse_ppy("x: nat; while (x < 3) { y: nat = 1; x = x + y; }")
    before = str(raw)
    lowered = lower_block_scope(raw)
    assert str(raw) == before and "y" not in raw.variables
    assert "y" in lowered.variables


def test_raw_block_declarations_are_rejected_by_the_type_checker():
    result = check_program(parse_ppy("x: nat; if (x == 0) { y: nat = 1; x = y; }"))
    assert isinstance(result, CheckFail)
    assert "has not been lowered" in result.message


def test_scope_error_without_parser():
    raw = parse_ppy("x: nat; if (x == 0) { y: nat = 1; }")
    raw.instructions.append(AsgnInstr("x", parse_ppy("y: nat; x: nat; x = y;").instructions[0].rhs))
    with pytest.raises(ScopeError, match="'y' is used outside the block") as info:
        lower_block_scope(raw)
    assert info.value.declaration is raw.instructions[0].true[0]
