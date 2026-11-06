"""``tick(e)`` and Iverson brackets ``[g]`` are not part of the language: programs using them are
rejected with a clear message at their position, in both syntaxes."""
import pytest
from lark.exceptions import UnexpectedInput

from prodigy.pgcl.parser import parse_expectation, parse_expr, parse_pgcl, parse_pgcl_with_lines
from prodigy.pgcl.ppy_parser import PpySyntaxError, parse_ppy, parse_ppy_expr
from prodigy.pgcl.syntax_errors import ProgramSyntaxError

TICK = "tick(e) is not supported."
IVERSON = "Iverson brackets '[g]' are not supported."


@pytest.mark.parametrize("source, line, column", [
    ("x: nat;\nx = 1;\ntick(x);", 3, 1),
    ("x: nat;\nwhile (x < 3) {\n    tick (1);\n    x = x + 1;\n}", 3, 5),
])
def test_ppy_tick(source, line, column):
    with pytest.raises(PpySyntaxError) as info:
        parse_ppy(source)
    assert (info.value.message, info.value.line, info.value.column) == (TICK, line, column)


@pytest.mark.parametrize("source, line, column", [
    ("x: nat;\nx = [x == 1];", 2, 5),
    ("x: nat;\nx = 2 * [x > 0] + 1;", 2, 9),
    ("x: nat;\nif ([x == 1]) { skip; }", 2, 5),
    ("x: nat;\nquery Ex[[x == 1]];", 2, 10),
])
def test_ppy_iverson(source, line, column):
    with pytest.raises(PpySyntaxError) as info:
        parse_ppy(source)
    assert (info.value.message, info.value.line, info.value.column) == (IVERSON, line, column)


def test_ppy_expression_iverson():
    with pytest.raises(PpySyntaxError, match="Iverson"):
        parse_ppy_expr("[x == 1]")


@pytest.mark.parametrize("source, line, column, message", [
    ("nat x;\nx := 1;\ntick(x);", 3, 1, TICK),
    ("nat x;\nx := [x = 1];", 2, 6, IVERSON),
    ("nat x;\n?Ex[[x = 1]];", 2, 5, IVERSON),
])
def test_pgcl(source, line, column, message):
    for parse in (parse_pgcl, parse_pgcl_with_lines):
        with pytest.raises(ProgramSyntaxError) as info:
            parse(source)
        assert (info.value.message, info.value.line, info.value.column) == (message, line, column)


def test_pgcl_expressions():
    for parse in (parse_expr, parse_expectation):
        with pytest.raises(ProgramSyntaxError, match="Iverson"):
            parse("[x = 1]")


def test_brackets_that_stay():
    """Choice, bounds and queries still use square brackets."""
    program = parse_ppy("x: nat[0, 3];\n{ x = 1; } [1/2] { x = 2; }\nquery Pr[x == 1];")
    assert len(program.instructions) == 2
    program = parse_pgcl("nat x [0, 3];\n{ x := 1 } [1/2] { x := 2 }\n?Pr[x = 1];")
    assert len(program.instructions) == 2


def test_tick_is_a_name_again():
    program = parse_ppy("tick: nat;\ntick = 1;")
    assert "tick" in program.variables
    assert "tick" in parse_pgcl("nat tick;\ntick := 1;").variables


def test_other_errors_are_unchanged():
    with pytest.raises(PpySyntaxError, match="Unexpected '\\['"):
        parse_ppy("x: nat;\nx = x[1];")
    with pytest.raises(UnexpectedInput):
        parse_pgcl("nat x;\nx := x[1];")
