"""``tick(e)`` and Iverson brackets ``[g]`` are not part of the language: programs using them are
rejected with a clear message at their position, in both syntaxes."""
import pytest
from lark.exceptions import UnexpectedInput

from prodigy.pgcl.parser import parse_pgcl
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


def test_brackets_that_stay():
    """Choice, bounds and queries still use square brackets."""
    program = parse_ppy("x: nat[0, 3];\n{ x = 1; } [1/2] { x = 2; }\nquery Pr[x == 1];")
    assert len(program.instructions) == 2
    program = parse_pgcl("""x: nat[0, 3];
{
    x = 1;
} [1/2] {
    x = 2;
}
query Pr[x == 1];""")
    assert len(program.instructions) == 2


def test_tick_is_a_name_again():
    program = parse_ppy("tick: nat;\ntick = 1;")
    assert "tick" in program.variables
    assert "tick" in parse_pgcl("""tick: nat;
tick = 1;""").variables


