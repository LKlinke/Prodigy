"""Tests for the .ppy syntax: parser, pretty printer and agreement with the old parser."""
from fractions import Fraction
from pathlib import Path

import pytest

from prodigy.pgcl.ast import (AsgnInstr, Binop, BinopExpr, CategoricalExpr,
                              FunctionCallExpr, IfInstr, LocalDeclInstr,
                              NatLitExpr, NatType, RealLitExpr, VarDecl,
                              VarExpr)
from prodigy.pgcl.check import CheckFail
from prodigy.pgcl.compiler import compile_pgcl
from prodigy.pgcl.converter import convert_pgcl, rename
from prodigy.pgcl.parser import parse_expectation, parse_pgcl, select_syntax
from prodigy.pgcl.ppy_parser import PpySyntaxError, parse_ppy, parse_ppy_expr
from prodigy.pgcl.ppy_printer import (PpyPrintError, expr_to_ppy,
                                      program_to_ppy)

EXAMPLES = Path(__file__).parents[2] / "pgfexamples"
ALL_EXAMPLES = sorted(EXAMPLES.rglob("*.ppy"))
# the former .pgcl version of an example whose variable 'switch' is a keyword in .ppy
MONTY_HALL = Path(__file__).parent / "former_examples/inference/loop_free/conditioning/monty_hall_nested.pgcl"


def _same(new, old):
    """Equal ASTs, and the same literal representations (Decimal vs. Fraction)."""
    assert new == old
    assert repr(new) == repr(old)
    assert repr(getattr(new, "declarations", None)) == repr(getattr(old, "declarations", None))


def _agree(ppy: str, pgcl: str):
    _same(parse_ppy(ppy), parse_pgcl(pgcl))


# --- all examples -------------------------------------------------------------


def test_examples_exist():
    assert len(ALL_EXAMPLES) > 100
    assert not list(EXAMPLES.rglob("*.pgcl")), "the examples are .ppy programs"


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_round_trip_example(path: Path):
    """Every example parses, and printing it gives back the same AST."""
    program = parse_ppy(path.read_text(encoding="utf-8"))
    _same(parse_ppy(program_to_ppy(program)), program)


def test_round_trip_after_renaming_keyword():
    old = parse_pgcl(MONTY_HALL.read_text())
    with pytest.raises(PpyPrintError):
        program_to_ppy(old)
    conversion = convert_pgcl(MONTY_HALL.read_text())
    assert conversion.renamings == {"switch": "switch_"}
    assert "switch_ = unif(1, 3);" in conversion.text
    assert conversion.messages() == ["renamed 'switch' to 'switch_' ('switch' is a keyword in .ppy)"]
    _same(parse_ppy(conversion.text), rename(old, conversion.renamings))


# --- declarations -------------------------------------------------------------


def test_declarations():
    _agree("x: nat; b: bool; r: real; d: dist;", "nat x; bool b; real r; dist d;")
    _agree("x, y: nat;", "nat x; nat y;")
    _agree("x: nat[0, 10];", "nat x [0, 10];")
    _agree("x, y: nat[1, 5];", "nat x [1, 5]; nat y [1, 5];")
    _agree("param p: real; param n: nat;", "rparam p; nparam n;")
    _agree("const N: nat = 10; const q: real = 1/2; const b: bool = true; const M = N + 1;",
           "const N := 10; const q := 1/2; const b := true; const M := N + 1;")


def test_bounds_are_not_shared():
    program = parse_ppy("x, y: nat[0, 3];")
    assert program.variables["x"] == program.variables["y"]
    assert program.variables["x"] is not program.variables["y"]


def test_initial_value():
    _agree("x: nat = 3; b: bool = true;", "nat x; bool b; x := 3; b := true;")
    _agree("x: nat = unif(1, 6);", "nat x; x := unif(1, 6);")


def test_declaration_after_statements_starts_at_zero():
    # at the start, a declaration keeps the old meaning (input variable);
    # later it gives 0 explicitly
    _agree("x: nat; x = 2; y: nat; b: bool; z: nat = x;",
           "nat x; nat y; bool b; nat z; x := 2; y := 0; b := false; z := x;")


def test_declarations_inside_blocks():
    program = parse_ppy("x: nat; while (x < 3) { y: nat = 1; z: bool; x = x + y; }")
    body = program.instructions[0].body
    assert body[:2] == [
        LocalDeclInstr(VarDecl("y", NatType(None)), NatLitExpr(1)),
        LocalDeclInstr(VarDecl("z", parse_ppy("z: bool;").variables["z"]), None),
    ]
    assert "y" not in program.variables


@pytest.mark.parametrize("source, expected", [
    ("x: nat; while (x < 3) { y: nat = 1; x = x + y; }",
     "nat x; nat y; while (x < 3) { y := 1; x := x + y; y := 0 }"),
    ("x: nat; if (x == 0) { y: nat; } else { skip; }",
     "nat x; nat y; if (x = 0) { y := 0 } else { skip }"),
    ("x: nat; if (x == 0) { b: bool = x == 0; if (b) { x = 2; } }",
     "nat x; bool b; if (x = 0) { b := x = 0; if (b) { x := 2 } else { } b := false } else { }"),
    ("def f(a: nat) -> nat { while (a > 0) { t: nat = a; a = a - 1; } return a; } x: nat; x = f(2);",
     "fun f := { nat a; nat t; while (a > 0) { t := a; a := a - 1; t := 0 } return a; }; nat x; x := f(2)"),
])
def test_block_declarations_are_lowered(source, expected):
    result = compile_pgcl(source, syntax="ppy")
    assert not isinstance(result, CheckFail), result
    assert result == compile_pgcl(expected)


def test_const_type_mismatch():
    with pytest.raises(PpySyntaxError, match="does not have the declared type"):
        parse_ppy("const N: nat = 0.5;")
    with pytest.raises(PpySyntaxError, match="cannot have bounds"):
        parse_ppy("const N: nat[0, 3] = 1;")


def test_parameter_types():
    with pytest.raises(PpySyntaxError, match="must be of type real or nat"):
        parse_ppy("param p: bool;")
    with pytest.raises(PpySyntaxError, match="cannot have bounds"):
        parse_ppy("param n: nat[0, 3];")


def test_use_before_declaration():
    with pytest.raises(PpySyntaxError, match=r"line 1, column 8: 'y' is used before its declaration"):
        parse_ppy("x = y; y: nat;")
    with pytest.raises(PpySyntaxError, match="used before its declaration"):
        parse_ppy("x: nat = x;")
    with pytest.raises(PpySyntaxError, match="used before its declaration"):
        parse_ppy("x: nat; x = N; const N: nat = 1;")


def test_initial_value_needs_single_variable():
    with pytest.raises(PpySyntaxError, match="single variable"):
        parse_ppy("x, y: nat = 1;")


# --- statements ---------------------------------------------------------------


def test_simple_statements():
    _agree("x: nat; skip; abort; x = 1; observe(x > 0);",
           "nat x; skip; abort; x := 1; observe(x > 0);")


def test_empty_statements_are_ignored():
    _agree("x: nat;; x = 1;;", "nat x; x := 1;")


def test_while_and_loop():
    _agree("x: nat; while (x < 10) { x = x + 1; }", "nat x; while (x < 10) { x := x + 1 }")
    _agree("x: nat; loop(3) { x = x + 1; }", "nat x; loop(3) { x := x + 1 }")


def test_loop_with_constant():
    _agree("const N: nat = 4; x: nat; loop(N) { x = x + 1; }",
           "const N := 4; nat x; loop(4) { x := x + 1 }")
    with pytest.raises(PpySyntaxError, match="loop\\(M\\) needs a constant"):
        parse_ppy("x: nat; loop(M) { skip; }")


def test_if_with_optional_else():
    _agree("x: nat; if (x == 1) { x = 2; }", "nat x; if (x = 1) { x := 2 } else { }")
    _agree("x: nat; if (x == 1) { x = 2; } else { x = 3; }",
           "nat x; if (x = 1) { x := 2 } else { x := 3 }")


def test_else_if():
    _agree("x: nat; if (x == 1) { x = 2; } else if (x == 2) { x = 3; } else { skip; }",
           "nat x; if (x = 1) { x := 2 } else { if (x = 2) { x := 3 } else { skip } }")


def test_binary_choice():
    _agree("x: nat; {x = 1;} [1/3] {x = 2;}", "nat x; {x := 1} [1/3] {x := 2}")
    _agree("x: nat; param p: real; {x = 1;} [p] { }", "nat x; rparam p; {x := 1} [p] {}")


def test_switch():
    ppy = """
        c, x: nat;
        c = categorical({1: 1/6, 2: 1/3, 3: 1/2});
        switch (c) {
            case 1: { x = 0; }
            case 2, 3: { x = 1; }
            default: { x = 2; }
        }
    """
    pgcl = """
        nat c; nat x;
        c := 1 : 1/6 + 2 : 1/3 + 3 : 1/2;
        if (c = 1) { x := 0 } else { if (c = 2 || c = 3) { x := 1 } else { x := 2 } }
    """
    _agree(ppy, pgcl)


def test_switch_without_default():
    _agree("c, x: nat; switch (c) { case 1: { x = 0; } case 2: { x = 1; } }",
           "nat c; nat x; if (c = 1) { x := 0 } else { if (c = 2) { x := 1 } else { } }")


def test_query_block():
    _agree("x: nat; query { x = 1; observe(x == 1); }",
           "nat x; query { x := 1; observe(x = 1) }")


# --- expressions --------------------------------------------------------------


@pytest.mark.parametrize("ppy, pgcl", [
    ("x and y", "x & y"),
    ("x && y", "x & y"),
    ("x or y", "x || y"),
    ("x || y", "x || y"),
    ("not x", "not x"),
    ("!x", "not x"),
    ("x == 1", "x = 1"),
    ("x ** 2", "x^2"),
    ("x ** y ** 2", "x^(y^2)"),
    ("-x", "-x"),
    ("-x ** 2", "-(x^2)"),
    ("2 ** -x", "2^(-x)"),
    ("-x + 1", "(-x) + 1"),
    ("x - y - 1", "(x - y) - 1"),
    ("x - (y - 1)", "x - (y - 1)"),
    ("x * y % 3", "(x * y) % 3"),
    ("1/2", "1/2"),
    ("1/2 * x", "1/2 * x"),
    ("x / 2", "x / 2"),
    ("0.5", "0.5"),
    ("∞", "∞"),
    ("\\infty", "∞"),
    ("not x < 2 and y", "(not (x < 2)) & y"),
    ("x < 2 or y >= 3 and z", "(x < 2) || ((y >= 3) & z)"),
    ("x <= 1 and x > 0", "x <= 1 & x > 0"),
    ("unif(1, 6)", "unif(1, 6)"),
    ("iid(geometric(1/2), x)", "iid(geometric(1/2), x)"),
])
def test_expressions(ppy, pgcl):
    _same(parse_ppy_expr(ppy), parse_expectation(pgcl))


def test_floor_division_is_parsed_and_checked():
    assert parse_ppy_expr("x // 2") == BinopExpr(Binop.FLOOR_DIVIDE, VarExpr("x"), NatLitExpr(2))
    assert parse_ppy_expr("x // 2 * 3").lhs.operator == Binop.FLOOR_DIVIDE
    for source in ["x: nat; x = 7 // 2;", "x: nat; while (x // 2 < 3) { x = x + 1; }",
                   "x: nat; query Ex[x // 2];", "const Q: nat = 3; x, y: nat; y = (x + 1) // Q;",
                   "x: nat; b: bool; x = b // 2;"]:
        assert not isinstance(compile_pgcl(source, syntax="ppy"), CheckFail), source


@pytest.mark.parametrize("source, message", [
    ("x, y: nat; x = x // y;", "only implemented for a positive natural constant q"),
    ("x: nat; x = x // 0;", "only implemented for a positive natural constant q"),
    ("x: nat; x = x // 1.5;", "only implemented for a positive natural constant q"),
    ("x: nat; param p: real; x = p // 2;", "only implemented for a natural number e"),
    ("x: nat[0, 10 // 2];", "not implemented in bounds"),
])
def test_floor_division_rejected(source, message):
    result = compile_pgcl(source, syntax="ppy")
    assert isinstance(result, CheckFail), source
    assert message in result.message


def test_comparisons_do_not_chain():
    with pytest.raises(PpySyntaxError, match="line 1, column 7: Unexpected '<'"):
        parse_ppy_expr("x < y < z")


def test_categorical():
    expected = CategoricalExpr([(NatLitExpr(1), RealLitExpr(Fraction(1, 6))),
                                (NatLitExpr(2), RealLitExpr(Fraction(1, 3))),
                                (NatLitExpr(5), RealLitExpr(Fraction(1, 2)))])
    assert parse_ppy("x: nat; x = categorical({1: 1/6, 2: 1/3, 5: 1/2});").instructions[0].rhs == expected
    _agree("x: nat; x = categorical({1: 1/6, 2: 1/3, 5: 1/2});", "nat x; x := 1 : 1/6 + 2 : 1/3 + 5 : 1/2")
    _agree("x: nat; x = categorical(0.2, 0.3, 0.5);", "nat x; x := 0 : 0.2 + 1 : 0.3 + 2 : 0.5")
    _agree("x, y: nat; x = categorical({y: 1/2, y + 1: 1/2});", "nat x; nat y; x := y : 1/2 + (y + 1) : 1/2")


def test_categorical_errors():
    with pytest.raises(PpySyntaxError, match="line 1, column 13: The probabilities of categorical.* sum to 5/6"):
        parse_ppy("x: nat; x = categorical({0: 1/2, 1: 1/3});")
    with pytest.raises(PpySyntaxError, match="must be a literal"):
        parse_ppy("x: nat; param p: real; x = categorical({0: p, 1: 1/2});")
    with pytest.raises(PpySyntaxError, match="only allowed as the whole right-hand side"):
        parse_ppy("x: nat; x = 1 + categorical(1/2, 1/2);")
    with pytest.raises(PpySyntaxError, match="only allowed as the whole right-hand side"):
        parse_ppy("x: nat; while (categorical(1/2, 1/2) == 1) { skip; }")


# --- functions and nested inference -------------------------------------------


def test_function_definition_and_calls():
    ppy = """
        def f(a: nat, b: nat) -> nat {
            x = a;
            return a + b;
        }
        y: nat;
        y = f(2, 3);
        y = f(2, b = 3);
        y = f(a = 2, b = 3);
        y = f();
    """
    pgcl = """
        fun f := { nat a; nat b; x := a; return a + b };
        nat y;
        y := f(2, 3);
        y := f(2, b := 3);
        y := f(a := 2, b := 3);
        y := f();
    """
    _agree(ppy, pgcl)


def test_function_with_local_variable():
    program = parse_ppy("def f(a: nat) -> nat { s: nat = a + 1; t: nat; return s + t; }")
    function = program.functions["f"]
    assert [decl.var for decl in function.declarations] == ["a", "s", "t"]
    assert function.instructions == [
        AsgnInstr("s", BinopExpr(Binop.PLUS, VarExpr("a"), NatLitExpr(1))),
        AsgnInstr("t", NatLitExpr(0)),
    ]


def test_function_return_type_is_optional():
    _agree("def f(a: nat) { return a; }", "fun f := { nat a; return a };")


def test_function_errors():
    with pytest.raises(PpySyntaxError, match="takes 1 argument"):
        parse_ppy("def f(a: nat) -> nat { s: nat; return a; } y: nat; y = f(1, 2);")
    with pytest.raises(PpySyntaxError, match="has no parameter 's'"):
        parse_ppy("def f(a: nat) -> nat { s: nat; return a; } y: nat; y = f(s = 1);")
    with pytest.raises(PpySyntaxError, match="given twice"):
        parse_ppy("def f(a: nat) -> nat { return a; } y: nat; y = f(1, a = 1);")
    with pytest.raises(PpySyntaxError, match="Positional arguments must come before"):
        parse_ppy("y: nat; y = unif(start = 1, 2);")
    with pytest.raises(PpySyntaxError, match="only return nat"):
        parse_ppy("def f(a: nat) -> bool { return a; }")
    with pytest.raises(PpySyntaxError, match="built-in distribution"):
        parse_ppy("def unif(a: nat) -> nat { return a; }")
    with pytest.raises(PpySyntaxError, match="declared twice"):
        parse_ppy("def f(a: nat) -> nat { a: nat; return a; }")


def test_infer_and_sample():
    ppy = """
        def f(a: nat) -> nat { return a + 1; }
        d: dist;
        x: nat;
        d = infer(f, 10);
        x = sample(d);
        d = infer(f, a = 2);
        d = infer(f);
    """
    pgcl = """
        fun f := { nat a; return a + 1 };
        dist d;
        nat x;
        d := infer { f(10) };
        x := sample { d };
        d := infer { f(a := 2) };
        d := infer { f() };
    """
    _agree(ppy, pgcl)
    _agree("def f(a: nat) -> nat { return a; } d: dist = infer(f, 1);",
           "fun f := { nat a; return a }; dist d; d := infer { f(1) }")


# --- queries ------------------------------------------------------------------


def test_queries():
    ppy = """
        x, y: nat;
        param p: real;
        query Pr[x == 3];
        query Ex[x + y];
        print;
        plot(x);
        plot(x, y);
        plot(x, 0.9);
        plot(x, 10);
        plot(x, ∞);
        plot(x, y, 0.5);
        optimize(x, p, MAX);
        optimize(x, p, MIN);
    """
    pgcl = """
        nat x; nat y; rparam p;
        ?Pr[x = 3]
        ?Ex[x + y]
        !Print
        !Plot[x]
        !Plot[x, y]
        !Plot[x, 0.9]
        !Plot[x, 10]
        !Plot[x, ∞]
        !Plot[x, y, 0.5]
        ?Opt[x, p, MAX]
        ?Opt[x, p, MIN]
    """
    _agree(ppy, pgcl)


def test_query_errors():
    with pytest.raises(PpySyntaxError, match="line 1, column 9: Queries must come at the end"):
        parse_ppy("x: nat; query Pr[x == 1]; x = 1;")
    with pytest.raises(PpySyntaxError, match="Unknown query 'Var'"):
        parse_ppy("x: nat; query Var[x];")
    with pytest.raises(PpySyntaxError, match="either MAX or MIN"):
        parse_ppy("x: nat; param p: real; optimize(x, p, BEST);")


# --- the example of the design document ---------------------------------------


def test_design_example():
    ppy = """
        n, m: nat;
        param p: real;

        while (n > 0) {
            {m = m + 1;} [p] {m = m + 2;}
            c: nat = categorical({1: 1/2, 2: 1/4, 3: 1/4});
            switch (c) {
                case 1: { n = n - 1; }
                case 2: { m = 0; }
            }
        }
        query Pr[m == 5];
    """
    program = parse_ppy(ppy)
    assert isinstance(program.instructions[0].body[1], LocalDeclInstr)
    assert isinstance(program.instructions[0].body[2], IfInstr)
    result = compile_pgcl(ppy, syntax="ppy")
    assert not isinstance(result, CheckFail), result
    assert "c" in result.variables
    assert str(result.instructions[0].body[-1]) == "c := 0;"


# --- syntax errors ------------------------------------------------------------


@pytest.mark.parametrize("source, message", [
    ("x: nat;\nx = 1", "line 2, column 6: The program ends unexpectedly, expected ';'."),
    ("x: nat;\nx := 1;", "line 2, column 4: Unexpected '='"),
    ("x: nat;\nif (x == 1) { x = 2 }", "line 2, column 21: Unexpected '}', expected ';'."),
    ("x: nat; x = (1;", "line 1, column 15: Unexpected ';', expected ')'."),
    ("x: nat; y = if;", "line 1, column 13: 'if' is a reserved word"),
    ("switch: nat;", "line 1, column 7: Unexpected ':'"),
])
def test_syntax_error_positions(source, message):
    with pytest.raises(PpySyntaxError) as info:
        parse_ppy(source)
    assert str(info.value).startswith(message)
    line, column = (int(part.split()[-1]) for part in message.split(":")[0].split(","))
    assert (info.value.line, info.value.column) == (line, column)
    assert isinstance(info.value, SyntaxError)


@pytest.mark.parametrize("source, hint", [
    ("x: nat; x := 1;", "assignments in .ppy are written 'x = e'"),
    ("x: nat; x = 1;\n// a comment\n", "comments in .ppy start with '#'"),
    ("x: nat; x = x ^ 2;", "powers in .ppy are written 'x ** 2'"),
    ("x: nat; x = 1; ?Pr[x = 1]", "queries in .ppy are written"),
    ("x: nat; while (x & 1) { skip; }", "use 'and' or '&&'"),
])
def test_syntax_error_hints(source, hint):
    with pytest.raises(PpySyntaxError) as info:
        parse_ppy(source)
    assert f"({hint}" in str(info.value)


def test_comments():
    _agree("# a comment\nx: nat; # another\nx = 1;", "nat x; x := 1")


def test_block_comments():
    # at the start of a line, inside an expression, spanning lines, next to '//' and '#'
    _agree("/* head */ x: nat;\nx = 1 /* one */ + /* two\n lines # not a line comment */ 2;",
           "nat x; x := 1 + 2")
    assert parse_ppy("x: nat; x = 7 // /* floor */ 2;") == parse_ppy("x: nat; x = 7 // 2;")
    # '//*' is floor division followed by '*', as the longest token wins
    with pytest.raises(PpySyntaxError, match="Unexpected '\\*'"):
        parse_ppy("x: nat; x = 7 //* 3 */ 2;")
    assert parse_ppy("x: nat; # /* not opened\nx = 1;") == parse_ppy("x: nat; x = 1;")
    # positions after a multi-line comment stay right
    with pytest.raises(PpySyntaxError) as error:
        parse_ppy("/* a\nb\n*/\nx: nat;\nx = = 1;")
    assert (error.value.line, error.value.column) == (5, 5)


@pytest.mark.parametrize("source, line, column", [
    ("x: nat;\n/* never closed\nx = 1;", 2, 1),
    ("x: nat;\nx = 1 /* open", 2, 7),
    ("/* closed */ x: nat; /* open", 1, 22),
])
def test_unterminated_block_comment(source, line, column):
    with pytest.raises(PpySyntaxError, match="never closed") as error:
        parse_ppy(source)
    assert (error.value.line, error.value.column) == (line, column)


# --- selecting the syntax -------------------------------------------------------


def test_select_syntax():
    assert select_syntax() == "pgcl"
    assert select_syntax(path="a.pgcl") == "pgcl"
    assert select_syntax(path=Path("dir/a.ppy")) == "ppy"
    assert select_syntax("ppy", path="a.pgcl") == "ppy"
    with pytest.raises(ValueError, match="Unknown syntax"):
        select_syntax("python")


def test_compile_by_extension_or_argument():
    expected = compile_pgcl("nat x; x := 1")
    assert compile_pgcl("x: nat; x = 1;", path="prog.ppy") == expected
    assert compile_pgcl("x: nat; x = 1;", syntax="ppy") == expected
    assert parse_pgcl("x: nat; x = 1;", "ppy") == parse_pgcl("nat x; x := 1")
    assert compile_pgcl("nat x; x := 1", path="prog.pgcl") == expected


# --- pretty printer -------------------------------------------------------------


def test_printer_canonical_spelling():
    assert expr_to_ppy(parse_ppy_expr("!a && b || c")) == "not a and b or c"
    assert expr_to_ppy(parse_ppy_expr("(a or b) and c")) == "(a or b) and c"
    assert expr_to_ppy(parse_ppy_expr("x - (y - 1)")) == "x - (y - 1)"
    assert expr_to_ppy(parse_ppy_expr("x * (1/2)")) == "x * (1/2)"
    assert expr_to_ppy(parse_ppy_expr("(-x) ** 2")) == "(-x) ** 2"
    assert expr_to_ppy(parse_ppy_expr("(x ** 2) ** 3")) == "(x ** 2) ** 3"


def test_printer_literals_round_trip():
    for text in ["5.", "0.10", "1E+5", "4/2", "∞"]:
        expr = parse_expectation(text)
        _same(parse_ppy_expr(expr_to_ppy(expr)), expr)


def test_printer_rejects_what_ppy_cannot_express():
    with pytest.raises(PpyPrintError, match="reserved word"):
        program_to_ppy(parse_pgcl("nat case; case := 1"))
    with pytest.raises(PpyPrintError, match="negative literal"):
        expr_to_ppy(RealLitExpr("-0.5"))


def test_printer_block_declarations():
    program = parse_ppy("x: nat; while (x < 3) { y: nat = 1; x = x + y; }")
    assert program_to_ppy(program) == (
        "x: nat;\nwhile (x < 3) {\n    y: nat = 1;\n    x = x + y;\n}\n")


def test_function_call_expression_spelling():
    expr = FunctionCallExpr("f", ([NatLitExpr(1)], {"b": NatLitExpr(2)}))
    assert expr_to_ppy(expr) == "f(1, b = 2)"


@pytest.mark.parametrize("code, column", [
    ("x: nat = 0;\n{x = 1;} [1/0] {x = 2;}", 11),
    ("x: nat = 0;\nx = bernoulli(1/0);", 15),
])
def test_literal_division_by_zero(code, column):
    # #236: a clear message with the position instead of Fraction(1, 0)
    with pytest.raises(PpySyntaxError, match=r"Division by zero in 1/0") as info:
        parse_ppy(code)
    assert (info.value.line, info.value.column) == (2, column)
