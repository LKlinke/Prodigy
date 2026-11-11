"""
A guard of ``if``/``while`` and the condition of ``observe`` must be of type ``bool`` (issue #269):
the parsers reject a number such as ``x`` or ``bernoulli(1/2)`` there with a type error and its
position, in both syntaxes and in function bodies. A ``bool`` variable is a guard of its own.
"""
import pytest

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.pgcl import Program
from prodigy.pgcl.compiler import compile_pgcl
from prodigy.pgcl.parser import parse_pgcl
from prodigy.pgcl.syntax_errors import ProgramSyntaxError, ProgramTypeError
from prodigy.web.diagnostics import check_source

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

Engine = ForwardAnalysisConfig.Engine
ENGINES = [pytest.param(Engine.SYMPY, id="SymPy"), pytest.param(Engine.SYMENGINE, id="SymEngine"),
           pytest.param(Engine.GINAC, id="GiNaC",
                        marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed"))]

# (source, syntax, line, column, what, guard, type)
REJECTED = [
    pytest.param("x: nat;\nif (x) { x = 1; } else { x = 2; }", "ppy", 2, 1,
                 "the guard of if", "x", "nat", id="ppy-if-nat"),
    pytest.param("nat x;\nif (x) { x := 1 } else { x := 2 }", "pgcl", 2, 1,
                 "the guard of if", "x", "nat", id="pgcl-if-nat"),
    pytest.param("x: nat;\nx = 1;\nif (bernoulli(1/2)) { x = 2; }", "ppy", 3, 1,
                 "the guard of if", "bernoulli(1/2)", "nat", id="ppy-if-draw"),
    pytest.param("nat x;\nx := 1;\nif (bernoulli(1/2)) { x := 2 } else { skip }", "pgcl", 3, 1,
                 "the guard of if", "bernoulli(1/2)", "nat", id="pgcl-if-draw"),
    pytest.param("x: nat;\nwhile (x + 1) { x = 0; }", "ppy", 2, 1,
                 "the guard of while", "x + 1", "nat", id="ppy-while"),
    pytest.param("nat x;\nwhile (x) { x := 0 }", "pgcl", 2, 1,
                 "the guard of while", "x", "nat", id="pgcl-while"),
    pytest.param("x: nat;\nobserve(x);", "ppy", 2, 1,
                 "the condition of observe", "x", "nat", id="ppy-observe"),
    pytest.param("nat x;\nobserve(x);", "pgcl", 2, 1,
                 "the condition of observe", "x", "nat", id="pgcl-observe"),
    pytest.param("x: nat;\nif (1/2) { x = 1; }", "ppy", 2, 1,
                 "the guard of if", "1/2", "real", id="ppy-real"),
    pytest.param("x: nat;\nwhile (x < 3) {\n    if (x) { x = 5; }\n    x = x + 1;\n}", "ppy", 3, 5,
                 "the guard of if", "x", "nat", id="ppy-nested"),
    pytest.param("def f(a: nat) -> nat {\n    if (a) { a = 1; }\n    return a;\n}\nx: nat;\nx = f(1);",
                 "ppy", 2, 5, "the guard of if", "a", "nat", id="ppy-function-body"),
    pytest.param("fun f := {\n    nat a;\n    if (a) { a := 1 } else { skip }\n    return a;\n}\n"
                 "nat x;\nx := f(1);", "pgcl", 3, 5, "the guard of if", "a", "nat",
                 id="pgcl-function-body"),
]


@pytest.mark.parametrize("source, syntax, line, column, what, guard, typ", REJECTED)
def test_guard_that_is_no_bool_is_a_type_error(source, syntax, line, column, what, guard, typ):
    with pytest.raises(ProgramTypeError) as info:
        parse_pgcl(source, syntax)
    error = info.value
    assert isinstance(error, ProgramSyntaxError)
    assert (error.line, error.column) == (line, column)
    assert f"Type error: {what} must be of type bool, but {guard} is of type {typ}." in error.message
    # compile_pgcl parses first, so it raises the same error
    with pytest.raises(ProgramTypeError):
        compile_pgcl(source, syntax=syntax)


def test_message_suggests_a_comparison_or_a_variable():
    with pytest.raises(ProgramTypeError, match="Write a comparison such as x > 0"):
        parse_pgcl("x: nat;\nif (x) { x = 1; }", "ppy")
    with pytest.raises(ProgramTypeError, match="Draw the value into a variable first"):
        parse_pgcl("x: nat;\nif (bernoulli(1/2)) { x = 1; }", "ppy")


def test_the_check_can_be_skipped():
    program = parse_pgcl("nat x; if (x) { x := 1 } else { skip }", check_guards=False)
    assert len(program.instructions) == 1


@pytest.mark.parametrize("source, syntax", [
    pytest.param("b: bool;\nx: nat;\nif (b) { x = 1; }\nwhile (!b) { b = true; }", "ppy", id="ppy"),
    pytest.param("bool b;\nnat x;\nif (b) { x := 1 } else { skip }\nwhile (not b) { b := true }",
                 "pgcl", id="pgcl"),
    pytest.param("const c := true;\nnat x;\nif (c) { x := 1 } else { skip }", "pgcl", id="constant"),
    pytest.param("x: nat;\nif (x > 0 && x < 3 || !(x == 7)) { x = 1; }", "ppy", id="comparisons"),
])
def test_bool_guards_are_accepted(source, syntax):
    assert isinstance(parse_pgcl(source, syntax), Program)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, syntax", [
    pytest.param("b: bool;\nc: bool;\nx: nat;\nb = bernoulli(1/3);\nc = bernoulli(1/2);\n"
                 "if (b && !c) { x = 1; } else { x = 2; }\nobserve(b || c);", "ppy", id="ppy"),
    pytest.param("bool b;\nbool c;\nnat x;\nb := bernoulli(1/3);\nc := bernoulli(1/2);\n"
                 "if (b & not c) { x := 1 } else { x := 2 }\nobserve(b || c);", "pgcl", id="pgcl"),
])
def test_bool_variable_as_guard_is_analysed(engine, source, syntax):
    program = compile_pgcl(source, syntax=syntax)
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine)
    with config.activate():
        result, error = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    # b = 1, c = 0: x = 1 (1/6); b = 1, c = 1: x = 2 (1/6); b = 0, c = 1: x = 2 (1/3);
    # b = 0, c = 0 is observed away (1/3), and the result is normalised by 2/3
    expected = config.factory.from_expr("1/4*b*x + 1/4*b*c*x^2 + 1/2*c*x^2", "b", "c", "x")
    assert error == config.factory.from_expr("1/3", *error.get_variables()), str(error)
    assert result == expected, str(result)


@pytest.mark.parametrize("source, syntax", [
    pytest.param("x: nat;\nif (x) { x = 1; }", "ppy", id="ppy"),
    pytest.param("nat x;\nif (x) { x := 1 } else { skip }", "pgcl", id="pgcl"),
])
def test_gui_reports_the_type_error_as_error(source, syntax):
    result = check_source(source, syntax)
    assert not result["ok"]
    error = result["diagnostics"][0]
    assert (error["severity"], error["line"], error["column"]) == ("error", 2, 1)
    assert "must be of type bool" in error["message"]
