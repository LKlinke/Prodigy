import pytest

from prodigy.pgcl.check import CheckFail
from prodigy.pgcl.compiler import compile_pgcl


def test_correct_function():
    compile_pgcl("""
def f(x: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(x = 10);""")


def test_return_type():
    prog = compile_pgcl("""
def f(x: nat) -> nat {
    x = 10;
    return 0.5;
}
x: nat;
x = f(x = 10);""")
    assert isinstance(prog, CheckFail)
    assert prog.message == "Functions may only return integers"

    prog = compile_pgcl("""
def f(x: nat) -> nat {
    x = 10;
    return true;
}
x: nat;
x = f(x = 10);""")
    assert isinstance(prog, CheckFail)
    assert prog.message == "Functions may only return integers"


def test_variable_types():
    prog = compile_pgcl("""
def f(x: bool) -> nat {
    x = true;
    return 1;
}
x: nat;
x = f(x = 10);""")
    assert isinstance(prog, CheckFail)
    assert prog.message == "Only variables of type nat or dist are allowed in functions"

    prog = compile_pgcl("""
def f(x: real) -> nat {
    x = true;
    return 1;
}
x: nat;
x = f(x = 10);""")
    assert isinstance(prog, CheckFail)
    assert prog.message == "Only variables of type nat or dist are allowed in functions"


def test_parameter_type():
    prog = compile_pgcl("""
def f(x: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(x = 1/2);""")
    assert isinstance(prog, CheckFail)
    assert prog.message == 'Expected value of type NatType(bounds=None), got RealType().'


def test_wrong_assignment():
    prog = compile_pgcl("""
def f(x: nat) -> nat {
    x = 10;
    return x;
}
x: bool;
x = f(x = 10);""")
    assert isinstance(prog, CheckFail)
    assert prog.message == "Expected value of type BoolType(), got NatType(bounds=None) that is not provably 0 or 1."


def test_wrong_input_distribution():
    with pytest.raises(SyntaxError, match="'f' has no parameter 'y'"):
        compile_pgcl("""
def f(x: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
y: nat;
x = f(y = 10);""")


def test_outside_variables():
    prog = compile_pgcl("""
def f() -> nat {
    y = geometric(p);
    return y;
}
x: nat;
y: nat;
x = f();""")
    assert isinstance(prog, CheckFail)
    assert prog.message == "y is not a variable."
