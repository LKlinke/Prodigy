from probably.pgcl.check import CheckFail
from probably.pgcl.compiler import compile_pgcl


def test_correct_function():
    compile_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return x;
        }
        nat x;
        x := f(x := 10);
    """)


def test_return_type():
    prog = compile_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return 0.5;
        }
        nat x;
        x := f(x := 10);
    """)
    assert isinstance(prog, CheckFail)
    assert prog.message == "Functions may only return integers"

    prog = compile_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return true;
        }
        nat x;
        x := f(x := 10);
    """)
    assert isinstance(prog, CheckFail)
    assert prog.message == "Functions may only return integers"


def test_variable_types():
    prog = compile_pgcl("""
        fun f := {
            bool x;
            x := true;
            return 1;
        }
        nat x;
        x := f(x := 10);
    """)
    assert isinstance(prog, CheckFail)
    assert prog.message == "Only variables of type NatType are allowed in functions"

    prog = compile_pgcl("""
        fun f := {
            real x;
            x := true;
            return 1;
        }
        nat x;
        x := f(x := 10);
    """)
    assert isinstance(prog, CheckFail)
    assert prog.message == "Only variables of type NatType are allowed in functions"


def test_function_calls():
    prog = compile_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return x;
        }
        nat x;
        x := f(x := 1/2);
    """)
    assert isinstance(prog, CheckFail)
    assert prog.message == 'Expected value of type NatType(bounds=None), got RealType().'
