from prodigy.language.ast.declarations import ConstDecl, VarDecl
from prodigy.language.ast.expressions import NatLitExpr, VarExpr
from prodigy.language.ast.types import NatType, RealType
from prodigy.language.check import CheckFail, check_program
from prodigy.language.parser import parse_program


def test_correct_program():
    program = parse_program("""
const pi: real = 3.14;
x: nat;
y: real;
const twopi = 2 * pi;
while (x < 10 and false) {
    skip;
}
if (pi == 5.0 or false) {
    x = 3;
} else {
    {
        loop(300) {
            skip;
        }
    } [y] {
        observe(true);
    }
}
query Pr[true];
query Pr[5];
query Ex[7 == 8];
plot(x, y, 5);""")
    assert check_program(program) is None


def test_for_declaration_errors():
    program = parse_program("""
const twopi = 2 * pi;
const pi: real = 3.14;""")
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == VarExpr("pi")
    assert "Constant is not yet defined" in check.message

    program = parse_program("""
const pi: real = 3.14;
x: nat;
const xpi = x * pi;""")
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == VarExpr("x")
    assert "Variable used in constant definition is not a constant" in check.message

    program = parse_program("""
x: nat;
x: nat;""")
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == VarDecl("x", NatType(None))
    assert "Already declared variable/constant before." in check.message

    program = parse_program("""
const x: nat = 4;
const x: nat = 5;""")
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == ConstDecl("x", NatLitExpr(5))
    assert "Already declared variable/constant before." in check.message


def test_for_instr_errors():
    program = parse_program("""
x: real;
while (x) {
    skip;
}""", check_guards=False)
    assert isinstance(check_program(program), CheckFail)

    program = parse_program("""
x: real;
if (x) {
    skip;
}
# ???""", check_guards=False)
    assert isinstance(check_program(program), CheckFail)

    program = parse_program("""
x: nat;
x = 0.5;""")
    assert isinstance(check_program(program), CheckFail)

    program = parse_program("""
x = 5;""")
    assert isinstance(check_program(program), CheckFail)

    program = parse_program("""
{
    skip;
} [true] {
    skip;
}""")
    # a bool is widened to the probability 0 or 1
    assert check_program(program) is None

    program = parse_program("""
param x: nat;
x = 5;""")
    assert isinstance(check_program(program), CheckFail)
