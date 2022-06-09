from probably.pgcl.ast.declarations import ConstDecl, VarDecl
from probably.pgcl.ast.expressions import NatLitExpr, VarExpr
from probably.pgcl.ast.types import NatType, RealType
from probably.pgcl.check import CheckFail, check_program
from probably.pgcl.parser import parse_pgcl


def test_correct_program():
    program = parse_pgcl("""
        const pi := 3.14
        nat x
        real y
        const twopi := 2 * pi
        while(x < 10 & false) {
            skip
        }
        if(pi = 5.0 || false) {
            x := 3
        } {
            {
                loop(300) {
                    // tick(77)
                    skip
                }
            } [y] {
                observe(true)
            }
        }
        ?Pr [true]
        ?Pr [5]
        ?Ex [7 = 8]
        !Plot [x, y, 5]
    """)
    assert check_program(program) is None


def test_for_declaration_errors():
    program = parse_pgcl("""
        const twopi := 2 * pi
        const pi := 3.14
    """)
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == VarExpr("pi")
    assert "Constant is not yet defined" in check.message

    program = parse_pgcl("""
        const pi := 3.14
        nat x
        const xpi := x * pi
    """)
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == VarExpr("x")
    assert "Variable used in constant definition is not a constant" in check.message

    program = parse_pgcl("""
        nat x
        nat x
    """)
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == VarDecl("x", NatType(None))
    assert "Already declared variable/constant before." in check.message

    program = parse_pgcl("""
        const x := 4
        const x := 5
    """)
    check = check_program(program)
    assert isinstance(check, CheckFail)
    assert check.location == ConstDecl("x", NatLitExpr(5))
    assert "Already declared variable/constant before." in check.message


def test_for_instr_errors():
    program = parse_pgcl("""
        real x
        while(x) {
            skip
        }
    """)
    assert isinstance(check_program(program), CheckFail)

    program = parse_pgcl("""
        real x
        if(x) {
            skip
        } else {
            // ???
        }
    """)
    assert isinstance(check_program(program), CheckFail)

    program = parse_pgcl("""
        real x
        x := 5
    """)
    assert isinstance(check_program(program), CheckFail)

    program = parse_pgcl("""
        x := 5
    """)
    assert isinstance(check_program(program), CheckFail)

    program = parse_pgcl("""
        {
            skip
        } [true] {
            skip
        }
    """)
    assert isinstance(check_program(program), CheckFail)

    program = parse_pgcl("""
        nparam x
        x := 5
    """)
    assert isinstance(check_program(program), CheckFail)
