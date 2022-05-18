from probably.pgcl.parser import parse_pgcl
from probably.pgcl.ast import *

def test_all_operators():
    program = parse_pgcl("""
        const pi := 3.14;
        bool b;
        nat x;
        real r;
        b := true || false & true
        b := 6.0 > pi & not (pi = pi*3)
        n := 1 : 0.5 + 2 : 1/2
        n := 1 + 2 + 3
        n := 1^2^3
        n := 1 % 2^3
    """)
    instr: List[AsgnInstr] = program.instructions

    assert instr[0].rhs == BinopExpr(Binop.OR, BoolLitExpr(True), BinopExpr(Binop.AND, BoolLitExpr(False), BoolLitExpr(True)))

    assert instr[1].rhs == BinopExpr(Binop.AND, BinopExpr(Binop.GE, RealLitExpr(Decimal(6.0)), VarExpr("pi", False)),\
         UnopExpr(Unop.NEG, BinopExpr(Binop.EQ, VarExpr("pi", False), BinopExpr(Binop.TIMES, VarExpr("pi", False), NatLitExpr(3)))))

    exprs = instr[2].rhs.exprs
    assert exprs[0] == (NatLitExpr(1), RealLitExpr(Decimal(0.5)))
    assert exprs[1] == (NatLitExpr(2), RealLitExpr(Fraction(1,2)))

    assert instr[3].rhs == BinopExpr(Binop.PLUS, BinopExpr(Binop.PLUS, NatLitExpr(1), NatLitExpr(2)), NatLitExpr(3))

    assert instr[4].rhs == BinopExpr(Binop.POWER, NatLitExpr(1), BinopExpr(Binop.POWER, NatLitExpr(2), NatLitExpr(3)))

    assert instr[5].rhs == BinopExpr(Binop.POWER, BinopExpr(Binop.MODULO, NatLitExpr(1), NatLitExpr(2)), NatLitExpr(3))

    