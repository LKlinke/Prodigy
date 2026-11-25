from pytest import raises

from prodigy.language.ast import *
from prodigy.language.parser import parse_program
from prodigy.language.syntax_errors import ProgramSyntaxError


def test_all_distributions():
    program = parse_program("""
x: nat;
x = unif_d(1, 2);
x = unif(1, 2);
x = unif_c(1, 2);
x = geometric(1);
x = poisson(1);
x = logdist(1);
x = bernoulli(1);
x = binomial(1, 2);""")
    exprs = [instr.rhs for instr in program.instructions]

    assert exprs == [
        FunctionCallExpr('unif_d',
                         ([NatLitExpr(1), NatLitExpr(2)], {})),
        FunctionCallExpr('unif', ([NatLitExpr(1), NatLitExpr(2)], {})),
        FunctionCallExpr('unif_c',
                         ([NatLitExpr(1), NatLitExpr(2)], {})),
        FunctionCallExpr('geometric', ([NatLitExpr(1)], {})),
        FunctionCallExpr('poisson', ([NatLitExpr(1)], {})),
        FunctionCallExpr('logdist', ([NatLitExpr(1)], {})),
        FunctionCallExpr('bernoulli', ([NatLitExpr(1)], {})),
        FunctionCallExpr('binomial',
                         ([NatLitExpr(1), NatLitExpr(2)], {}))
    ]


def test_literal_division_by_zero():
    # #236: a clear message with the position instead of Fraction(1, 0)
    for code, column in [("x: nat;\n{ x = 1; } [1/0] { x = 2; }", None), ("x: nat;\nx = bernoulli(1/0);", None)]:
        with raises(ProgramSyntaxError, match=r"Division by zero in 1/0") as info:
            parse_program(code)
        assert info.value.line == 2 and column is None
