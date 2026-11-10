from pytest import raises

from prodigy.pgcl.ast import *
from prodigy.pgcl.parser import parse_pgcl
from prodigy.pgcl.syntax_errors import ProgramSyntaxError


def test_all_distributions():
    program = parse_pgcl("""
        nat x;
        x := unif_d(1,2)
        x := unif(1,2)
        x := unif_c(1,2)
        x := geometric(1)
        x := poisson(1)
        x := logdist(1)
        x := bernoulli(1)
        x := binomial(1,2)
    """)
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


def test_syntax_errors():
    illegal_names = {"true", "false"}
    for name in illegal_names:
        with raises(SyntaxError) as e:
            parse_pgcl(f"""
                nat {name}
            """)
        assert "Illegal variable name: " in e.value.msg

    with raises(SyntaxError) as e:
        parse_pgcl(f"""
            nat x
            nat y
            !Plot[x,y,true]
        """)
    assert "Plot instructions cannot handle boolean literals as arguments" in e.value.msg

    with raises(SyntaxError) as e:
        parse_pgcl(f"""
            nat x
            !Plot[x,true]
        """)
    assert "Plot instruction does not support boolean operators" in e.value.msg


def test_literal_division_by_zero():
    # #236: a clear message with the position instead of Fraction(1, 0)
    for code, column in [("nat x;\n{x := 1} [1/0] {x := 2}", 11), ("nat x;\nx := bernoulli(1/0)", 16)]:
        with raises(ProgramSyntaxError, match=r"Division by zero in 1/0") as info:
            parse_pgcl(code)
        assert (info.value.line, info.value.column) == (2, column)
