from pytest import raises

from probably.pgcl.ast import *
from probably.pgcl.parser import parse_pgcl


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
        DUniformExpr(NatLitExpr(1), NatLitExpr(2)),
        DUniformExpr(NatLitExpr(1), NatLitExpr(2)),
        CUniformExpr(NatLitExpr(1), NatLitExpr(2)),
        GeometricExpr(NatLitExpr(1)),
        PoissonExpr(NatLitExpr(1)),
        LogDistExpr(NatLitExpr(1)),
        BernoulliExpr(NatLitExpr(1)),
        BinomialExpr(NatLitExpr(1), NatLitExpr(2))
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
