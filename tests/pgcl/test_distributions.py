from probably.pgcl.parser import parse_pgcl
from probably.pgcl.ast import *

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

    assert exprs == [DUniformExpr(NatLitExpr(1), NatLitExpr(2)),
        DUniformExpr(NatLitExpr(1), NatLitExpr(2)),
        CUniformExpr(NatLitExpr(1), NatLitExpr(2)),
        GeometricExpr(NatLitExpr(1)),
        PoissonExpr(NatLitExpr(1)),
        LogDistExpr(NatLitExpr(1)),
        BernoulliExpr(NatLitExpr(1)),
        BinomialExpr(NatLitExpr(1), NatLitExpr(2))]
