from probably.pgcl.ast.declarations import VarDecl
from probably.pgcl.ast.expressions import (FunctionCallExpr, InferExpr,
                                           NatLitExpr)
from probably.pgcl.ast.instructions import AsgnInstr
from probably.pgcl.ast.types import DistributionType
from probably.pgcl.parser import parse_pgcl


def test_basic_inference():
    prog = parse_pgcl("""
        dist d;
        d := infer { f(x := 10) };
    """)
    assert prog.declarations == [VarDecl('d', DistributionType())]
    assert prog.instructions == [
        AsgnInstr(lhs='d',
                  rhs=InferExpr(FunctionCallExpr('f', {'x': NatLitExpr(10)})))
    ]
