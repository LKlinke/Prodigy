from probably.pgcl.ast.declarations import VarDecl
from probably.pgcl.ast.expressions import (FunctionCallExpr, InferExpr,
                                           NatLitExpr)
from probably.pgcl.ast.instructions import AsgnInstr
from probably.pgcl.ast.types import DistributionType
from probably.pgcl.parser import parse_pgcl


def test_basic_inference():
    prog = parse_pgcl("""
        dist d;
        fun f := {
            nat x;
            return 7;
        }
        d := infer { f(x := 10) };
    """)
    assert prog.declarations[0] == VarDecl('d', DistributionType())
    assert prog.instructions == [
        AsgnInstr(lhs='d',
                  rhs=InferExpr(
                      FunctionCallExpr('f', ([], {
                          'x': NatLitExpr(10)
                      }))))
    ]
