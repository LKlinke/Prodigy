from probably.pgcl.ast.declarations import VarDecl
from probably.pgcl.ast.expressions import (FunctionCallExpr, InferExpr,
                                           NatLitExpr, SampleExpr)
from probably.pgcl.ast.instructions import AsgnInstr
from probably.pgcl.ast.program import Program
from probably.pgcl.ast.types import DistributionType
from probably.pgcl.compiler import compile_pgcl


def test_basic_inference():
    prog = compile_pgcl("""
        dist d;
        fun f := {
            nat x;
            return 7;
        }
        d := infer { f(x := 10) };
    """)
    assert isinstance(prog, Program)
    assert prog.declarations[0] == VarDecl('d', DistributionType())
    assert prog.instructions == [
        AsgnInstr(lhs='d',
                  rhs=InferExpr(
                      FunctionCallExpr('f', ([], {
                          'x': NatLitExpr(10)
                      }))))
    ]


def test_basic_sampling():
    prog = compile_pgcl("""
        dist d;
        nat x;
        fun f := {
            nat x;
            return 7;
        }
        d := infer { f(x := 10) };
        x := sample { d };
    """)
    assert isinstance(prog, Program)
    assert prog.instructions[1] == AsgnInstr(lhs='x', rhs=SampleExpr('d'))
