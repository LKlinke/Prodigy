from prodigy.language.ast.declarations import VarDecl
from prodigy.language.ast.expressions import (FunctionCallExpr, InferExpr,
                                           NatLitExpr, SampleExpr)
from prodigy.language.ast.instructions import AsgnInstr
from prodigy.language.ast.program import Program
from prodigy.language.ast.types import DistributionType
from prodigy.language.compiler import compile_program


def test_basic_inference():
    prog = compile_program("""
d: dist;
def f(x: nat) -> nat {
    return 7;
}
d = infer(f, x = 10);""")
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
    prog = compile_program("""
d: dist;
x: nat;
def f(x: nat) -> nat {
    return 7;
}
d = infer(f, x = 10);
x = sample(d);""")
    assert isinstance(prog, Program)
    assert prog.instructions[1] == AsgnInstr(lhs='x', rhs=SampleExpr('d'))
