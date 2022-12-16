from pytest import raises

from probably.pgcl import parse_pgcl
from probably.pgcl.ast.declarations import FunctionDecl, VarDecl
from probably.pgcl.ast.expressions import FunctionCallExpr, NatLitExpr, VarExpr
from probably.pgcl.ast.instructions import AsgnInstr
from probably.pgcl.ast.types import NatType


def test_basic_function():
    prog = parse_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return x;
        }
        nat x;
        x := f(x := 10);
    """)

    decl = prog.declarations[0]
    assert isinstance(decl, FunctionDecl)
    fun = decl.body
    assert fun.declarations == [VarDecl(var='x', typ=NatType(bounds=None))]
    assert fun.instructions == [AsgnInstr(lhs='x', rhs=NatLitExpr(10))]
    assert fun.returns == VarExpr(var='x')
    assert prog.instructions == [
        AsgnInstr(lhs='x',
                  rhs=FunctionCallExpr(function='f',
                                       input_distr={'x': NatLitExpr(10)}))
    ]


def text_illegal_names():
    with raises(SyntaxError, match='Illegal function name: poisson'):
        parse_pgcl("""
            fun poisson := {
                nat x;
                x := 10;
                return x;
            }
            nat x;
            x := poisson(x := 10);
        """)
