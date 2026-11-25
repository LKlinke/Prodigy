from fractions import Fraction

from pytest import raises

from prodigy.language import parse_program
from prodigy.language.ppy_parser import parse_ppy_expr
from prodigy.language.ast.declarations import FunctionDecl, VarDecl
from prodigy.language.ast.expressions import (Binop, BinopExpr, CategoricalExpr,
                                           FunctionCallExpr, NatLitExpr,
                                           RealLitExpr, VarExpr)
from prodigy.language.ast.instructions import AsgnInstr
from prodigy.language.ast.types import NatType
from prodigy.language.ppy_parser import parse_ppy_expr


def test_basic_function():
    prog = parse_program("""
def f(x: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(x = 10);""")
    decl = prog.declarations[0]
    assert isinstance(decl, FunctionDecl)
    fun = decl.body
    assert fun.declarations == [VarDecl(var='x', typ=NatType(bounds=None))]
    assert fun.instructions == [AsgnInstr(lhs='x', rhs=NatLitExpr(10))]
    assert fun.returns == VarExpr(var='x')
    assert prog.instructions == [
        AsgnInstr(lhs='x',
                  rhs=FunctionCallExpr('f', ([], {
                      'x': NatLitExpr(10)
                  })))
    ]

    prog = parse_program("""
def f(x: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(10);""")
    decl = prog.declarations[0]
    assert isinstance(decl, FunctionDecl)
    fun = decl.body
    assert fun.declarations == [VarDecl(var='x', typ=NatType(bounds=None))]
    assert fun.instructions == [AsgnInstr(lhs='x', rhs=NatLitExpr(10))]
    assert fun.returns == VarExpr(var='x')
    assert prog.instructions == [
        AsgnInstr(lhs='x', rhs=FunctionCallExpr('f', ([NatLitExpr(10)], {})))
    ]


def test_positional_and_named_parameters():
    prog = parse_program("""
def f(x: nat, y: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(5, y = 10);""")
    assert prog.instructions == [
        AsgnInstr(lhs='x',
                  rhs=FunctionCallExpr('f', ([NatLitExpr(5)], {
                      'y': NatLitExpr(10)
                  })))
    ]
    assert prog.functions['f'].params_to_dict(
        prog.instructions[0].rhs.params) == {
            'x': NatLitExpr(5),
            'y': NatLitExpr(10)
        }

    prog = parse_program("""
def f(x: nat, y: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(y = 10);""")
    assert prog.instructions == [
        AsgnInstr(lhs='x',
                  rhs=FunctionCallExpr('f', ([], {
                      'y': NatLitExpr(10)
                  })))
    ]
    assert prog.functions['f'].params_to_dict(
        prog.instructions[0].rhs.params) == {
            'x': NatLitExpr(0),
            'y': NatLitExpr(10)
        }

    prog = parse_program("""
def f(x: nat, y: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(7);""")
    assert prog.instructions == [
        AsgnInstr(lhs='x', rhs=FunctionCallExpr('f', ([NatLitExpr(7)], {})))
    ]
    assert prog.functions['f'].params_to_dict(
        prog.instructions[0].rhs.params) == {
            'x': NatLitExpr(7),
            'y': NatLitExpr(0)
        }

    prog = parse_program("""
def f(x: nat, y: nat, z: nat, a: nat, b: nat, c: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(7, 8, 9, 10, 42);""")
    assert prog.instructions[0].rhs.params[0] == [
        NatLitExpr(7),
        NatLitExpr(8),
        NatLitExpr(9),
        NatLitExpr(10),
        NatLitExpr(42)
    ]
    assert prog.functions['f'].params_to_dict(
        prog.instructions[0].rhs.params) == {
            'x': NatLitExpr(7),
            'y': NatLitExpr(8),
            'z': NatLitExpr(9),
            'a': NatLitExpr(10),
            'b': NatLitExpr(42),
            'c': NatLitExpr(0)
        }


def test_likely_expr():
    prog = parse_program("""
def f(x: nat, y: nat) -> nat {
    x = 10;
    return x;
}
x: nat;
x = f(categorical({7: 3/5, 42: 2/5}));""")
    assert prog.instructions == [
        AsgnInstr(lhs='x',
                  rhs=FunctionCallExpr('f', ([
                      CategoricalExpr(
                          [(NatLitExpr(7), RealLitExpr(Fraction(6, 10))),
                           (NatLitExpr(42), RealLitExpr(Fraction(4, 10)))])
                  ], {})))
    ]
    assert prog.functions['f'].params_to_dict(
        prog.instructions[0].rhs.params) == {
            'x':
            CategoricalExpr([(NatLitExpr(7), RealLitExpr(Fraction(6, 10))),
                             (NatLitExpr(42), RealLitExpr(Fraction(4, 10)))]),
            'y':
            NatLitExpr(0)
        }


def test_nested_params():
    prog = parse_program("""
x = f(g(h), y);""")
    assert prog.instructions == [
        AsgnInstr(lhs='x',
                  rhs=FunctionCallExpr(function="f",
                                       params=([
                                           FunctionCallExpr(
                                               function="g",
                                               params=([VarExpr("h")], {})),
                                           VarExpr("y")
                                       ], {})))
    ]


def test_expression():
    expr = parse_ppy_expr("x * sin(x)")
    assert expr == BinopExpr(Binop.TIMES, VarExpr("x"),
                             FunctionCallExpr("sin", ([VarExpr("x")], {})))
