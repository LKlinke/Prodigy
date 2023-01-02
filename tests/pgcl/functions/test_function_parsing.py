from fractions import Fraction

from pytest import raises

from probably.pgcl import parse_pgcl
from probably.pgcl.ast.declarations import FunctionDecl, VarDecl
from probably.pgcl.ast.expressions import (CategoricalExpr, FunctionCallExpr,
                                           NatLitExpr, RealLitExpr, VarExpr)
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
                  rhs=FunctionCallExpr('f', ([], {
                      'x': NatLitExpr(10)
                  })))
    ]

    prog = parse_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return x;
        }
        nat x;
        x := f(10);
    """)
    decl = prog.declarations[0]
    assert isinstance(decl, FunctionDecl)
    fun = decl.body
    assert fun.declarations == [VarDecl(var='x', typ=NatType(bounds=None))]
    assert fun.instructions == [AsgnInstr(lhs='x', rhs=NatLitExpr(10))]
    assert fun.returns == VarExpr(var='x')
    assert prog.instructions == [
        AsgnInstr(lhs='x', rhs=FunctionCallExpr('f', ([NatLitExpr(10)], {})))
    ]


def test_illegal_names():
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


def test_wrong_declaration_type():
    with raises(SyntaxError, match="Only VarDecls are allowed in functions"):
        parse_pgcl("""
            fun f := {
                rparam x;
                return 10;
            }
            nat x;
            x := f(x := 10);
        """)

    with raises(SyntaxError, match="Only VarDecls are allowed in functions"):
        parse_pgcl("""
            fun f := {
                fun g := {
                    return 42;
                };
                return 10;
            }
            nat x;
            x := f(x := 10);
        """)

    with raises(SyntaxError, match="Only VarDecls are allowed in functions"):
        parse_pgcl("""
            fun f := {
                const g := 10;
                return g;
            }
            nat x;
            x := f(x := 10);
        """)


def test_positional_and_named_parameters():
    prog = parse_pgcl("""
        fun f := {
            nat x;
            nat y;
            x := 10;
            return x;
        }
        nat x;
        x := f(5, y := 10);
    """)
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

    prog = parse_pgcl("""
        fun f := {
            nat x;
            nat y;
            x := 10;
            return x;
        }
        nat x;
        x := f(y := 10);
    """)
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

    prog = parse_pgcl("""
        fun f := {
            nat x;
            nat y;
            x := 10;
            return x;
        }
        nat x;
        x := f(7);
    """)
    assert prog.instructions == [
        AsgnInstr(lhs='x', rhs=FunctionCallExpr('f', ([NatLitExpr(7)], {})))
    ]
    assert prog.functions['f'].params_to_dict(
        prog.instructions[0].rhs.params) == {
            'x': NatLitExpr(7),
            'y': NatLitExpr(0)
        }

    prog = parse_pgcl("""
        fun f := {
            nat x;
            nat y;
            nat z;
            nat a;
            nat b;
            nat c;
            x := 10;
            return x;
        }
        nat x;
        x := f(7, 8, 9, 10, 42);
    """)
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
    prog = parse_pgcl("""
        fun f := {
            nat x;
            nat y;
            x := 10;
            return x;
        }
        nat x;
        x := f(7 : 6/10 + 42: 4/10);
    """)
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
    prog = parse_pgcl("""
        x := f(g(h), y);
    """)
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
