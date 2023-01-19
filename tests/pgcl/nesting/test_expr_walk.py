from typing import Iterable

from probably.pgcl.ast.expressions import FunctionCallExpr, NatLitExpr, VarExpr
from probably.pgcl.ast.walk import Walk, walk_expr
from probably.util.ref import Mut


def test_function_parameters():
    exprs = [
        ref.val for ref in walk_expr(
            Walk.UP,
            Mut.alloc(
                FunctionCallExpr(function='unif',
                                 params=([NatLitExpr(0),
                                          VarExpr('x')], {}))))
    ]
    assert exprs == [
        NatLitExpr(0),
        VarExpr('x'),
        FunctionCallExpr(function='unif',
                         params=([NatLitExpr(0), VarExpr('x')], {}))
    ]

    exprs = [
        ref.val for ref in walk_expr(
            Walk.UP,
            Mut.alloc(
                FunctionCallExpr(function='unif',
                                 params=([], {
                                     'start': NatLitExpr(0),
                                     'end': VarExpr('x')
                                 }))))
    ]
    assert exprs == [
        NatLitExpr(0),
        VarExpr('x'),
        FunctionCallExpr(function='unif',
                         params=([], {
                             'start': NatLitExpr(0),
                             'end': VarExpr('x')
                         }))
    ]

    exprs = [
        ref.val for ref in walk_expr(
            Walk.UP,
            Mut.alloc(
                FunctionCallExpr(function='unif',
                                 params=([NatLitExpr(0)], {
                                     'end': VarExpr('x')
                                 }))))
    ]
    assert exprs == [
        NatLitExpr(0),
        VarExpr('x'),
        FunctionCallExpr(function='unif',
                         params=([NatLitExpr(0)], {
                             'end': VarExpr('x')
                         }))
    ]
