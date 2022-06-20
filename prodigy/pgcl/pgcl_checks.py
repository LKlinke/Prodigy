from typing import Union

from probably.pgcl import (Binop, BinopExpr, Expr, NatLitExpr, Program, Unop,
                           UnopExpr, VarExpr)
from probably.pgcl.ast.walk import Walk, mut_expr_children, walk_expr
from probably.util.ref import Mut

from prodigy.distribution.distribution import Distribution


def has_variable(expr: Expr, context: Union[Distribution, Program]) -> bool:
    if isinstance(expr, UnopExpr) and expr.operator == Unop.IVERSON:
        return False
    if isinstance(expr, VarExpr) and expr.var not in context.get_parameters():
        return True
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        if has_variable(child_ref.val, context):
            return True
    return False


def check_is_modulus_condition(expression: Expr) -> bool:
    if isinstance(expression, BinopExpr) \
            and expression.operator == Binop.EQ \
            and isinstance(expression.rhs, NatLitExpr) \
            and isinstance(expression.lhs, BinopExpr) \
            and expression.lhs.operator == Binop.MODULO:
        mod_expr = expression.lhs
        if isinstance(mod_expr.lhs, VarExpr) and isinstance(
                mod_expr.rhs, NatLitExpr):
            return True
    return False


def check_is_constant_constraint(
        expression: Expr, context: Union[Distribution, Program]) -> bool:
    if not isinstance(expression, BinopExpr):
        return False
    if expression.operator not in (Binop.EQ, Binop.LEQ, Binop.LT, Binop.GT,
                                   Binop.GEQ):
        return False
    if isinstance(expression.lhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.rhs)):
            if has_variable(sub_expr.val, context):
                return False
        return True
    elif isinstance(expression.rhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.lhs)):
            if has_variable(sub_expr.val, context):
                return False
        return True
    else:
        return False
