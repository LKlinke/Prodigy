from typing import NamedTuple, Optional, Set, TypeGuard

from prodigy.pgcl import Binop, BinopExpr, Expr, NatLitExpr, VarExpr
from prodigy.pgcl.ast.walk import Walk, mut_expr_children, walk_expr
from prodigy.pgcl.util.ref import Mut


def has_variable(expr: Expr, params: Optional[Set[str]]) -> bool:
    """
    Evaluates whether the given expression contains a variable. Symbols in the
    `params` set are not considered variables.
    """

    if isinstance(expr, VarExpr) and (params is None
                                      or expr.var not in params):
        return True
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        if has_variable(child_ref.val, params):
            return True
    return False


class ModulusCondition(NamedTuple):
    """The parts of a condition ``var % modulus = remainder``."""
    variable: VarExpr
    modulus: NatLitExpr
    remainder: NatLitExpr


def modulus_condition(expression: Expr) -> Optional[ModulusCondition]:
    """
    Returns the parts of ``expression`` if it has the form
    ``var % modulus = remainder`` with natural literals ``modulus`` and
    ``remainder``, otherwise ``None``.
    """
    if isinstance(expression, BinopExpr) \
            and expression.operator == Binop.EQ \
            and isinstance(expression.rhs, NatLitExpr) \
            and isinstance(expression.lhs, BinopExpr) \
            and expression.lhs.operator == Binop.MODULO:
        mod_expr = expression.lhs
        if isinstance(mod_expr.lhs, VarExpr) and isinstance(
                mod_expr.rhs, NatLitExpr):
            return ModulusCondition(mod_expr.lhs, mod_expr.rhs,
                                    expression.rhs)
    return None


def check_is_modulus_condition(expression: Expr) -> bool:
    return modulus_condition(expression) is not None


def check_is_constant_constraint(
        expression: Expr, params: Optional[Set[str]]) -> TypeGuard[BinopExpr]:
    if not isinstance(expression, BinopExpr) or expression.operator not in (
            Binop.EQ, Binop.LEQ, Binop.LT, Binop.GT, Binop.GEQ):
        return False
    if isinstance(expression.lhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.rhs)):
            if has_variable(sub_expr.val, params):
                return False
        return True
    elif isinstance(expression.rhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.lhs)):
            if has_variable(sub_expr.val, params):
                return False
        return True
    else:
        return False
