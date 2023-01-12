from __future__ import annotations

from typing import List

from probably.pgcl import Expr, Var, UnopExpr, Unop, VarExpr, mut_expr_children
from probably.util.ref import Mut


def _vars_of_expr(expr: Expr) -> List[Var]:
    """Generates a list of variables an expression "reads" from; also known as "free variables" when having binding."""
    if isinstance(expr, UnopExpr) and expr.operator == Unop.IVERSON:
        return []
    if isinstance(expr, VarExpr):
        return [expr.var]
    buffer: List[Var] = []
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        buffer.extend(_vars_of_expr(child_ref.val))
    return buffer
