from typing import Iterable, List, Optional, Set

from prodigy.pgcl import (Binop, BinopExpr, Expr, Instr, NatLitExpr, Program,
                           Unop, UnopExpr, VarDecl, VarExpr)
from prodigy.pgcl.ast.expressions import InferExpr, SampleExpr
from prodigy.pgcl.ast.types import DistributionType
from prodigy.pgcl.ast.walk import (Walk, mut_expr_children, walk_expr,
                                   walk_instr_exprs, walk_instrs)
from prodigy.pgcl.util.ref import Mut

NESTED_INFERENCE_UNSUPPORTED = (
    "Nested inference (dist variables, infer { ... } and sample { ... }) "
    "is not supported by the analysis yet")


def has_variable(expr: Expr, params: Optional[Set[str]]) -> bool:
    """
    Evaluates whether the given expression contains a variable. Symbols in the
    `params` set are not considered variables.
    """

    if isinstance(expr, UnopExpr) and expr.operator == Unop.IVERSON:
        return False
    if isinstance(expr, VarExpr) and (params is None
                                      or expr.var not in params):
        return True
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        if has_variable(child_ref.val, params):
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


def check_is_constant_constraint(expression: Expr,
                                 params: Optional[Set[str]]) -> bool:
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


def find_nested_inference(program: Program) -> Optional[str]:
    """
    Returns the first use of nested inference in ``program`` (a ``dist``
    declaration, an ``infer`` or a ``sample`` expression, also inside
    functions) as pGCL text, or ``None`` if there is none. The analysis has no
    semantics for them yet.
    """
    declarations: List[VarDecl] = [
        decl for decl in program.declarations if isinstance(decl, VarDecl)
    ]
    bodies: List[tuple[List[Instr], List[Expr]]] = [(program.instructions, [])]
    for function in program.functions.values():
        declarations.extend(function.declarations)
        bodies.append((function.instructions, [function.returns]))

    for decl in declarations:
        if isinstance(decl.typ, DistributionType):
            return f"dist {decl.var}"
    for instructions, exprs in bodies:
        for expr in _all_exprs(instructions, exprs):
            if isinstance(expr, (InferExpr, SampleExpr)):
                return str(expr)
    return None


def _all_exprs(instructions: List[Instr],
               exprs: List[Expr]) -> Iterable[Expr]:
    for instr_ref in walk_instrs(Walk.DOWN, instructions):
        for expr_ref in walk_instr_exprs(Walk.DOWN, instr_ref.val):
            yield expr_ref.val
    for expr in exprs:
        for expr_ref in walk_expr(Walk.DOWN, Mut.alloc(expr)):
            yield expr_ref.val
