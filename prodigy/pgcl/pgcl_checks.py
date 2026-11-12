from typing import List, NamedTuple, Optional, Set, TypeGuard

from prodigy.pgcl import Binop, BinopExpr, Expr, FunctionCallExpr, NatLitExpr, Program, VarExpr
from prodigy.pgcl.ast.walk import Walk, instr_exprs, mut_expr_children, walk_expr, walk_instrs
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


def called_functions(program: Program, name: str) -> List[str]:
    """The functions of ``program`` that the body of the function ``name`` calls (also in
    ``infer``), in order."""
    function = program.functions[name]
    exprs: List[Expr] = [function.returns]
    for ref in walk_instrs(Walk.DOWN, function.instructions):
        exprs.extend(instr_exprs(ref.val))
    return [sub.val.function for expr in exprs for sub in walk_expr(Walk.DOWN, Mut.alloc(expr))
            if isinstance(sub.val, FunctionCallExpr) and sub.val.function in program.functions]


def recursion(program: Program, name: str) -> Optional[str]:
    """
    If the function ``name`` calls itself, directly or through other functions, the calls that
    lead back to it (e.g. ``"f calls g, g calls f"``), else ``None``.

    .. doctest::

        >>> from prodigy.pgcl.ppy_parser import parse_ppy
        >>> program = parse_ppy("def f(a: nat) -> nat { b: nat = g(a); return b; }\\n"
        ...                     "def g(a: nat) -> nat { b: nat = f(a); return b; }\\n"
        ...                     "def h(a: nat) -> nat { b: nat = f(a); return b; }\\nx: nat = h(1);")
        >>> recursion(program, "f"), recursion(program, "h")
        ('f calls g, g calls f', None)
    """
    def cycle(path: List[str]) -> Optional[List[str]]:
        for callee in called_functions(program, path[-1]):
            if callee == name:
                return path
            if callee not in path:
                found = cycle(path + [callee])
                if found is not None:
                    return found
        return None

    path = cycle([name])
    if path is None:
        return None
    return ", ".join(f"{caller} calls {callee}" for caller, callee in zip(path, path[1:] + [name]))
