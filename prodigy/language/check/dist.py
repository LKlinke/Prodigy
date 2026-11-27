"""
Values of type ``dist`` in expressions: which expressions mention one (:func:`mentions_dist`), and the rule that
queries are about the program state, not about a ``dist`` value (:func:`check_query_without_dist`).
"""
from __future__ import annotations

from typing import Iterable, Optional

from prodigy.language.ast import (Binop, DistributionType, ExpectationExpr, Expr, InferExpr, Instr,
                                  ProbabilityExpr, Program, SampleExpr, VarExpr, Walk, walk_expr)
from prodigy.language.util.ref import Mut

from .context import CheckFail, spell_node

INFER_HINT = ("Inside a program, Pr[…] and Ex[…] are about a value of type dist. To use the "
              "distribution of program variables, wrap the code in a function f that returns "
              "the value and write d: dist = infer(f, …); then Pr[d == k] and Ex[d] are real "
              "numbers.")
"""The hint for ``Pr[…]`` and ``Ex[…]`` on the program state inside a program."""

DIST_COMPARISONS = (Binop.EQ, Binop.LT, Binop.LEQ, Binop.GT, Binop.GEQ)
"""The comparisons of a ``dist`` variable with a number that ``Pr[…]`` inside a program accepts."""


def is_dist_variable(program: Program, expr: Expr) -> bool:
    """Whether ``expr`` is a variable of type ``dist``."""
    return isinstance(expr, VarExpr) and isinstance(program.variables.get(expr.var),
                                                    DistributionType)


def mentions_dist(program: Program, expr: Expr) -> bool:
    """Whether ``expr`` uses a variable of type ``dist`` or one of the expressions ``Pr[d …]``,
    ``Ex[d]`` and ``sample(d)``."""
    for ref in walk_expr(Walk.DOWN, Mut.alloc(expr)):
        node = ref.val
        if isinstance(node, (ProbabilityExpr, ExpectationExpr, SampleExpr, InferExpr)) \
                or is_dist_variable(program, node):
            return True
    return False


def check_query_without_dist(program: Program, instr: Instr,
                             exprs: Iterable[Optional[Expr]]) -> Optional[CheckFail]:
    """Queries are about the program state; a ``dist`` value in them is a type error with a hint."""
    for expr in exprs:
        if expr is not None and mentions_dist(program, expr):
            return CheckFail(
                instr, f"The query {spell_node(instr)} uses a value of type dist; queries are about the "
                "program state. Draw from the distribution and query the draw, e.g. "
                "y = sample(d) and then a query of Pr[y == 3] or Ex[y].")
    return None
