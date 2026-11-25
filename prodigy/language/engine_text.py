"""
---------------------------
Engine text of expressions
---------------------------

:func:`to_engine_text` writes an expression of a program as *engine text*, the one format in which expressions
reach the computer algebra of the engines as strings (SymPy and GiNaC both read it). The format is described in
``docs/source/language-design.md`` ("Engine text"); the one parser of it is
:func:`prodigy.distribution.engine_text.parse_engine_text`. Only expressions go to an engine as text:
instructions and programs never do, and ``str()`` of every node of the AST is its ``.ppy`` form
(:mod:`prodigy.language.ppy_printer`).

This module knows nothing of SymPy or the engines; it only writes the text.

.. doctest::

    >>> from prodigy.language.ppy_parser import parse_ppy_expr
    >>> to_engine_text(parse_ppy_expr("x ** 2 + 3 * y"))
    '((x) ^ (2)) + (3 * y)'
    >>> to_engine_text(parse_ppy_expr("x < 2 and not true"))
    '(x < 2) & not true'
"""
from __future__ import annotations

from prodigy.language.ast.expressions import (Binop, BinopExpr, BoolLitExpr, CategoricalExpr, ExpectationExpr, Expr,
                                              FunctionCallExpr, InferExpr, NatLitExpr, ProbabilityExpr, RealLitExpr,
                                              SampleExpr, SubstExpr, Unop, UnopExpr, VarExpr)

__all__ = ["to_engine_text", "ENGINE_OPERATORS"]

ENGINE_OPERATORS = {
    Binop.OR: "||",
    Binop.AND: "&",
    Binop.LEQ: "<=",
    Binop.LT: "<",
    Binop.GT: ">",
    Binop.GEQ: ">=",
    Binop.EQ: "=",
    Binop.PLUS: "+",
    Binop.MINUS: "-",
    Binop.TIMES: "*",
    Binop.POWER: "^",
    Binop.DIVIDE: "/",
    Binop.MODULO: "%",
    Binop.FLOOR_DIVIDE: "//",
}
"""The spelling of each binary operator in engine text."""

_SIMPLE = (VarExpr, BoolLitExpr, NatLitExpr, RealLitExpr, UnopExpr, ProbabilityExpr, ExpectationExpr)


def _operand(expr: Expr) -> str:
    """An operand: in parentheses unless it is a name, a literal, a unary operation or ``Pr``/``Ex``."""
    text = to_engine_text(expr)
    return text if isinstance(expr, _SIMPLE) else f"({text})"


# pylint: disable=too-many-return-statements
def to_engine_text(expr: Expr) -> str:
    """
    ``expr`` as engine text: every compound operand in parentheses, ``^`` for powers, exact literals (``1/3``,
    ``0.25``), ``&``, ``||``, ``not`` and ``=`` for the Boolean operators and equality. The internal nodes that have
    no ``.ppy`` form (substitutions) are written as well; ``categorical``, ``infer`` and ``sample`` only occur in
    messages, never in engine input.
    """
    if isinstance(expr, VarExpr):
        return expr.var
    if isinstance(expr, BoolLitExpr):
        return str(expr.value).lower()
    if isinstance(expr, (NatLitExpr, RealLitExpr)):
        return str(expr.value)
    if isinstance(expr, UnopExpr):
        if expr.operator == Unop.NEG:
            return f"not {_operand(expr.expr)}"
        return f"-{_operand(expr.expr)}"
    if isinstance(expr, BinopExpr):
        symbol = ENGINE_OPERATORS[expr.operator]
        if expr.operator == Binop.POWER:
            return f"({to_engine_text(expr.lhs)}) {symbol} ({to_engine_text(expr.rhs)})"
        return f"{_operand(expr.lhs)} {symbol} {_operand(expr.rhs)}"
    if isinstance(expr, CategoricalExpr):
        return " + ".join(f"{_operand(value)} : {_operand(prob)}" for value, prob in expr.exprs)
    if isinstance(expr, SubstExpr):
        substitutions = ", ".join(f"{key}/{to_engine_text(value)}" for key, value in expr.subst.items())
        return f"({to_engine_text(expr.expr)})[{substitutions}]"
    if isinstance(expr, FunctionCallExpr):
        arguments = [to_engine_text(arg) for arg in expr.params[0]] + \
            [f"{name} := {to_engine_text(arg)}" for name, arg in expr.params[1].items()]
        return f"{expr.function}({', '.join(arguments)})"
    if isinstance(expr, InferExpr):
        return f"infer {{ {to_engine_text(expr.function)} }}"
    if isinstance(expr, SampleExpr):
        return f"sample {{ {expr.dist} }}"
    if isinstance(expr, ProbabilityExpr):
        return f"Pr[{to_engine_text(expr.cond)}]"
    if isinstance(expr, ExpectationExpr):
        return f"Ex[{to_engine_text(expr.expr)}]"
    raise TypeError(f"{type(expr).__name__} is not an expression")
