# pylint: skip-file
"""
----------------------
Expression Translation
----------------------

Translation of pGCL expressions to pySMT formulas.
"""

from fractions import Fraction

from pysmt.fnode import FNode
from pysmt.shortcuts import (FALSE, LE, LT, TRUE, And, EqualsOrIff, Int, Ite,
                             Minus, Not, Or, Plus, Real, Times, ToReal,
                             get_type)
from pysmt.typing import INT

from prodigy.pgcl.ast import (Binop, BinopExpr, BoolLitExpr, Expr,
                              RealLitExpr, NatLitExpr, SubstExpr, Unop,
                              UnopExpr, VarExpr)
from prodigy.pysmt.context import TranslationContext


def expr_to_pysmt(context: TranslationContext,
                  expr: Expr,
                  *,
                  is_expectation: bool = False,
                  allow_infinity: bool = False) -> FNode:
    """
    Translate a pGCL expression to a pySMT formula.

    Note that substitution expressions are not allowed here (they are not
    supported in pySMT).

    You can pass in the optional `is_expectation` parameter to have all integer
    values converted to real values.

    If `allow_infinity` is `True`, then infinity expressions will be mapped
    directly to the `infinity` variable of the given
    :py:class:`TranslationContext`. Take care to appropriately constrain the
    `infinity` variable! Note that arithmetic expressions may not contain
    infinity, to prevent expressions like `infinity - infinity`.

    .. doctest::

        >>> from prodigy.pgcl.parser import parse_expr
        >>> from pysmt.shortcuts import Symbol
        >>> from pysmt.typing import INT

        >>> expr = parse_expr("x + 4 * 13")
        >>> context = TranslationContext({"x": Symbol("x", INT)})
        >>> expr_to_pysmt(context, expr)
        (x + (4 * 13))
    """
    if isinstance(expr, BoolLitExpr):
        return TRUE() if expr.value else FALSE()
    elif isinstance(expr, NatLitExpr):
        if is_expectation:
            return ToReal(Int(expr.value))
        else:
            return Int(expr.value)
    elif isinstance(expr, RealLitExpr):
        if expr.is_infinite():
            if not allow_infinity:
                raise Exception(
                    f"Infinity is not allowed in this expression: {expr}")
            return context.infinity
        else:
            return Real(Fraction(expr.value))
    elif isinstance(expr, VarExpr):
        var = context.variables[expr.var]
        if is_expectation and get_type(var) == INT:
            var = ToReal(var)
        return var
    elif isinstance(expr, UnopExpr):
        operand = expr_to_pysmt(context,
                                expr.expr,
                                is_expectation=False,
                                allow_infinity=allow_infinity)
        if expr.operator == Unop.NEG:
            return Not(operand)
        elif expr.operator == Unop.IVERSON:
            return Ite(operand, Real(1), Real(0))
    elif isinstance(expr, BinopExpr):
        # `is_expectation` is disabled if we enter a non-arithmetic expression
        # (we do not convert integers to reals within a boolean expression such
        # as `x == y`, for example).
        #
        # Similarly, `allow_infinity` is disabled if we enter an arithmetic
        # expression because calculations with infinity are hard to make sense of.
        is_arith_op = expr.operator in [Binop.PLUS, Binop.MINUS, Binop.TIMES]
        is_expectation = is_expectation  # TODO: and is_arith_op
        allow_infinity = allow_infinity  # TODO: and not is_arith_op?!??!
        lhs = expr_to_pysmt(context,
                            expr.lhs,
                            is_expectation=is_expectation,
                            allow_infinity=allow_infinity)
        rhs = expr_to_pysmt(context,
                            expr.rhs,
                            is_expectation=is_expectation,
                            allow_infinity=allow_infinity)
        if expr.operator == Binop.OR:
            return Or(lhs, rhs)
        elif expr.operator == Binop.AND:
            return And(lhs, rhs)
        elif expr.operator == Binop.LEQ:
            return LE(lhs, rhs)
        elif expr.operator == Binop.LE:
            return LT(lhs, rhs)
        elif expr.operator == Binop.EQ:
            return EqualsOrIff(lhs, rhs)
        elif expr.operator == Binop.PLUS:
            return Plus(lhs, rhs)
        elif expr.operator == Binop.MINUS:
            return Ite(LE(lhs, rhs),
                       (Int(0) if get_type(lhs) == INT else Real(0)),
                       Minus(lhs, rhs))
        elif expr.operator == Binop.TIMES:
            return Times(lhs, rhs)
    elif isinstance(expr, SubstExpr):
        raise Exception("Substitution expression is not allowed here.")

    raise Exception("unreachable")
