"""
Terms of expressions for the shared logic of distributions (#511): constant values, natural-number expressions
and the terms that are computed into fresh variables (floor division, truncated subtraction).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, Optional, Set

from prodigy.language import Binop, BinopExpr, BoolLitExpr, Expr, NatLitExpr, RealLitExpr, Unop, UnopExpr, VarExpr
from prodigy.language.ast.walk import Walk, walk_expr
from prodigy.language.ppy_printer import spell
from prodigy.language.util.ref import Mut
from prodigy.distribution.values import State


def constant_value(expression: Expr) -> Fraction | None:
    """
    The value of an arithmetic expression without variables, or None if the expression contains
    variables, an operator other than + - * / % ^, or an intermediate result that natural-number
    arithmetic does not allow (division by zero, non-integer modulo or exponent).

    A difference of two natural numbers is truncated at 0 (monus, `3 - 5 = 0`), as `-` on `nat`
    is everywhere (see :meth:`Distribution.update`). A difference with an operand that is not a
    natural number (e.g. `1 - 1/3` in a probability) is the rational difference if it is not
    negative, and None otherwise.
    """
    if isinstance(expression, NatLitExpr):
        return Fraction(expression.value)
    if isinstance(expression, RealLitExpr):
        return expression.to_fraction()
    if not isinstance(expression, BinopExpr):
        return None
    left, right = constant_value(expression.lhs), constant_value(expression.rhs)
    if left is None or right is None:
        return None
    operation = _CONSTANT_OPERATIONS.get(expression.operator)
    return None if operation is None else operation(left, right)


def _difference(left: Fraction, right: Fraction) -> Fraction | None:
    """`left - right` as in :func:`constant_value`: truncated for natural numbers, else only if not negative."""
    if _is_natural(left) and _is_natural(right):
        return max(left - right, Fraction(0))
    return left - right if left >= right else None


def _integers(left: Fraction, right: Fraction) -> bool:
    """Whether `left` and `right` are integers and `right` is no divisor 0."""
    return left.denominator == 1 and right.denominator == 1 and right != 0


_CONSTANT_OPERATIONS: Dict[Binop, Callable[[Fraction, Fraction], Optional[Fraction]]] = {
    Binop.PLUS: lambda left, right: left + right,
    Binop.TIMES: lambda left, right: left * right,
    Binop.MINUS: _difference,
    Binop.DIVIDE: lambda left, right: left / right if right != 0 else None,
    Binop.MODULO: lambda left, right: Fraction(left.numerator % right.numerator) if _integers(left, right) else None,
    Binop.FLOOR_DIVIDE: lambda left, right: (Fraction(left.numerator // right.numerator) if _integers(left, right)
                                             else None),
    Binop.POWER: lambda left, right: left ** right.numerator if right.denominator == 1 else None,
}
"""The operators of :func:`constant_value` on exact values; None for a result that natural-number arithmetic does
not allow."""


def _is_natural(value: Fraction) -> bool:
    return value.denominator == 1 and value >= 0


def is_natural_expression(expression: Expr, variables: Set[str]) -> bool:
    """
    Whether `expression` has a natural number as value in every state: it is built from the
    variables in `variables`, natural number literals and `+ - * % // ^` (no division `/`, no
    parameters, no constants that are not natural numbers). A `-` between two such expressions is
    truncated subtraction (monus).
    """
    if isinstance(expression, NatLitExpr):
        return True
    if isinstance(expression, RealLitExpr):
        return _is_natural(expression.to_fraction())
    if isinstance(expression, VarExpr):
        return expression.var in variables
    if isinstance(expression, BinopExpr) and expression.operator in _NATURAL_OPERATORS:
        return is_natural_expression(expression.lhs, variables) and \
            is_natural_expression(expression.rhs, variables)
    return False


_NATURAL_OPERATORS = {Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.MODULO, Binop.FLOOR_DIVIDE, Binop.POWER}


def without_fractional_constants(expression: BinopExpr) -> BinopExpr | None:
    """
    Rewrites a binary expression with an operand that is a constant but not a natural number (e.g.
    1/2 in `1/2*x`) such that every intermediate result is a natural number again, or returns None
    if neither operand is such a constant.

    A product `p/q * e` becomes `(p * e) / q` and a quotient `e / (p/q)` becomes `(q * e) / p`, so
    the division update checks that the result is a natural number in every state, exactly as for
    `e / q`. Any other operator with such an operand cannot yield a natural number in general and
    raises an error.
    """
    left, right = constant_value(expression.lhs), constant_value(expression.rhs)
    fraction, other = (left, expression.rhs) if left is not None and left.denominator != 1 \
        else (right, expression.lhs)
    if fraction is None or fraction.denominator == 1:
        return None
    numerator, denominator = NatLitExpr(fraction.numerator), NatLitExpr(fraction.denominator)
    if expression.operator == Binop.TIMES:
        return BinopExpr(Binop.DIVIDE, BinopExpr(Binop.TIMES, numerator, other), denominator)
    if expression.operator == Binop.DIVIDE and other is expression.lhs:
        return BinopExpr(Binop.DIVIDE, BinopExpr(Binop.TIMES, denominator, other), numerator)
    if expression.operator == Binop.DIVIDE:
        return BinopExpr(Binop.DIVIDE, numerator, BinopExpr(Binop.TIMES, denominator, other))
    if expression.operator == Binop.FLOOR_DIVIDE:
        raise ValueError(f"Cannot compute {expression}: floor division by or of the non-integer constant "
                         f"{fraction} is not implemented")
    raise ValueError(
        f"Cannot evaluate {expression} because the constant {fraction} is not a natural number")


def computed_term(expression: Expr, variables: Set[str]) -> BinopExpr | None:
    """
    An innermost term of `expression` that the engines cannot evaluate themselves, or None: a
    floor division `e // q`, or a subtraction `a - b` of natural-number expressions (monus, see
    :func:`is_natural_expression`).
    """
    if isinstance(expression, UnopExpr):
        return computed_term(expression.expr, variables)
    if not isinstance(expression, BinopExpr):
        return None
    inner = computed_term(expression.lhs, variables) or computed_term(expression.rhs, variables)
    if inner is not None:
        return inner
    if expression.operator == Binop.FLOOR_DIVIDE:
        return expression
    if expression.operator == Binop.MINUS and is_natural_expression(expression, variables):
        return expression
    return None


def state_value(expression: Expr, state: State) -> Fraction | None:
    """The value of `expression` in `state` (see :func:`constant_value`), or None."""
    def assigned(expr: Expr) -> Expr:
        if isinstance(expr, VarExpr) and expr.var in state.valuations:
            return NatLitExpr(state[expr.var])
        if isinstance(expr, BinopExpr):
            return BinopExpr(expr.operator, assigned(expr.lhs), assigned(expr.rhs))
        return expr
    return constant_value(assigned(expression))


def replace_term(expression: Expr, term: Expr, replacement: Expr) -> Expr:
    """`expression` with the subexpression `term` (by identity) replaced."""
    if expression is term:
        return replacement
    if isinstance(expression, UnopExpr):
        return UnopExpr(expression.operator, replace_term(expression.expr, term, replacement))
    if isinstance(expression, BinopExpr):
        return BinopExpr(expression.operator, replace_term(expression.lhs, term, replacement),
                         replace_term(expression.rhs, term, replacement))
    return expression


def variables_of(expression: Expr) -> Set[str]:
    """The names of the variables in `expression`."""
    return {node.val.var for node in walk_expr(Walk.DOWN, Mut.alloc(expression)) if isinstance(node.val, VarExpr)}


def substitute_value(expression: Expr, name: str, value: int) -> Optional[Expr]:
    """`expression` with the variable `name` replaced by the literal `value`, or None if it contains an
    expression other than variables, literals and unary or binary operators."""
    if isinstance(expression, VarExpr):
        return NatLitExpr(value) if expression.var == name else expression
    if isinstance(expression, (NatLitExpr, RealLitExpr, BoolLitExpr)):
        return expression
    if isinstance(expression, BinopExpr):
        lhs, rhs = substitute_value(expression.lhs, name, value), substitute_value(expression.rhs, name, value)
        return None if lhs is None or rhs is None else BinopExpr(expression.operator, lhs, rhs)
    if isinstance(expression, UnopExpr):
        inner = substitute_value(expression.expr, name, value)
        return None if inner is None else UnopExpr(expression.operator, inner)
    return None


def is_true(variable: VarExpr) -> Expr:
    """The condition that the ``bool`` variable ``variable`` holds: ``not (variable = 0)``."""
    return UnopExpr(Unop.NEG, BinopExpr(Binop.EQ, variable, NatLitExpr(0)))


def spelled(expression: Expr) -> str:
    """The expression as ``.ppy`` text for a message, inside an analysis or not (#489)."""
    return spell(expression)
