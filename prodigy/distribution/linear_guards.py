"""
Exact normalisation of linear guards (:func:`normalize_guard`).

A comparison whose sides are built from variables, constants, `+`, multiplication by a constant
and division by a constant is linear. It is compared exactly over the rationals, e.g.
`1/2*x < 2` holds if and only if `x < 4`. As everywhere, `/` is rational division, so `x/2 <= 1`
means `x <= 2` and not floor division. The comparison is rewritten into an equivalent guard with natural
numbers only:

- with one variable it becomes `x < n`, `x <= n`, `x = n`, `x > n` or `x >= n` for a natural
  number `n`, or `true` or `false` (`x = 3/2` never holds), which every engine can filter even on
  infinite distributions;
- with several variables and a constant that is not a natural number or a division, both sides
  are multiplied by the least common denominator, e.g. `1/2*x + 1/3*y < 1` becomes
  `3*x + 2*y < 6`;
- without variables it becomes `true` or `false`.

Subtraction is not linear here, as `x - y` on natural numbers is truncated at 0 by some engines.
A guard with a constant that is not a natural number but that is not linear (e.g. `x*y < 1/2` or
`x % 2 = 1/2`) is rejected on every engine, as its value depends on how an engine treats
non-natural intermediate results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Set

from prodigy.pgcl import (Binop, BinopExpr, BoolLitExpr, Expr, NatLitExpr,
                          RealLitExpr, VarExpr)
from prodigy.pgcl.ast.walk import Walk, walk_expr
from prodigy.pgcl.util.ref import Mut

LinearForm = Dict[str, Fraction]
"""A linear combination of variables; the key `""` holds the constant."""

_COMPARISONS = {Binop.LT, Binop.LEQ, Binop.EQ, Binop.GT, Binop.GEQ}
_FLIPPED = {Binop.LT: Binop.GT, Binop.LEQ: Binop.GEQ, Binop.EQ: Binop.EQ,
            Binop.GT: Binop.LT, Binop.GEQ: Binop.LEQ}


def _constant(form: LinearForm) -> Fraction | None:
    """The value of `form` if it has no variables, otherwise None."""
    if any(key != "" for key in form):
        return None
    return form.get("", Fraction(0))


def _scaled(form: LinearForm, factor: Fraction) -> LinearForm:
    return {key: value * factor for key, value in form.items() if value * factor != 0}


def _sum(left: LinearForm, right: LinearForm, sign: int = 1) -> LinearForm:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, Fraction(0)) + sign * value
    return {key: value for key, value in result.items() if value != 0}


def _linear_form(expression: Expr, variables: Set[str]) -> LinearForm | None:
    """The linear form of `expression`, or None if it is not linear in `variables`."""
    # pylint: disable=too-many-return-statements
    if isinstance(expression, NatLitExpr):
        return {"": Fraction(expression.value)} if expression.value != 0 else {}
    if isinstance(expression, RealLitExpr):
        if expression.is_infinite():
            return None
        value = expression.to_fraction()
        return {"": value} if value != 0 else {}
    if isinstance(expression, VarExpr):
        return {expression.var: Fraction(1)} if expression.var in variables else None
    if not isinstance(expression, BinopExpr):
        return None
    left = _linear_form(expression.lhs, variables)
    right = _linear_form(expression.rhs, variables)
    if left is None or right is None:
        return None
    left_value, right_value = _constant(left), _constant(right)
    if expression.operator == Binop.PLUS:
        return _sum(left, right)
    if expression.operator == Binop.TIMES:
        if left_value is not None:
            return _scaled(right, left_value)
        return _scaled(left, right_value) if right_value is not None else None
    if expression.operator == Binop.DIVIDE:
        return _scaled(left, 1 / right_value) if right_value else None
    # Other operators only on constants, with the restrictions of natural-number arithmetic
    if left_value is None or right_value is None:
        return None
    if expression.operator == Binop.MINUS and left_value >= right_value:
        return _sum(left, right, -1)
    if left_value.denominator != 1 or right_value.denominator != 1:
        return None
    if expression.operator == Binop.MODULO and right_value != 0:
        return _sum({}, {"": left_value % right_value})
    if expression.operator == Binop.POWER:
        return _sum({}, {"": left_value**right_value.numerator})
    return None


def _has_fractional_literal(expression: Expr) -> bool:
    """Whether `expression` contains a literal that is not a natural number, e.g. `1/2` or `0.5`."""
    for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression)):
        value = sub_expr.val
        if isinstance(value, RealLitExpr) and not value.is_infinite() and value.to_fraction().denominator != 1:
            return True
    return False


def _has_division(expression: Expr) -> bool:
    return any(isinstance(sub_expr.val, BinopExpr) and sub_expr.val.operator == Binop.DIVIDE
               for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression)))


def _compare(value: Fraction, operator: Binop) -> bool:
    return {Binop.LT: value < 0, Binop.LEQ: value <= 0, Binop.EQ: value == 0,
            Binop.GT: value > 0, Binop.GEQ: value >= 0}[operator]


def _single_variable_guard(variable: str, operator: Binop, bound: Fraction) -> Expr:
    """The guard `variable operator bound` for a natural `variable` with natural numbers only."""
    # pylint: disable=too-many-return-statements
    if operator == Binop.EQ:
        if bound.denominator != 1 or bound < 0:
            return BoolLitExpr(False)
        return BinopExpr(Binop.EQ, VarExpr(variable), NatLitExpr(bound.numerator))
    if operator in (Binop.LT, Binop.GEQ):
        # x < t iff x < ceil(t), and x >= t iff x >= ceil(t)
        limit = math.ceil(bound)
        if limit <= 0:
            return BoolLitExpr(operator == Binop.GEQ)
    else:
        # x <= t iff x <= floor(t), and x > t iff x > floor(t)
        limit = math.floor(bound)
        if limit < 0:
            return BoolLitExpr(operator == Binop.GT)
    return BinopExpr(operator, VarExpr(variable), NatLitExpr(limit))


def _natural_side(form: LinearForm) -> Expr:
    """The sum of the terms of `form` with natural coefficients, which are all positive."""
    terms: list[Expr] = []
    for key in sorted(form):
        coefficient = form[key].numerator
        if key == "":
            terms.append(NatLitExpr(coefficient))
        elif coefficient == 1:
            terms.append(VarExpr(key))
        else:
            terms.append(BinopExpr(Binop.TIMES, NatLitExpr(coefficient), VarExpr(key)))
    if not terms:
        return NatLitExpr(0)
    result = terms[0]
    for term in terms[1:]:
        result = BinopExpr(Binop.PLUS, result, term)
    return result


def normalize_guard(condition: Expr, variables: Set[str]) -> Expr | None:
    """
    An equivalent guard with natural numbers only if `condition` is a linear comparison (see the
    module documentation), or None if there is nothing to normalise.

    Raises a ValueError if `condition` is a comparison with a constant that is not a natural
    number but not linear.
    """
    if not isinstance(condition, BinopExpr) or condition.operator not in _COMPARISONS:
        return None
    left = _linear_form(condition.lhs, variables)
    right = _linear_form(condition.rhs, variables)
    if left is None or right is None:
        if _has_fractional_literal(condition):
            raise ValueError(
                f"Cannot evaluate the guard {condition} exactly: a constant that is not a natural number "
                "is only supported in linear guards (variables, constants, +, and * or / by constants)")
        return None

    difference = _sum(left, right, -1)
    constant = difference.pop("", Fraction(0))
    if not difference:
        return BoolLitExpr(_compare(constant, condition.operator))
    if len(difference) == 1:
        ((variable, coefficient),) = difference.items()
        operator = condition.operator if coefficient > 0 else _FLIPPED[condition.operator]
        return _single_variable_guard(variable, operator, -constant / coefficient)
    if not (_has_fractional_literal(condition) or _has_division(condition)):
        return None

    difference[""] = constant
    denominator = math.lcm(*(value.denominator for value in difference.values()))
    scaled = _scaled(difference, Fraction(denominator))
    positive = {key: value for key, value in scaled.items() if value > 0}
    negative = {key: -value for key, value in scaled.items() if value < 0}
    return BinopExpr(condition.operator, _natural_side(positive), _natural_side(negative))


@dataclass(frozen=True)
class DifferenceComparison:
    """
    A comparison of two variables x and y that holds if and only if `y - x >= shift` (relation
    `"ge"`) or `y - x = shift` (relation `"eq"`) holds, or does not hold if `complement` is set.
    """
    x: str
    y: str
    relation: str
    shift: int
    complement: bool = False


def difference_comparison(condition: Expr, variables: Set[str]) -> DifferenceComparison | BoolLitExpr | None:
    """
    The comparison `condition` of two natural variables as a :class:`DifferenceComparison` if it is
    linear with coefficients a and -a for its two variables, e.g. `x < y + 2` or `2*x <= 2*y + 1`;
    `false` for an equation that never holds (`2*x = 2*y + 1`); None for every other condition.
    """
    if not isinstance(condition, BinopExpr) or condition.operator not in _COMPARISONS:
        return None
    left = _linear_form(condition.lhs, variables)
    right = _linear_form(condition.rhs, variables)
    if left is None or right is None:
        return None
    difference = _sum(left, right, -1)
    constant = difference.pop("", Fraction(0))
    if len(difference) != 2 or sum(difference.values()) != 0:
        return None
    # difference = a * (x - y) + constant with a > 0, so the condition compares x - y with -constant / a
    (x, scale), = ((key, value) for key, value in difference.items() if value > 0)
    (y,) = (key for key, value in difference.items() if value < 0)
    bound = constant / scale  # the condition is `x - y + bound OP 0`, i.e. `y - x  OP'  bound`
    floor_plus_one, ceiling = math.floor(bound) + 1, math.ceil(bound)
    operator = condition.operator
    if operator == Binop.EQ:
        if bound.denominator != 1:
            return BoolLitExpr(False)
        return DifferenceComparison(x, y, "eq", bound.numerator)
    if operator == Binop.LT:  # y - x > bound
        return DifferenceComparison(x, y, "ge", floor_plus_one)
    if operator == Binop.LEQ:  # y - x >= bound
        return DifferenceComparison(x, y, "ge", ceiling)
    if operator == Binop.GT:  # y - x < bound
        return DifferenceComparison(x, y, "ge", ceiling, complement=True)
    # y - x <= bound
    return DifferenceComparison(x, y, "ge", floor_plus_one, complement=True)
