"""
The types of expressions: :func:`get_type`, with one handler of :func:`~prodigy.language.check.context.type_of`
per expression class.
"""
# pylint: disable=broad-exception-raised  # an operator without a rule is a bug of the checker
from __future__ import annotations

from typing import List, Mapping

from prodigy.language.ast import (Binop, BinopExpr, BoolLitExpr, BoolType, CategoricalExpr, DistributionType,
                                  ExpectationExpr, Expr, FunctionCallExpr, InferExpr, NatLitExpr, NatType,
                                  ProbabilityExpr, Program, RealLitExpr, RealType, SampleExpr, Type, Unop, UnopExpr,
                                  Var, VarDecl, VarExpr)

from .calls import check_function_call, sample_predefined_functions
from .compatibility import is_compatible, is_numeric, join
from .context import CheckFail, Ctx, CheckedType, expect, spell_node, type_of
from .dist import DIST_COMPARISONS, INFER_HINT, is_dist_variable, mentions_dist


def get_type(
    program: Program,
    expr: Expr,
    check: bool = True,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> CheckedType:
    """
    Get the type of the expression or expectation.

    Set `check` to `True` to force checking correctness of all subexpressions, even if that is not necessary to
    determine the given expression's type.

    .. doctest::

        >>> from prodigy.language.ast import *
        >>> nat = NatLitExpr(10)
        >>> program = Program(list(), dict(), dict(), dict(), dict(), list())

        >>> get_type(program, BinopExpr(Binop.TIMES, nat, nat))
        NatType(bounds=None)

        >>> get_type(program, VarExpr('x'))
        CheckFail(location=VarExpr('x'), message='variable is not declared')

        >>> get_type(program, UnopExpr(Unop.NEG, BoolLitExpr(True)))
        BoolType()
    """
    return type_of(expr, Ctx(program, predefined_functions, check))


@type_of.register
def _variable(expr: VarExpr, ctx: Ctx) -> CheckedType:
    # a variable, a parameter or a constant (whose type is the type of its value)
    variable_type = ctx.program.variables.get(expr.var)
    if variable_type is not None:
        return variable_type
    parameter_type = ctx.program.parameters.get(expr.var)
    if parameter_type is not None:
        return parameter_type
    value = ctx.program.constants.get(expr.var)
    if value is None:
        return CheckFail(expr, "variable is not declared")
    return type_of(value, ctx.checked())


@type_of.register
def _bool_literal(_expr: BoolLitExpr, _ctx: Ctx) -> CheckedType:
    return BoolType()


@type_of.register
def _nat_literal(_expr: NatLitExpr, _ctx: Ctx) -> CheckedType:
    return NatType(bounds=None)


@type_of.register
def _real_literal(_expr: RealLitExpr, _ctx: Ctx) -> CheckedType:
    return RealType()


@type_of.register
def _unary(expr: UnopExpr, ctx: Ctx) -> CheckedType:
    if ctx.check:
        # Currently, all unary operators take boolean operands
        assert expr.operator == Unop.NEG
        failure = expect(ctx, expr.expr, BoolType())
        if failure is not None:
            return failure
    if expr.operator == Unop.NEG:
        return BoolType()
    raise Exception("unreachable")


_BOOLEAN_OPERATORS = (Binop.OR, Binop.AND)
"""Take booleans, return a boolean."""

_COMPARISONS = (Binop.LEQ, Binop.LT, Binop.EQ, Binop.GEQ, Binop.GT)
"""Take numbers, return a boolean."""

_ARITHMETIC = (Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.MODULO, Binop.POWER, Binop.DIVIDE)
"""Take numbers, return the wider operand type, but at least ``nat`` (``true + true = 2``)."""


@type_of.register
def _binary(expr: BinopExpr, ctx: Ctx) -> CheckedType:
    if expr.operator == Binop.FLOOR_DIVIDE:
        return _floor_division(expr, ctx)
    if expr.operator in _BOOLEAN_OPERATORS:
        if ctx.check:
            for operand in [expr.lhs, expr.rhs]:
                failure = expect(ctx, operand, BoolType())
                if failure is not None:
                    return failure
        return BoolType()
    return _numeric_operation(expr, ctx)


def _numeric_operation(expr: BinopExpr, ctx: Ctx) -> CheckedType:
    # the operands are numbers (a bool counts as 0/1); they are checked also without ctx.check
    operand_types: List[Type] = []
    for operand in [expr.lhs, expr.rhs]:
        typ = type_of(operand, ctx)
        if isinstance(typ, CheckFail):
            return typ
        if not is_numeric(typ):
            return CheckFail.expected_numeric_got(operand, typ)
        operand_types.append(typ)
    if expr.operator in _COMPARISONS:
        return BoolType()
    if expr.operator in _ARITHMETIC:
        return join(NatType(bounds=None), *operand_types)
    raise Exception("unreachable")


def _floor_division(expr: BinopExpr, ctx: Ctx) -> CheckedType:
    """
    ``e // q`` rounds down a quotient of numbers; the result is natural. A literal divisor 0 is rejected (the
    division is undefined); any other divisor and a real dividend or divisor are accepted here, and the analysis
    decides what it can compute (syntax and semantics are separate, #356, #372).
    """
    divisor = expr.rhs
    if isinstance(divisor, (NatLitExpr, RealLitExpr)) and divisor.value == 0:
        return CheckFail(divisor, "Floor division by 0 is undefined.")
    divisor_type = type_of(divisor, ctx)
    if isinstance(divisor_type, CheckFail):
        return divisor_type
    typ = type_of(expr.lhs, ctx)
    if isinstance(typ, CheckFail):
        return typ
    for operand, operand_type in ((divisor, divisor_type), (expr.lhs, typ)):
        if not isinstance(operand_type, (NatType, BoolType, RealType)):
            return CheckFail(operand, f"Floor division '//' expects numbers, got {operand_type}.")
    return NatType(bounds=None)


@type_of.register
def _categorical(expr: CategoricalExpr, ctx: Ctx) -> CheckedType:
    # the widest type of the values; with ctx.check, all values must have comparable types
    typ = type_of(expr.exprs[0][0], ctx)
    if isinstance(typ, CheckFail):
        return typ
    if ctx.check:
        for other_expr, _other_prob in expr.exprs[1:]:
            other_typ = type_of(other_expr, ctx)
            if isinstance(other_typ, CheckFail):
                return other_typ
            if is_compatible(other_typ, typ):
                typ = other_typ
            elif not is_compatible(typ, other_typ):
                return CheckFail.expected_type_got(expr, typ, other_typ)
    return typ


@type_of.register
def _call(expr: FunctionCallExpr, ctx: Ctx) -> CheckedType:
    if ctx.check:
        failure = check_function_call(ctx, expr)
        if failure is not None:
            return failure
    return NatType(bounds=None)


@type_of.register
def _infer(expr: InferExpr, ctx: Ctx) -> CheckedType:
    if ctx.check:
        failure = check_function_call(ctx, expr.function)
        if failure is not None:
            return failure
    return DistributionType()


@type_of.register
def _sample(expr: SampleExpr, ctx: Ctx) -> CheckedType:
    if ctx.check:
        if expr.dist not in ctx.program.variables:
            return CheckFail(expr, f"Unknown variable: {expr.dist}")
        typ = ctx.program.variables[expr.dist]
        if not isinstance(typ, DistributionType):
            return CheckFail.expected_type_got(expr, DistributionType(), typ)
    return NatType(None)


@type_of.register
def _dist_probability(expr: ProbabilityExpr, ctx: Ctx) -> CheckedType:
    """``Pr[d == k]`` (also ``<``, ``<=``, ``>``, ``>=``, and ``k == d``) for a ``dist``
    variable ``d`` and a natural number ``k`` is a real number."""
    program, cond = ctx.program, expr.cond
    if not mentions_dist(program, cond):
        return CheckFail(expr, f"{spell_node(expr)} is not about a value of type dist. {INFER_HINT}")
    if not (isinstance(cond, BinopExpr) and cond.operator in DIST_COMPARISONS
            and (is_dist_variable(program, cond.lhs) or is_dist_variable(program, cond.rhs))):
        return CheckFail(
            expr, f"Inside {spell_node(expr)}, compare a variable d of type dist with a natural number, "
            "e.g. Pr[d == 3] or Pr[d < 3].")
    value = cond.rhs if is_dist_variable(program, cond.lhs) else cond.lhs
    if mentions_dist(program, value):
        return CheckFail(expr, f"Inside {spell_node(expr)}, the value compared with the dist variable must be "
                         "a natural number.")
    failure = expect(ctx.checked(), value, NatType(None))
    if failure is not None:
        return failure
    return RealType()


@type_of.register
def _dist_expectation(expr: ExpectationExpr, ctx: Ctx) -> CheckedType:
    """``Ex[d]`` for a ``dist`` variable ``d`` is a real number (maybe ∞)."""
    if is_dist_variable(ctx.program, expr.expr):
        return RealType()
    if mentions_dist(ctx.program, expr.expr):
        return CheckFail(expr, f"{spell_node(expr)}: Ex[…] inside a program takes a single variable of type "
                         "dist, e.g. Ex[d].")
    return CheckFail(expr, f"{spell_node(expr)} is not about a value of type dist. {INFER_HINT}")
