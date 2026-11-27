# pylint: disable=line-too-long  # doctest output as taken over from probably
"""
The check of instructions: :func:`check_instr`, with one handler of
:func:`~prodigy.language.check.context.check_node` per instruction class.
"""
from __future__ import annotations

from typing import List, Mapping, Optional, Union

from prodigy.language.ast import (AbortInstr, AsgnInstr, BoolType, CategoricalExpr, ChoiceInstr, CovarianceInstr,
                                  ExpectationInstr, Expr, FunctionCallExpr, IfInstr, Instr, LocalDeclInstr,
                                  LoopInstr, NatLitExpr, NatType, ObserveInstr, OptimizationQuery, PlotInstr,
                                  PrintInstr, ProbabilityQueryInstr, Program, QueryInstr, RealType, SkipInstr,
                                  Var, VarDecl, VarExpr, VarianceInstr, WhileInstr)

from .calls import sample_predefined_functions
from .compatibility import is_compatible, is_numeric
from .context import CheckFail, Ctx, check_block, check_node, expect, spell_node, type_of
from .dist import check_query_without_dist


def check_instr(
    program: Program,
    instr: Instr,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """
    Check a single instruction for type-safety.

    .. doctest::

        >>> from prodigy.language.parser import parse_program

        >>> program = parse_program("x: bool; x = true;")
        >>> check_instr(program, program.instructions[0])

        >>> program = parse_program("x: nat; x = true;")
        >>> check_instr(program, program.instructions[0])

        >>> program = parse_program("x: bool; x = bernoulli(1/2);")
        >>> check_instr(program, program.instructions[0])

        >>> program = parse_program("x: bool; x = 2;")
        >>> check_instr(program, program.instructions[0])
        CheckFail(location=..., message='Expected value of type BoolType(), got NatType(bounds=None) that is not provably 0 or 1.')

        >>> program = parse_program("x: nat; x = 1/2;")
        >>> check_instr(program, program.instructions[0])
        CheckFail(location=..., message='Expected value of type NatType(bounds=None), got RealType().')

        >>> program = parse_program("const x: bool = true; x = true;")
        >>> check_instr(program, program.instructions[0])
        CheckFail(location=..., message='x is not a variable.')

        >>> program = parse_program("x: nat[1, 5]; x = 3;")
        >>> check_instr(program, program.instructions[0])

        >>> program = parse_program("x: nat[1, 5]; y: nat; x = x + 3;")
        >>> check_instr(program, program.instructions[0])
    """
    return check_node(instr, Ctx(program, predefined_functions))


@check_node.register(SkipInstr)
@check_node.register(AbortInstr)
@check_node.register(PrintInstr)
def _nothing_to_check(_instr: Union[SkipInstr, AbortInstr, PrintInstr], _ctx: Ctx) -> Optional[CheckFail]:
    return None


@check_node.register
def _while(instr: WhileInstr, ctx: Ctx) -> Optional[CheckFail]:
    return expect(ctx, instr.cond, BoolType()) or check_block(ctx, instr.body)


@check_node.register
def _if(instr: IfInstr, ctx: Ctx) -> Optional[CheckFail]:
    return expect(ctx, instr.cond, BoolType()) or check_block(ctx, instr.true, instr.false)


@check_node.register
def _choice(instr: ChoiceInstr, ctx: Ctx) -> Optional[CheckFail]:
    return expect(ctx, instr.prob, RealType()) or check_block(ctx, instr.lhs, instr.rhs)


@check_node.register
def _loop(instr: LoopInstr, ctx: Ctx) -> Optional[CheckFail]:
    return expect(ctx, instr.iterations, NatType(None)) or check_block(ctx, instr.body)


@check_node.register
def _query_block(instr: QueryInstr, ctx: Ctx) -> Optional[CheckFail]:
    return check_block(ctx, instr.instrs)


@check_node.register
def _observe(instr: ObserveInstr, ctx: Ctx) -> Optional[CheckFail]:
    return expect(ctx, instr.cond, BoolType())


@check_node.register
def _assignment(instr: AsgnInstr, ctx: Ctx) -> Optional[CheckFail]:
    lhs_type = ctx.program.variables.get(instr.lhs)
    if lhs_type is None:
        return CheckFail(instr, f'{instr.lhs} is not a variable.')
    rhs_type = type_of(instr.rhs, ctx)
    if isinstance(rhs_type, CheckFail):
        return rhs_type
    failure = _check_sample_pgf_variable(instr, ctx)
    if failure is not None or is_compatible(lhs_type, rhs_type):
        return failure
    # narrowing to bool is fine if the value is provably 0 or 1
    if isinstance(lhs_type, BoolType) and is_numeric(rhs_type):
        if _is_zero_or_one(ctx.unchecked(), instr.rhs):
            return None
        return CheckFail(
            instr,
            f'Expected value of type {lhs_type}, got {rhs_type} that is not provably 0 or 1.'
        )
    return CheckFail.expected_type_got(instr, lhs_type, rhs_type)


def _check_sample_pgf_variable(instr: AsgnInstr, ctx: Ctx) -> Optional[CheckFail]:
    """``x = sample_pgf(G, y)`` samples ``G`` in ``y``; only ``y = x`` is supported."""
    rhs = instr.rhs
    if isinstance(rhs, FunctionCallExpr) and rhs.function == 'sample_pgf' \
            and rhs.function not in ctx.program.functions \
            and rhs.params[0][1] != VarExpr(instr.lhs):
        return CheckFail(
            instr, f"{spell_node(rhs)} samples a generating function in "
            f"{spell_node(rhs.params[0][1])}, but only sampling in the assigned variable "
            f"{instr.lhs} is supported.")
    return None


_ZERO_ONE_DISTRIBUTIONS = {'bernoulli'}
_RANGE_DISTRIBUTIONS = {'unif', 'unif_d'}


def _is_zero_or_one(ctx: Ctx, expr: Expr) -> bool:
    """
    Whether every value of ``expr`` is provably 0 or 1, so it may be assigned to a ``bool``
    variable. This is a cheap syntactic check: the literals 0 and 1, boolean expressions,
    ``bernoulli(p)``, uniform distributions with bounds in {0, 1}, and categorical choices
    between such values. ``ctx`` asks only for types (``check`` off).

    .. doctest::

        >>> from prodigy.language.ast import Program
        >>> from prodigy.language.ppy_parser import parse_ppy_expr
        >>> ctx = Ctx(Program(list(), dict(), dict(), dict(), dict(), list()), sample_predefined_functions, False)
        >>> _is_zero_or_one(ctx, parse_ppy_expr("bernoulli(1/3)"))
        True
        >>> _is_zero_or_one(ctx, parse_ppy_expr("unif(0, 2)"))
        False
    """
    if isinstance(expr, NatLitExpr):
        return expr.value in (0, 1)
    if isinstance(expr, CategoricalExpr):
        return all(
            _is_zero_or_one(ctx, value) for value, _prob in expr.exprs)
    if isinstance(expr, FunctionCallExpr) and expr.function not in ctx.program.functions:
        if expr.function in _ZERO_ONE_DISTRIBUTIONS:
            return True
        if expr.function in _RANGE_DISTRIBUTIONS:
            bounds = list(expr.params[0]) + list(expr.params[1].values())
            return len(bounds) == 2 and all(
                _is_zero_or_one(ctx, bound) for bound in bounds)
        return False
    return isinstance(type_of(expr, ctx), BoolType)


@check_node.register
def _probability_query(instr: ProbabilityQueryInstr, ctx: Ctx) -> Optional[CheckFail]:
    failure = check_query_without_dist(ctx.program, instr, [instr.expr])
    if failure is not None:
        return failure
    cond_type = type_of(instr.expr, ctx)
    if isinstance(cond_type, CheckFail):
        return cond_type
    # Pr[x] (a marginal) or Pr[condition]
    if not (is_compatible(BoolType(), cond_type) or is_compatible(NatType(None), cond_type)):
        return CheckFail.expected_type_got(instr.expr, BoolType(), cond_type)
    return None


@check_node.register(ExpectationInstr)
@check_node.register(VarianceInstr)
@check_node.register(CovarianceInstr)
def _moment_query(instr: Union[ExpectationInstr, VarianceInstr, CovarianceInstr],
                  ctx: Ctx) -> Optional[CheckFail]:
    operands = [instr.lhs, instr.rhs] if isinstance(instr, CovarianceInstr) else [instr.expr]
    failure = check_query_without_dist(ctx.program, instr, operands)
    if failure is not None:
        return failure
    for operand in operands:
        expr_type = type_of(operand, ctx)
        if isinstance(expr_type, CheckFail):
            return expr_type
    return None


@check_node.register
def _optimization_query(instr: OptimizationQuery, ctx: Ctx) -> Optional[CheckFail]:
    failure = check_query_without_dist(ctx.program, instr, [instr.expr])
    if failure is not None:
        return failure
    expr_type = type_of(instr.expr, ctx)
    if isinstance(expr_type, CheckFail):
        return expr_type
    if instr.parameter not in ctx.program.parameters:
        return CheckFail(
            instr,
            f"In optimization queries, the variable can only be a program parameter, was {instr.parameter}."
        )
    return None


@check_node.register
def _plot(instr: PlotInstr, ctx: Ctx) -> Optional[CheckFail]:
    failure = check_query_without_dist(ctx.program, instr, [instr.var_1, instr.var_2])
    if failure is not None:
        return failure
    for var in (instr.var_1, instr.var_2):
        if var:
            var_type = type_of(var, ctx)
            if isinstance(var_type, CheckFail):
                return var_type
    if instr.prob:
        failure = expect(ctx, instr.prob, RealType())
        if failure is not None:
            return failure
    if instr.term_count:
        return expect(ctx, instr.term_count, NatType(None))
    return None


@check_node.register
def _local_declaration(instr: LocalDeclInstr, _ctx: Ctx) -> Optional[CheckFail]:
    # Block scope is lowered into plain assignments before type checking
    # (prodigy.language.scope.lower_block_scope, done by parse_program).
    return CheckFail(
        instr,
        f"The declaration of '{instr.decl.var}' inside a block has not been "
        f"lowered; parse the program with parse_program or compile_program.")
