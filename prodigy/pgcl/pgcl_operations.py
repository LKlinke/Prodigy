from __future__ import annotations

import copy
import functools
import logging
from typing import Dict, List, Optional, Tuple

from prodigy.pgcl import Program, WhileInstr
from prodigy.pgcl.ast import (Binop, BinopExpr, BoolLitExpr, Expr, IfInstr,
                               Instr, NatLitExpr, SkipInstr, VarExpr)

from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def make_else_if(
        instructions: List[Tuple[BinopExpr, List[Instr]]]) -> Optional[IfInstr]:
    """
    Creates an else-if chain from a list of conditions and bodies for if-instructions.
    Returns ``None`` for an empty list.
    """

    outer_instr: IfInstr | None = None
    curr_instr: IfInstr | None = None
    for cond, body in instructions:
        instr = IfInstr(cond=cond, true=body, false=[])
        if outer_instr is None:
            outer_instr = instr
        if curr_instr is not None:
            curr_instr.false = [instr]
        curr_instr = instr
    if curr_instr:
        curr_instr.false = [SkipInstr()]
    return outer_instr


def state_to_equality_expression(state: Dict[str, int]) -> Expr:
    equalities: List[Expr] = []
    for var, val in state.items():
        equalities.append(
            BinopExpr(Binop.EQ, lhs=VarExpr(var), rhs=NatLitExpr(val)))

    def conjoin(expr1: Expr, expr2: Expr) -> Expr:
        return BinopExpr(Binop.AND, expr1, expr2)

    true: Expr = BoolLitExpr(value=True)
    return functools.reduce(conjoin, equalities, true)


def cav_phi(program: Program, invariant: Program) -> Program:
    """
        The characteristic loop functional. It unrolls a loop exactly once.

        The result is ``if (G) { B; I } else { skip }`` for a program ``while (G) { B }`` and an
        invariant program ``I``. Declarations, constants, parameters and functions are taken from
        the invariant. The instructions are copied, so the result shares no nodes with its inputs.

        .. returns: A new program object equivalent to one loop unrolling of :param: program.
        .. raises: ValueError if :param: program is not a single while loop.
    """
    if len(program.instructions) != 1 or not isinstance(program.instructions[0], WhileInstr):
        raise ValueError("Program can only be one big loop to analyze.")
    loop = program.instructions[0]
    logger.debug("Create modified invariant program.")
    new_instructions = copy.deepcopy(loop.body) + copy.deepcopy(invariant.instructions)

    guarded_instr = IfInstr(cond=copy.deepcopy(loop.cond),
                            true=new_instructions,
                            false=[SkipInstr()])

    return Program(declarations=invariant.declarations,
                   variables=invariant.variables,
                   constants=invariant.constants,
                   parameters=invariant.parameters,
                   instructions=[guarded_instr],
                   functions=invariant.functions)
