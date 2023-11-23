from __future__ import annotations

import functools
import logging
from typing import Dict, List, Tuple

from probably.pgcl import Program, WhileInstr
from probably.pgcl.ast import (Binop, BinopExpr, BoolLitExpr, Expr, IfInstr,
                               Instr, NatLitExpr, SkipInstr, VarExpr)

from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG)


def make_else_if(instructions: List[Tuple[BinopExpr, List[Instr]]]) -> IfInstr:
    """Creates an else-if chain from a list of conditions and bodies for if-instructions"""

    outer_instr: IfInstr = None
    curr_instr: IfInstr = None
    for cond, body in instructions:
        instr = IfInstr(cond=cond, true=body, false=None)
        if outer_instr is None:
            outer_instr = instr
        if curr_instr is not None:
            curr_instr.false = [instr]
        curr_instr = instr

    curr_instr.false = [SkipInstr()]
    return outer_instr


def state_to_equality_expression(state: Dict[str, int]) -> BinopExpr:
    equalities: List[Expr] = []
    for var, val in state.items():
        equalities.append(
            BinopExpr(Binop.EQ, lhs=VarExpr(var), rhs=NatLitExpr(val)))
    return functools.reduce(
        lambda expr1, expr2: BinopExpr(Binop.AND, expr1, expr2), equalities,
        BoolLitExpr(value=True))


def cav_phi(program: Program, invariant: Program) -> Program:
    """
        The characteristic loop functional. It unrolls a loop exactly once.

        .. returns: A new program object equivalent to one loop unrolling of :param: program.
    """
    assert isinstance(
        program.instructions[0],
        WhileInstr), "Program can only be one big loop to analyze."
    logger.debug("Create modified invariant program.")
    new_instructions = program.instructions[0].body.copy()

    for instr in invariant.instructions:
        new_instructions.append(instr)

    guarded_instr = IfInstr(cond=program.instructions[0].cond,
                            true=new_instructions,
                            false=[SkipInstr()])

    return Program(declarations=invariant.declarations,
                   variables=invariant.variables,
                   constants=invariant.constants,
                   parameters=invariant.parameters,
                   instructions=[guarded_instr],
                   functions=invariant.functions)
