import functools
from typing import Dict, List, Tuple

from probably.pgcl.ast import (Binop, BinopExpr, BoolLitExpr, Expr, IfInstr,
                               Instr, NatLitExpr, SkipInstr, VarExpr)


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
