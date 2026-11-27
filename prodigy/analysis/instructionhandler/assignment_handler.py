from __future__ import annotations

import copy
import dataclasses
import logging
import re
from typing import get_args, List

from prodigy.language import (AsgnInstr, CategoricalExpr, FunctionCallExpr, Expr, BinopExpr, Binop,
                          BoolLitExpr, IfInstr, NatType, Unop, UnopExpr, VarExpr)
from prodigy.language.draws import Helpers, lower_draws

from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.exceptions import ForwardAnalysisError
from prodigy.analysis.instructionhandler import function_handler
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: AsgnInstr, state: ProgramState) -> ProgramState:
    """An assignment ``x = e``: draws inside ``e`` are lowered first, a Boolean operation becomes a case split, a
    categorical choice and a function call have their own rules, and anything else updates the distribution."""
    prog_info = frame.program
    taken = set(prog_info.variables) | set(prog_info.parameters) | set(prog_info.functions)
    names: List[str] = []

    def fresh() -> str:
        name = state.dist.get_fresh_variable(exclude=taken | set(names))
        names.append(name)
        return name

    helpers, rhs = lower_draws(instruction.rhs, set(prog_info.functions), fresh)
    if helpers:
        return _with_helpers(frame, AsgnInstr(instruction.lhs, rhs), helpers, state)

    if _is_boolean_operation(instruction.rhs):
        # "b = cond" is "if (cond) { b = true } else { b = false }" (#319)
        logger.info("Lower %s to a case split", instruction)
        return frame.run(IfInstr(cond=instruction.rhs,
                                 true=[AsgnInstr(instruction.lhs, BoolLitExpr(True))],
                                 false=[AsgnInstr(instruction.lhs, BoolLitExpr(False))]), state)

    if isinstance(instruction.rhs, CategoricalExpr):
        return _categorical(frame, instruction.lhs, instruction.rhs, state)

    if isinstance(instruction.rhs, FunctionCallExpr):
        return function_handler.handle(frame, instruction, state)

    if not isinstance(instruction.rhs, get_args(Expr)):
        raise SyntaxError(
            f"Assignment {instruction} is ill-formed. right-hand-side must be an expression."
        )
    logger.info("Update %s", instruction)
    return ProgramState(state.dist.update(BinopExpr(operator=Binop.EQ, lhs=VarExpr(instruction.lhs),
                                                    rhs=instruction.rhs)), state.error)


def _is_boolean_operation(expr: object) -> bool:
    """Is ``expr`` a comparison or a boolean operation (``not``, ``and``, ``or``), not a literal?"""
    if isinstance(expr, UnopExpr):
        return expr.operator == Unop.NEG
    return isinstance(expr, BinopExpr) and expr.operator.returns_boolean()


def _with_helpers(frame: Frame, instruction: AsgnInstr, helpers: Helpers, state: ProgramState) -> ProgramState:
    """
    An assignment with draws inside its right-hand side (see :mod:`prodigy.language.draws`): the draws
    are assigned to their fresh helper variables first, then ``instruction`` (the right-hand side
    with the helpers in place of the draws) runs, and the helpers are marginalised. A message
    about a helper names its draw instead.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.nested_inference import without
    logger.info("Lower the draws of %s into %s", instruction, ", ".join(name for name, _ in helpers))
    names = [name for name, _ in helpers]
    program = copy.copy(frame.program.program)
    program.variables = {**program.variables, **{name: NatType(None) for name in names}}
    inner = frame.for_program(dataclasses.replace(frame.program, program=program))
    distribution = state.dist
    missing = [name for name in names if name not in distribution.get_variables()]
    state = ProgramState(distribution.set_variables(*distribution.get_variables(), *missing), state.error)
    try:
        for name, draw in helpers:
            state = inner.run(AsgnInstr(name, draw), state)
        state = inner.run(instruction, state)
    except (NotImplementedError, ValueError, ArithmeticError, ForwardAnalysisError) as exc:
        if len(exc.args) == 1 and isinstance(exc.args[0], str):
            message = exc.args[0]
            for name, draw in helpers:
                message = _replace_name(message, name, frame.context.spell(draw))
            exc.args = (message,)
        raise
    return ProgramState(without(state.dist, names), without(state.error, names))


def _replace_name(message: str, name: str, text: str) -> str:
    """``message`` with the name ``name`` (as a whole word) replaced by ``text``."""
    return re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", lambda _: text, message)


def _categorical(frame: Frame, var: str, expr: CategoricalExpr, state: ProgramState) -> ProgramState:
    """
    ``var = categorical({v1: p1, …})`` is the choice between the assignments ``var = v_i`` with the
    probabilities ``p_i`` (literals that sum to 1, checked by the parser). Every assignment runs on
    the whole distribution and is weighted with its probability, as in a nested probabilistic
    choice; for literal values this is the distribution with ``var`` marginalised out times
    ``sum p_i * var^v_i``. The values are evaluated in the current state, so they may depend on
    the program variables.
    """
    logger.info("Update %s = %s", var, expr)
    zero = state.error * "0"
    result = state.dist * "0"
    errors = state.error
    for value, prob in expr.exprs:
        probability = prob.to_fraction()
        if probability == 0:
            continue
        branch = frame.run(AsgnInstr(var, value), ProgramState(state.dist, zero))
        result += branch.dist * str(probability)
        errors += branch.error * str(probability)
    return ProgramState(result, errors)
