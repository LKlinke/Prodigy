from __future__ import annotations

import copy
import dataclasses
import logging
import re
from typing import get_args, Callable, List, Union, Sequence

from prodigy.pgcl import (Instr, AsgnInstr, CategoricalExpr, FunctionCallExpr, Expr, BinopExpr, Binop,
                          BoolLitExpr, IfInstr, NatType, Unop, UnopExpr, VarExpr)
from prodigy.pgcl.draws import Helpers, lower_draws

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import ForwardAnalysisError
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.function_handler import FunctionHandler
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


class AssignmentHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]], prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        instruction = _assume(instruction, AsgnInstr, 'AssignmentHandler')

        taken = set(prog_info.variables) | set(prog_info.parameters) | set(prog_info.functions)
        names: List[str] = []

        def fresh() -> str:
            name = distribution.get_fresh_variable(exclude=taken | set(names))
            names.append(name)
            return name

        helpers, rhs = lower_draws(instruction.rhs, set(prog_info.functions), fresh)
        if helpers:
            return _with_helpers(AsgnInstr(instruction.lhs, rhs), helpers, prog_info, distribution,
                                 error_prob, config, analyzer)

        if _is_boolean_operation(instruction.rhs):
            # "b = cond" is "if (cond) { b = true } else { b = false }" (#319)
            logger.info("Lower %s to a case split", instruction)
            return analyzer(IfInstr(cond=instruction.rhs,
                                    true=[AsgnInstr(instruction.lhs, BoolLitExpr(True))],
                                    false=[AsgnInstr(instruction.lhs, BoolLitExpr(False))]),
                            prog_info, distribution, error_prob, config)

        if isinstance(instruction.rhs, CategoricalExpr):
            return _categorical(instruction.lhs, instruction.rhs, prog_info, distribution, error_prob,
                                config, analyzer)

        if isinstance(instruction.rhs, FunctionCallExpr):
            return FunctionHandler.compute(instruction, prog_info,
                                           distribution, error_prob, config, analyzer)

        if not isinstance(instruction.rhs, get_args(Expr)):
            raise SyntaxError(
                f"Assignment {instruction} is ill-formed. right-hand-side must be an expression."
            )
        logger.info("Update %s", instruction)
        return distribution.update(
            BinopExpr(operator=Binop.EQ,
                      lhs=VarExpr(instruction.lhs),
                      rhs=instruction.rhs)), \
            error_prob


def _is_boolean_operation(expr: object) -> bool:
    """Is ``expr`` a comparison or a boolean operation (``not``, ``and``, ``or``), not a literal?"""
    if isinstance(expr, UnopExpr):
        return expr.operator == Unop.NEG
    return isinstance(expr, BinopExpr) and expr.operator.returns_boolean()


def _with_helpers(
        instruction: AsgnInstr, helpers: Helpers, prog_info: ProgramInfo, distribution: Distribution,
        error_prob: Distribution, config: ForwardAnalysisConfig,
        analyzer: Callable[
            [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
            tuple[Distribution, Distribution]
        ]
) -> tuple[Distribution, Distribution]:
    """
    An assignment with draws inside its right-hand side (see :mod:`prodigy.pgcl.draws`): the draws
    are assigned to their fresh helper variables first, then ``instruction`` (the right-hand side
    with the helpers in place of the draws) runs, and the helpers are marginalised. A message
    about a helper names its draw instead.
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.nested_inference import without
    from prodigy.analysis.session import current_session
    logger.info("Lower the draws of %s into %s", instruction, ", ".join(name for name, _ in helpers))
    names = [name for name, _ in helpers]
    program = copy.copy(prog_info.program)
    program.variables = {**program.variables, **{name: NatType(None) for name in names}}
    info = dataclasses.replace(prog_info, program=program)
    missing = [name for name in names if name not in distribution.get_variables()]
    distribution = distribution.set_variables(*distribution.get_variables(), *missing)
    try:
        for name, draw in helpers:
            distribution, error_prob = analyzer(AsgnInstr(name, draw), info, distribution, error_prob,
                                                config)
        distribution, error_prob = analyzer(instruction, info, distribution, error_prob, config)
    except (NotImplementedError, ValueError, ArithmeticError, ForwardAnalysisError) as exc:
        if len(exc.args) == 1 and isinstance(exc.args[0], str):
            message = exc.args[0]
            for name, draw in helpers:
                message = _replace_name(message, name, current_session().spell(draw))
            exc.args = (message,)
        raise
    return without(distribution, names), without(error_prob, names)


def _replace_name(message: str, name: str, text: str) -> str:
    """``message`` with the name ``name`` (as a whole word) replaced by ``text``."""
    return re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", lambda _: text, message)


def _categorical(
        var: str, expr: CategoricalExpr, prog_info: ProgramInfo, distribution: Distribution,
        error_prob: Distribution, config: ForwardAnalysisConfig,
        analyzer: Callable[
            [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
            tuple[Distribution, Distribution]
        ]
) -> tuple[Distribution, Distribution]:
    """
    ``var = categorical({v1: p1, …})`` is the choice between the assignments ``var = v_i`` with the
    probabilities ``p_i`` (literals that sum to 1, checked by the parser). Every assignment runs on
    the whole distribution and is weighted with its probability, as in a nested probabilistic
    choice; for literal values this is the distribution with ``var`` marginalised out times
    ``sum p_i * var^v_i``. The values are evaluated in the current state, so they may depend on
    the program variables.
    """
    logger.info("Update %s = %s", var, expr)
    zero = error_prob * "0"
    result = distribution * "0"
    errors = error_prob
    for value, prob in expr.exprs:
        probability = prob.to_fraction()
        if probability == 0:
            continue
        branch, branch_error = analyzer(AsgnInstr(var, value), prog_info, distribution, zero, config)
        result += branch * str(probability)
        errors += branch_error * str(probability)
    return result, errors
