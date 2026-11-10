from __future__ import annotations

import logging
from typing import get_args, Callable, Union, Sequence

from prodigy.pgcl import Instr, AsgnInstr, CategoricalExpr, FunctionCallExpr, Expr, BinopExpr, Binop, VarExpr

from prodigy.analysis.config import ForwardAnalysisConfig
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
