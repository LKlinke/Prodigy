from __future__ import annotations

import logging
from typing import get_args, Callable, Union, Sequence

from probably.pgcl import Instr, AsgnInstr, CategoricalExpr, FunctionCallExpr, Expr, BinopExpr, Binop, VarExpr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.function_handler import FunctionHandler
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG)


class AssignmentHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr, prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, AsgnInstr, 'AssignmentHandler')

        # rhs is a categorical expression (explicit finite distr)
        if isinstance(instruction.rhs, CategoricalExpr):
            raise NotImplementedError(
                "Categorical expression are currently not supported.")

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
