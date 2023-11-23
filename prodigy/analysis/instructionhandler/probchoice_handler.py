from __future__ import annotations

import logging
from typing import Callable, Union, Sequence

from probably.pgcl import Instr, ChoiceInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG)


class PChoiceHandler(InstructionHandler):
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
        _assume(instruction, ChoiceInstr, 'PChoiceHandlerGF')

        lhs_block, lhs_error_prob = analyzer(instruction.lhs, prog_info, distribution, error_prob, config)
        rhs_block, rhs_error_prob = analyzer(instruction.rhs, prog_info, distribution, error_prob, config)
        logger.info("Combining PChoice branches.\n%s", instruction)
        new_prob = lhs_error_prob * str(instruction.prob) + rhs_error_prob * f"1-({instruction.prob})"
        res_error_prob = (
                lhs_error_prob * str(instruction.prob) +
                rhs_error_prob * f"1-({instruction.prob})").set_variables(
            *error_prob.get_variables())
        return lhs_block * str(
            instruction.prob) + rhs_block * f"1-({instruction.prob})", \
            res_error_prob
