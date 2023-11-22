from __future__ import annotations

import logging
from typing import Callable, Union, Sequence

from probably.pgcl import Instr, IfInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


class ITEHandler(InstructionHandler):
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
        _assume(instruction, IfInstr, 'ITEHandler')

        logger.info("Filtering the guard %s", instruction.cond)
        sat_part = distribution.filter(instruction.cond)
        non_sat_part = distribution - sat_part
        zero = error_prob * "0"
        if config.show_intermediate_steps:
            print(
                f"\n{Style.YELLOW}Filter:{Style.RESET} {instruction.cond} \t " \
                f"{Style.GREEN}Result:{Style.RESET} {sat_part}"
            )
            print(
                f"\n{Style.YELLOW} If-branch: ({instruction.cond}){Style.RESET}"
            )
            if_branch, if_error_prob = analyzer(
                instruction.true, prog_info, sat_part, zero, config)
            print(f"\n{Style.YELLOW} Else-branch:{Style.RESET}")
            else_branch, else_error_prob = analyzer(
                instruction.false, prog_info, non_sat_part, zero, config)
            print(f"\n{Style.YELLOW}Combined:{Style.RESET}")
        else:
            if_branch, if_error_prob = analyzer(
                instruction.true, prog_info, sat_part, zero, config)
            else_branch, else_error_prob = analyzer(
                instruction.false, prog_info, non_sat_part, zero, config)
        result = if_branch + else_branch
        res_error_prob = error_prob + if_error_prob + else_error_prob
        logger.info("Combining if-branches.\n%s", instruction)
        return result, res_error_prob
