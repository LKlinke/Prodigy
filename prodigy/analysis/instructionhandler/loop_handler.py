from __future__ import annotations

import logging
from typing import Callable, Union, Sequence

from probably.pgcl import Instr, Program, LoopInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


class LoopHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            prog_info: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, LoopInstr, 'LoopHandler')
        for i in range(instruction.iterations.value):
            logger.info("Computing iteration %d out of %d", i + 1,
                        instruction.iterations.value)
            distribution, error_prob = analyzer(instruction.body, prog_info, distribution, error_prob, config)
        return distribution, error_prob
