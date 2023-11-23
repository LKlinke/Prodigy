from __future__ import annotations

import logging
from typing import Callable, Union, Sequence

from probably.pgcl import Instr, ObserveInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".")[-1], logging.DEBUG)


class ObserveHandler(InstructionHandler):
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
        _assume(instruction, ObserveInstr, 'ObserverHandler')

        sat_part = distribution.filter(instruction.cond)
        unsat_part = distribution - sat_part
        so_vars = prog_info.so_vars
        if so_vars is None or len(so_vars) == 0:
            unsat_prob: Distribution | str = unsat_part.get_probability_mass()
        else:
            unsat_prob = unsat_part.marginal(*so_vars)
        res_error_prob = error_prob + unsat_prob
        return sat_part, res_error_prob
