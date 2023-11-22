from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable

from probably.pgcl import Instr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution


class InstructionHandler(ABC):
    """ Abstract class that defines a strategy for handling a specific program instruction. """

    @staticmethod
    @abstractmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]],
            prog_info: ProgramInfo,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        """Computes the updated distribution after executing the instruction `instruction` on input `distribution`"""
