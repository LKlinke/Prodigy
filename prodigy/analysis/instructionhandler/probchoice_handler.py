from __future__ import annotations

import logging
from typing import Callable, Union, Sequence

from prodigy.pgcl import Instr, ChoiceInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.instructionhandler.state_dependent import (checked_probability, instantiate,
                                                                 split_states, state_variables)
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


class PChoiceHandler(InstructionHandler):
    """
    A probabilistic choice ``{lhs} [p] {rhs}``. If ``p`` does not depend on the program state,
    both branches run on the whole distribution and are weighted with ``p`` and ``1 - p``.
    Otherwise the distribution is split by the values of the program variables in ``p``
    (:mod:`~prodigy.analysis.instructionhandler.state_dependent`): the left branch runs on the
    sum of the parts weighted with the value of ``p`` in them, the right branch on the rest. The
    semantics is linear, so this is the same as running both branches in every state.
    """
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
        instruction = _assume(instruction, ChoiceInstr, 'PChoiceHandlerGF')
        construct = f"The probability {instruction.prob} of the probabilistic choice"

        variables = state_variables([instruction.prob], distribution)
        if not variables:
            checked_probability(instruction.prob, construct)
            lhs_block, lhs_error_prob = analyzer(instruction.lhs, prog_info, distribution, error_prob, config)
            rhs_block, rhs_error_prob = analyzer(instruction.rhs, prog_info, distribution, error_prob, config)
            logger.info("Combining PChoice branches.\n%s", instruction)
            new_prob = lhs_error_prob * str(instruction.prob) + rhs_error_prob * f"1-({instruction.prob})"
            return lhs_block * str(
                instruction.prob) + rhs_block * f"1-({instruction.prob})", new_prob

        logger.info("Splitting the states for the probability %s", instruction.prob)
        left = distribution * "0"
        for part in split_states(distribution, variables, construct):
            probability = instantiate(instruction.prob, part.state)
            checked_probability(probability, construct, part)
            left += part.distribution * str(probability)
        right = distribution - left
        zero = error_prob * "0"
        lhs_block, lhs_error_prob = analyzer(instruction.lhs, prog_info, left, zero, config)
        rhs_block, rhs_error_prob = analyzer(instruction.rhs, prog_info, right, zero, config)
        logger.info("Combining PChoice branches.\n%s", instruction)
        return lhs_block + rhs_block, error_prob + lhs_error_prob + rhs_error_prob
