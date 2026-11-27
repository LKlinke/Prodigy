from __future__ import annotations

import logging
from prodigy.language import ChoiceInstr
from prodigy.language.engine_text import engine_text_of

from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.instructionhandler.counter_operators import (apply_operator, apply_ratio, counter_probability,
                                                                   ratio_probability)
from prodigy.analysis.instructionhandler.state_dependent import (checked_probability, instantiate,
                                                                 split_states, state_variables)
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: ChoiceInstr, state: ProgramState) -> ProgramState:
    """
    A probabilistic choice ``{lhs} [p] {rhs}``. If ``p`` does not depend on the program state,
    both branches run on the whole distribution and are weighted with ``p`` and ``1 - p``.
    Otherwise the distribution is split by the values of the program variables in ``p``
    (:mod:`~prodigy.analysis.instructionhandler.state_dependent`): the left branch runs on the
    sum of the parts weighted with the value of ``p`` in them, the right branch on the rest. The
    semantics is linear, so this is the same as running both branches in every state.
    """
    distribution = state.dist
    construct = f"The probability {instruction.prob} of the probabilistic choice"

    variables = state_variables([instruction.prob], distribution)
    if not variables:
        checked_probability(instruction.prob, construct)
        lhs = frame.run(instruction.lhs, state)
        rhs = frame.run(instruction.rhs, state)
        logger.info("Combining PChoice branches.\n%s", instruction)
        prob = engine_text_of(instruction.prob)
        return ProgramState(lhs.dist * prob + rhs.dist * f"1-({prob})", lhs.error * prob + rhs.error * f"1-({prob})")

    operator = counter_probability(instruction.prob, variables)
    ratio = ratio_probability(instruction.prob, variables)
    if operator is not None and not distribution.is_zero_dist() \
            and distribution.finite_support(operator.variable) is None:
        # infinitely many values of the counter: the probability acts as an operator (#323)
        logger.info("Applying the probability %s as an operator in %s", instruction.prob, operator.variable)
        left, right = apply_operator(operator, distribution, construct)
    elif ratio is not None and not distribution.is_zero_dist() \
            and distribution.finite_support(*sorted(variables)) is None:
        # x/n with infinitely many values of x and n: θ_x followed by 1/θ_n (#237)
        logger.info("Applying the probability %s as an operator in %s", instruction.prob, ratio.denominator)
        left, right = apply_ratio(ratio, distribution, construct)
    else:
        logger.info("Splitting the states for the probability %s", instruction.prob)
        left = distribution * "0"
        for part in split_states(distribution, variables, construct):
            probability = instantiate(instruction.prob, part.state)
            checked_probability(probability, construct, part)
            left += part.distribution * engine_text_of(probability)
        right = distribution - left
    zero = state.error * "0"
    lhs = frame.run(instruction.lhs, ProgramState(left, zero))
    rhs = frame.run(instruction.rhs, ProgramState(right, zero))
    logger.info("Combining PChoice branches.\n%s", instruction)
    return ProgramState(lhs.dist + rhs.dist, state.error + lhs.error + rhs.error)
