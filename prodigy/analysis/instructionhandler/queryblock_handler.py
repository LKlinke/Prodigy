from __future__ import annotations

import logging
from prodigy.language import QueryInstr, AsgnInstr, ObserveInstr

from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.instructionhandler import assignment_handler
from prodigy.distribution import Distribution, MarginalType
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: QueryInstr, state: ProgramState) -> ProgramState:
    """A query block ``query { x = …; observe(…) }`` on a finite distribution without violation mass."""
    distribution, error_prob = state
    instrs = instruction.instrs
    assert len(instrs) == 2
    assert isinstance(instrs[0], AsgnInstr)
    assert isinstance(instrs[1], ObserveInstr)
    assert distribution.is_finite()
    assert error_prob.is_zero_dist(
    )  # this means the sum of all probabilities used below is 1 respectively

    dist = assignment_handler.handle(frame, instrs[0], state).dist
    marginal = dist.marginal(instrs[0].lhs)
    assert marginal.is_finite()

    # this assumes that the updated variable is independent from the rest of the distribution
    zero: Distribution = frame.config.factory.one(*dist.get_variables()) * '0'
    for rest_prob, rest_state in dist.marginal(
            instrs[0].lhs, method=MarginalType.EXCLUDE):
        if rest_prob == "0":
            continue
        new = zero
        for me_prob, me_state in marginal:
            if me_prob == "0":
                continue
            joint = rest_state.extend(me_state)
            if dist.evaluate_condition(instrs[1].cond, joint):
                new += dist.filter_state(joint)
        new /= str(new.get_probability_mass())
        new *= rest_prob
        dist -= dist.filter_state(rest_state)
        dist += new

    return ProgramState(dist, error_prob)
