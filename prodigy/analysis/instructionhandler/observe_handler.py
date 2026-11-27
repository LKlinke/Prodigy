from __future__ import annotations

import logging

from prodigy.language import ObserveInstr

from prodigy.analysis.context import Frame, ProgramState
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: ObserveInstr, state: ProgramState) -> ProgramState:
    """``observe(c)``: the part that violates c leaves the distribution and is added to the violation mass (as a
    distribution over the second-order variables, if the program has some)."""
    sat_part = state.dist.filter(instruction.cond)
    unsat_part = state.dist - sat_part
    so_vars = frame.program.so_vars
    if so_vars is None or len(so_vars) == 0:
        unsat_prob: Distribution | str = unsat_part.get_probability_mass()
    else:
        unsat_prob = unsat_part.marginal(*so_vars)
    return ProgramState(sat_part, state.error + unsat_prob)
