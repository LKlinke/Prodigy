from __future__ import annotations

import logging

from prodigy.language import IfInstr

from prodigy.analysis.context import Frame, ProgramState
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: IfInstr, state: ProgramState) -> ProgramState:
    """``if (c) {A} else {B}``: A runs on the part that satisfies c, B on the rest, and the results add up."""
    logger.info("Filtering the guard %s", instruction.cond)
    sat_part = state.dist.filter(instruction.cond)
    non_sat_part = state.dist - sat_part
    zero = state.error * "0"
    if_branch = frame.run(instruction.true, ProgramState(sat_part, zero))
    else_branch = frame.run(instruction.false, ProgramState(non_sat_part, zero))
    logger.info("Combining if-branches.\n%s", instruction)
    return ProgramState(if_branch.dist + else_branch.dist, state.error + if_branch.error + else_branch.error)
