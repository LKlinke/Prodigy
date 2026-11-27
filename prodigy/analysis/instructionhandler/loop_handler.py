from __future__ import annotations

import logging

from prodigy.language import LoopInstr

from prodigy.analysis.context import Frame, ProgramState
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: LoopInstr, state: ProgramState) -> ProgramState:
    """``loop (n) {B}``: B runs ``n`` times."""
    for i in range(instruction.iterations.value):
        logger.info("Computing iteration %d out of %d", i + 1, instruction.iterations.value)
        state = frame.run(instruction.body, state)
    return state
