"""
While loops.

The handler asks the frame's :class:`~prodigy.analysis.context.AnalysisContext` which
:mod:`loop strategy <prodigy.analysis.loops.api>` to use for a loop and runs it. The strategies live in
:mod:`prodigy.analysis.loops` and are registered by :mod:`prodigy.analysis.loops.builtin`; in the interactive CLI
they show up in the same menu, with the same prompts, as before.
"""
from __future__ import annotations

from prodigy.language import WhileInstr

from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.loops.api import LoopTask
from prodigy.analysis.loops.registry import get_loop_strategy


def handle(frame: Frame, instruction: WhileInstr, state: ProgramState) -> ProgramState:
    """Asks the frame's context which strategy analyses this occurrence of the loop and runs it. The
    context is the current one while the strategy runs, for the checks it calls that look it up at their
    boundary (:func:`~prodigy.analysis.context.boundary_context`)."""
    session = frame.context
    with session.activate():
        loop = session.loops.lookup(instruction)
        decision = session.decide_loop(loop, instruction)
        strategy = get_loop_strategy(decision.strategy)
        task = LoopTask(instruction=instruction, frame=frame, state=state, loop=loop, decision=decision,
                        strategy=strategy)
        outcome = session.run_loop(task)
    return ProgramState(outcome.distribution, outcome.error_prob)
