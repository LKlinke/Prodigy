"""
The strategy ``evt_iterate``: iterates of the EVT operator, for guessing an invariant
(:mod:`prodigy.analysis.evt_exploration`); the result is their terminating part, a lower bound.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Tuple

import sympy

from prodigy.analysis import evt_exploration
from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import register_loop_strategy
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup
from prodigy.analysis.loops.common import _exit_part, _mass, _require_unconditioned


logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def _strategy_evt_iterate(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    distribution, error_prob = task.distribution, task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations of EVT operator", max_iter)

    evt = distribution * 0
    mass_in = _mass(distribution)
    # iterates kept for the exploration (#324): H_N is no output distribution, but shows the expected visiting
    # times, from which a user (or a rational fit) guesses an EVT invariant
    samples = set(evt_exploration.sample_points(max_iter)) if task.session.collect_details else set()
    kept: List[Tuple[int, Distribution]] = []
    recent: List[Distribution] = []
    with task.session.sub_analysis("EVT iteration") as scope:
        for i in range(max_iter):
            logger.debug("Iteration %i", i)
            task.progress(i + 1, max_iter)
            evt = distribution + \
                  analyzer(instruction.body, prog_info, evt.filter(instruction.cond), error_prob, config)[0]
            # the terminating part of the i-th iterate, a lower bound unless inner loops were
            # over-approximated; the mass not accounted for is what has not left the loop yet
            task.iterate(i + 1, functools.partial(_exit_part, evt, instruction.cond),
                         residual=lambda exit_dist: mass_in - _mass(exit_dist),
                         kind=Bound.LOWER if scope.kind in (Bound.EXACT, Bound.LOWER) else Bound.UNKNOWN)
            if samples:
                recent = (recent + [evt])[-3:]
                if i + 1 in samples:
                    kept.append((i + 1, evt))
    result = evt - evt.filter(instruction.cond)
    remaining = sympy.simplify(mass_in - _mass(result))
    certificate: Dict[str, Any] = {"type": "evt_iteration", "iterations": max_iter, "remaining_mass": str(remaining),
                                   "evt": str(evt)}
    if kept:
        try:
            certificate["exploration"] = evt_exploration.exploration(
                kept, [str(v) for v in prog_info.program.variables.keys()], recent)
        except Exception as exc:  # pylint: disable=broad-except  # the exploration is optional
            logger.debug("EVT exploration failed: %s", exc)
    if scope.kind not in (Bound.EXACT, Bound.LOWER):
        return LoopOutcome(result, error_prob, kind=Bound.UNKNOWN, certificate=certificate,
                           messages=["Loops inside the body were over-approximated; no guarantee."])
    if remaining == 0:
        return LoopOutcome(result, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: all mass has left the loop within {max_iter} iterations."])
    return LoopOutcome(result, error_prob, kind=Bound.LOWER, residual=remaining, certificate=certificate,
                       messages=[f"Lower bound from {max_iter} iterations of the EVT operator; "
                                 f"mass {remaining} is not accounted for. The iterates H_N are expected "
                                 "visiting times, not the output distribution."])

register_loop_strategy(LoopStrategy(
    name="evt_iterate",
    title="Expected visiting times H_N",
    group="explore",
    outcome="exploration",
    summary="Shows the iterates H_N of the expected visiting times (not the output distribution) to help you "
            "guess an EVT invariant; you provide N; the result is only a lower bound.",
    legacy_key="4",
    soundness=Bound.LOWER,
    delivers=(Delivers(Bound.LOWER), Delivers(Bound.EXACT, "no mass is left inside the loop")),
    description="Iterates the expected-visiting-time operator Ψ_G(H) = G + ⟦body⟧(⟨H⟩_guard) N times "
                "from 0. The coefficient of a state in H_N is the expected number of times the loop head "
                "is reached in that state within N iterations; the GUI shows how they grow with N and a "
                "guessed EVT invariant to check with Check EVT invariant. The loop's result is the "
                "terminating part of H_N, a lower bound (the mass that left the loop within N iterations).",
    params=(ParamSpec("iterations", "int", "Iterations N", prompt="Enter the number of iterations: ",
                      default=50, minimum=0, help="Number of applications of the EVT operator."),),
    run=_strategy_evt_iterate,
    requires_unconditioned=True,
))
