"""
The strategies that unroll a loop: ``unroll`` (N iterations) and ``mass_threshold`` (until the captured mass
reaches a threshold). Both give a lower bound; exact if no mass is left inside the loop.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any, Dict

import sympy

from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import register_loop_strategy
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.util.color import Style
from prodigy.util.logger import log_setup
from prodigy.analysis.loops.common import _mass


logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def _strategy_unroll(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations", max_iter)

    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    task.iterate(0, non_sat_part, inside=sat_part)
    fixpoint_after = None
    for i in range(max_iter):
        task.progress(i + 1, max_iter)
        iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        # config.factory.zero() has no variables, so this is False whenever iterated_non_sat has some (as with
        # SympyPGF.zero() before, T5 of the architecture page)
        if iterated_non_sat == config.factory.zero() and iterated_sat == sat_part:
            print(
                f"\n{Style.OKGREEN}Terminated already after {i + 1} step(s)!{Style.RESET}"
            )
            fixpoint_after = i + 1
            break
        non_sat_part += iterated_non_sat
        sat_part = iterated_sat
        task.iterate(i + 1, non_sat_part, inside=sat_part)

    try:
        remaining = _mass(sat_part)
    except NonFiniteMarginalError:
        # e.g. the test distribution of the second-order check, which has infinite mass by design (#403);
        # SymPy computes oo there. An unbounded residual keeps the lower bound sound.
        remaining = sympy.oo
    certificate: Dict[str, Any] = {"type": "unrolling", "iterations": max_iter,
                                   "iterations_done": fixpoint_after or max_iter,
                                   "remaining_mass": str(remaining)}
    if fixpoint_after is not None:
        # One more iteration changes nothing: the mass still inside the loop never leaves it.
        certificate["fixpoint"] = True
        messages = [f"Exact: the iteration reached a fixed point after {fixpoint_after} step(s)."]
        if remaining != 0:
            messages.append(f"Mass {remaining} stays inside the loop forever (divergence).")
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=messages)
    if remaining == 0:
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: no mass is left inside the loop after {max_iter} iterations."])
    return LoopOutcome(non_sat_part, error_prob, kind=Bound.LOWER, residual=remaining,
                       certificate=certificate,
                       messages=[f"Lower bound after {max_iter} iterations; mass {remaining} is still "
                                 "inside the loop."])


def _strategy_mass_threshold(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    captured_probability_threshold = Fraction(task.param("threshold"))
    max_iterations = task.param("max_iterations")
    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    captured_part = non_sat_part + error_prob
    iterations = 0
    task.iterate(0, non_sat_part, inside=sat_part)
    while (captured := Fraction(captured_part.get_probability_mass())) < captured_probability_threshold:
        if max_iterations is not None and iterations >= max_iterations:
            task.note(f"Stopped after {iterations} iterations before reaching the threshold.")
            break
        logger.info("Collected %f of the desired mass", float(captured / captured_probability_threshold) * 100)
        iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        non_sat_part += iterated_non_sat
        captured_part = non_sat_part + error_prob
        sat_part = iterated_sat
        iterations += 1
        task.iterate(iterations, non_sat_part, inside=sat_part)
        task.progress(min(100, int(
            (Fraction(captured_part.get_probability_mass()) /
             captured_probability_threshold) * 100)), 100,
            f"{iterations} iterations")
    remaining = _mass(sat_part)
    certificate = {"type": "mass_threshold", "threshold": str(captured_probability_threshold),
                   "iterations": iterations, "captured_mass": str(captured_part.get_probability_mass()),
                   "remaining_mass": str(remaining)}
    if remaining == 0:
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: no mass is left inside the loop after {iterations} iterations."])
    return LoopOutcome(non_sat_part, error_prob, kind=Bound.LOWER, residual=remaining, certificate=certificate,
                       messages=[f"Lower bound after {iterations} iterations; mass {remaining} is still "
                                 "inside the loop."])

register_loop_strategy(LoopStrategy(
    name="unroll",
    title="Unroll N iterations",
    group="bounds",
    outcome="lower bound",
    summary="A lower bound whose missing mass makes every query an interval; you provide N; the cost grows "
            "with N.",
    legacy_key="2",
    soundness=Bound.LOWER,
    delivers=(Delivers(Bound.LOWER),
              Delivers(Bound.EXACT, "no mass is left inside the loop, or the iteration reaches a fixed point")),
    description="Executes at most N iterations. Mass that is still inside the loop afterwards is "
                "reported as residual mass, which turns every query into a sound interval.",
    params=(ParamSpec("iterations", "int", "Iterations N", prompt="Specify a maximum iteration limit: ",
                      default=50, minimum=0, help="Maximal number of loop iterations."),),
    run=_strategy_unroll,
))

register_loop_strategy(LoopStrategy(
    name="mass_threshold",
    title="Iterate until mass captured",
    group="bounds",
    outcome="lower bound",
    summary="A lower bound that captures at least the given mass; you provide the mass, e.g. 0.99; may not "
            "stop if the loop rarely terminates.",
    legacy_key="3",
    soundness=Bound.LOWER,
    delivers=(Delivers(Bound.LOWER), Delivers(Bound.EXACT, "no mass is left inside the loop")),
    description="Iterates until the terminated (or conditioned-away) mass reaches the threshold. "
                "Does not terminate if the loop does not terminate with at least this probability.",
    params=(ParamSpec("threshold", "probability", "Threshold", prompt="Enter the probability threshold: ",
                      default="0.99", help="Probability mass to capture, e.g. 0.999 or 999/1000."),
            ParamSpec("max_iterations", "int", "Iteration cap", default=None, minimum=1, optional=True,
                      help="Optional safety cap on the number of iterations.")),
    run=_strategy_mass_threshold,
))
