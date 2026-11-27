"""
The strategies that unroll a loop: ``unroll`` (N iterations) and ``mass_threshold`` (until the captured mass
reaches a threshold). Both give a lower bound; exact if no mass is left inside the loop or an iteration reaches a
fixed point (:func:`_fixed_point`, #508), where they stop early.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any, Dict, List, Optional

import sympy

from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import register_loop_strategy
from prodigy.distribution import Distribution
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.util.logger import log_setup
from prodigy.analysis.loops.common import _mass


logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def _finite_mass(dist: Distribution) -> Optional[sympy.Expr]:
    """The mass of ``dist`` if it is proven to be a finite number, otherwise ``None``."""
    try:
        mass = _mass(dist)
    except NonFiniteMarginalError:
        return None
    return mass if mass.is_number and mass.is_finite else None


def _fixed_point(sat_part: Distribution, error_before: Distribution, iterated_sat: Distribution,
                 iterated_non_sat: Distribution, error_after: Distribution) -> bool:
    """
    Whether one iteration of the body maps the mass inside the loop, ``sat_part``, to itself: nothing leaves the
    loop (``iterated_non_sat`` is provably zero), the part that stays is ``sat_part`` again, and no mass is lost to
    an observation (the violation mass is unchanged). Then the unrolling is exact as it stands, and the mass of
    ``sat_part`` stays inside the loop forever (#508).

    Soundness. ``is_provably_zero`` and ``==`` are true only with a proof, on both engines. The analysis of the body
    is a function of its input, so the next iteration has the same input as this one and, by induction, every later
    iteration adds zero to the exit part and zero to the violation mass: the current exit part and violation mass
    are the limit of the unrolling. The violation mass must be compared as well: a body may keep the mass inside
    the loop but lose some to ``observe`` in every iteration (on the infinite-mass test distribution of the
    second-order check, ``while (true) { if (x == 0) { observe(false); } else { x = x - 1; } }`` maps the uniform
    weight on all ``x`` to itself). The mass of ``sat_part`` must also be a proven finite number. Then the argument
    holds also if the body itself is only approximated (an inner loop with a bound): a lower bound L of the body's
    exact semantics T that keeps all of the finite mass (L(s) = s, nothing exits, nothing violated) leaves T no mass
    to add, so T(s) = L(s) on this input; an upper bound U with U(s) = s and nothing exiting gives T no exit mass in
    any later iteration. The guarantee of the body itself is booked by its own loops (``Bound.combine``).
    """
    if not (iterated_non_sat.is_provably_zero() and iterated_sat == sat_part and error_after == error_before):
        return False
    return _finite_mass(sat_part) is not None


def _strategy_unroll(task: LoopTask) -> LoopOutcome:
    """Unrolls the loop at most N times. It stops early, exactly, when no mass is left inside the loop, or when an
    iteration reaches a fixed point (:func:`_fixed_point`); otherwise the mass still inside the loop is the
    residual of a lower bound."""
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations", max_iter)

    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    task.iterate(0, non_sat_part, inside=sat_part)
    done, fixpoint = max_iter, False
    for i in range(max_iter):
        if sat_part.is_provably_zero():
            done = i
            break
        task.progress(i + 1, max_iter)
        iterated_part, error_after = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        if _fixed_point(sat_part, error_prob, iterated_sat, iterated_non_sat, error_after):
            done, fixpoint = i + 1, True
            break
        error_prob = error_after
        non_sat_part += iterated_non_sat
        sat_part = iterated_sat
        task.iterate(i + 1, non_sat_part, inside=sat_part)

    try:
        remaining = _mass(sat_part)
    except NonFiniteMarginalError:
        # e.g. the test distribution of the second-order check, which has infinite mass by design (#403);
        # SymPy computes oo there. An unbounded residual keeps the lower bound sound.
        remaining = sympy.oo
    certificate: Dict[str, Any] = {"type": "unrolling", "iterations": max_iter, "iterations_done": done,
                                   "remaining_mass": str(remaining)}
    if fixpoint:
        certificate["fixpoint"] = True
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=_fixed_point_messages(done, remaining))
    if remaining == 0:
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: no mass is left inside the loop after {done} iterations."])
    return LoopOutcome(non_sat_part, error_prob, kind=Bound.LOWER, residual=remaining,
                       certificate=certificate,
                       messages=[f"Lower bound after {max_iter} iterations; mass {remaining} is still "
                                 "inside the loop."])


def _fixed_point_messages(done: int, remaining: Any) -> List[str]:
    messages = [f"Exact: the iteration reached a fixed point after {done} step(s)."]
    if remaining != 0:
        messages.append(f"Mass {remaining} stays inside the loop forever (divergence).")
    return messages


def _strategy_mass_threshold(task: LoopTask) -> LoopOutcome:
    """Iterates until the terminated and violated mass reaches the threshold (or the cap). Like ``unroll``, it
    stops early, exactly, when no mass is left inside the loop or an iteration reaches a fixed point
    (:func:`_fixed_point`), also if the threshold is never reached then (#508)."""
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    captured_probability_threshold = Fraction(task.param("threshold"))
    max_iterations = task.param("max_iterations")
    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    captured_part = non_sat_part + error_prob
    iterations, fixpoint, capped = 0, False, False
    task.iterate(0, non_sat_part, inside=sat_part)
    while (captured := Fraction(captured_part.get_probability_mass())) < captured_probability_threshold:
        if max_iterations is not None and iterations >= max_iterations:
            capped = True
            break
        if sat_part.is_provably_zero():
            break
        logger.info("Collected %f of the desired mass", float(captured / captured_probability_threshold) * 100)
        iterated_part, error_after = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        if _fixed_point(sat_part, error_prob, iterated_sat, iterated_non_sat, error_after):
            iterations, fixpoint = iterations + 1, True
            break
        error_prob = error_after
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
    certificate: Dict[str, Any] = {"type": "mass_threshold", "threshold": str(captured_probability_threshold),
                                   "iterations": iterations, "captured_mass": str(captured_part.get_probability_mass()),
                                   "remaining_mass": str(remaining)}
    if fixpoint:
        certificate["fixpoint"] = True
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=_fixed_point_messages(iterations, remaining))
    if remaining == 0:
        return LoopOutcome(non_sat_part, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: no mass is left inside the loop after {iterations} iterations."])
    return LoopOutcome(non_sat_part, error_prob, kind=Bound.LOWER, residual=remaining, certificate=certificate,
                       messages=[f"Lower bound after {iterations} iterations"
                                 f"{' (the cap, before reaching the threshold)' if capped else ''}; mass {remaining} "
                                 "is still inside the loop."])


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
    delivers=(Delivers(Bound.LOWER),
              Delivers(Bound.EXACT, "no mass is left inside the loop, or the iteration reaches a fixed point")),
    description="Iterates until the terminated (or conditioned-away) mass reaches the threshold. "
                "Does not terminate if the loop does not terminate with at least this probability.",
    params=(ParamSpec("threshold", "probability", "Threshold", prompt="Enter the probability threshold: ",
                      default="0.99", help="Probability mass to capture, e.g. 0.999 or 999/1000."),
            ParamSpec("max_iterations", "int", "Iteration cap", default=None, minimum=1, optional=True,
                      help="Optional safety cap on the number of iterations.")),
    run=_strategy_mass_threshold,
))
