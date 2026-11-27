"""
The strategy ``auto`` (the GUI's default, #324): a short synthesis of an EVT invariant, kept only if it is certified
exact, otherwise unrolling.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Tuple, Type

import sympy

from prodigy.analysis.exceptions import ForwardAnalysisError, HeuristicsError, SolveError, VerificationError
from prodigy.distribution.equality import parse_engine_text
from prodigy.analysis.loops.api import (Bound, Delivers, LoopDecision, LoopOutcome, LoopStrategy, LoopTask, ParamSpec,
                                        QueryKey, query_key)
from prodigy.analysis.loops.registry import get_loop_strategy, register_loop_strategy
from prodigy.distribution import Distribution
from prodigy.util.logger import log_setup


logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


AUTO_MAX_DEGREE = 2
AUTO_TIME_LIMIT = 5
AUTO_TEMPLATE_TIME_LIMIT = 2


def _upper_query_bounds(task: LoopTask, upper: Distribution) -> Dict[QueryKey, Tuple[Any, Any]]:
    """Upper bounds on the ``?Pr`` and ``?Ex`` queries directly after the loop, evaluated on ``upper``, an
    upper bound of the loop's result (#336). The queries are monotone in the measure (an expectation only if its
    expression is non-negative), so each value bounds the query of the exact result. A query that cannot be
    evaluated, or whose value is not finite, is left out: the bound is only an addition."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.instructionhandler.query_handler import filter_by_cases
    from prodigy.analysis.values import is_monotone_nonnegative
    from prodigy.language import ExpectationInstr, PlotInstr, PrintInstr, ProbabilityQueryInstr, VarExpr
    program = task.prog_info.program
    instructions = list(program.instructions)
    index = next((i for i, ins in enumerate(instructions) if ins is task.instruction), None)
    bounds: Dict[QueryKey, Tuple[Any, Any]] = {}
    for ins in instructions[index + 1:] if index is not None else []:
        if isinstance(ins, (PlotInstr, PrintInstr)):
            continue
        if not isinstance(ins, (ProbabilityQueryInstr, ExpectationInstr)):
            break
        try:
            if isinstance(ins, ProbabilityQueryInstr):
                if isinstance(ins.expr, VarExpr):
                    continue  # a marginal
                value = filter_by_cases(upper, ins.expr).get_probability_mass()
                kind = "?Pr"
            else:
                if not is_monotone_nonnegative(ins.expr, program):
                    continue
                value = upper.get_expected_value_of(ins.expr)
                kind = "?Ex"
            if not isinstance(value, sympy.Basic):
                value = parse_engine_text(str(value), upper.get_variables() | upper.get_parameters())
            if value.is_finite is not True or value.is_real is not True or value < 0:
                continue
        except Exception:  # pylint: disable=broad-except
            # the bound is optional: without it, the query keeps the bound of the loop's own strategy
            logger.debug("No upper bound for the query %s", ins, exc_info=True)
            continue
        bounds[query_key(kind, ins.expr)] = (None, value)
    return bounds


# Errors of the synthesis that mean "no invariant found": solver, CAS, verification, unsupported
# input, and the time limit. Everything else is a bug and is reported as such (#318).
_EXPECTED_SYNTHESIS_ERRORS: Tuple[Type[BaseException], ...] = (
    ForwardAnalysisError, VerificationError, SolveError, HeuristicsError, NotImplementedError,
    TimeoutError, ArithmeticError, sympy.SympifyError, sympy.PolynomialError,
    sympy.polys.polyerrors.BasePolynomialError)


def _strategy_auto(task: LoopTask) -> LoopOutcome:
    """Synthesis first, accepted only if certified exact; otherwise unrolling. Each part is the
    unchanged strategy with its own checks, so the outcome is exactly as sound as theirs."""
    reason = ""
    upper_bounds: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
    if not task.error_prob.is_zero_dist():
        reason = "an observe statement comes before the loop, which EVT reasoning does not support"
    else:
        synthesis = get_loop_strategy("evt_synthesis")
        params = {"max_degree": task.param("max_degree"), "time_limit": task.param("time_limit"),
                  "template_time_limit": AUTO_TEMPLATE_TIME_LIMIT}
        try:
            outcome = synthesis.run(dataclasses.replace(task, strategy=synthesis,
                                                        decision=LoopDecision("evt_synthesis", params)))
        except _EXPECTED_SYNTHESIS_ERRORS as exc:
            reason = (str(exc) or type(exc).__name__).splitlines()[0][:300]
        except Exception as exc:  # pylint: disable=broad-except  # unexpected: logged, never proof of anything
            # A bug inside the synthesis must not look like "no invariant found". Unrolling is still
            # sound (it is its own strategy with its own checks), so the analysis continues, but the
            # error is logged with its traceback and named in the message.
            logger.error("The invariant synthesis of the automatic strategy failed unexpectedly",
                         exc_info=True)
            reason = f"unexpected error {type(exc).__name__}: {(str(exc) or '-').splitlines()[0][:200]}"
        else:
            if outcome.kind == Bound.EXACT:
                outcome.certificate["automatic"] = {"chosen": "evt_synthesis"}
                outcome.messages.insert(0, "Automatic: an EVT invariant was synthesised and certified exact.")
                return outcome
            reason = (f"the synthesised invariant {outcome.certificate.get('invariant')} gives only "
                      f"{'an upper bound' if outcome.kind == Bound.UPPER else 'no guarantee'}")
            if outcome.kind == Bound.UPPER:
                # kept: it tightens the queries right after the loop to intervals (like evt_sandwich)
                upper_bounds = _upper_query_bounds(task, outcome.distribution)
    unroll = get_loop_strategy("unroll")
    outcome = unroll.run(dataclasses.replace(task, strategy=unroll, decision=LoopDecision(
        "unroll", {"iterations": task.param("iterations")})))
    outcome.certificate["automatic"] = {"chosen": "unroll", "synthesis": reason}
    if upper_bounds:
        outcome.query_bounds = {key: (outcome.query_bounds.get(key, (None, None))[0], upper)
                                for key, (_, upper) in upper_bounds.items()}
        outcome.messages.append("The upper bound of the synthesised invariant (infinite mass) is kept: it bounds "
                                "the queries right after the loop from above.")
    outcome.messages.insert(0, f"Automatic: unrolled {task.param('iterations')} iterations, since no exact "
                               f"EVT invariant was found ({reason}).")
    return outcome


register_loop_strategy(LoopStrategy(
    name="auto",
    title="Automatic",
    group="automatic",
    outcome="exact or lower bound",
    summary="Exact if Prodigy finds an EVT invariant within a few seconds, otherwise a lower bound from "
            "unrolling N iterations; you provide nothing.",
    soundness=Bound.LOWER,
    delivers=(Delivers(Bound.LOWER),
              Delivers(Bound.EXACT, "the short synthesis finds an invariant certified exact, or no mass is left "
                                    "inside the loop")),
    description="Runs Synthesise EVT invariant with small templates (degree ≤ 2) and a short time limit and "
                "keeps its result only if it is certified exact. Otherwise (no invariant found, only an "
                "upper bound, an observe statement before the loop, or any failure) it unrolls the loop N "
                "times, which gives a lower bound whose missing mass turns every query into an interval. "
                "Both parts are the unchanged strategies with their own checks.",
    params=(ParamSpec("iterations", "int", "Iterations N", default=50, minimum=0,
                      prompt="Specify a maximum iteration limit: ",
                      help="Loop iterations to unroll when the synthesis finds no exact invariant."),
            ParamSpec("max_degree", "int", "Maximal template degree", default=AUTO_MAX_DEGREE, minimum=0,
                      optional=True, help="Templates of the synthesis up to this degree in each variable."),
            ParamSpec("time_limit", "int", "Synthesis time limit (seconds)", default=AUTO_TIME_LIMIT, minimum=1,
                      optional=True, help="How long to search for an invariant before unrolling.")),
    run=_strategy_auto,
))
