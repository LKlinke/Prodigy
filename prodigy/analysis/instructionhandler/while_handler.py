"""
While loops.

The handler asks the current :class:`~prodigy.analysis.session.AnalysisSession` which
:mod:`loop strategy <prodigy.analysis.loop_strategies>` to use for a loop and runs it.
The six historic strategies are implemented and registered here; in the interactive CLI
they show up in the same menu, with the same prompts, as before.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy
from prodigy.pgcl import Instr, parse_pgcl, Program, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategy, SynthesisStrategies
from prodigy.analysis.evtinvariants.invariant_synthesis import evt_invariant_synthesis
from prodigy.analysis.evtinvariants.invariant_verification import evt_invariant_check
from prodigy.analysis.exceptions import ForwardAnalysisError, VerificationError
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.loop_strategies import (Bound, LoopOutcome, LoopStrategy, LoopTask, ParamSpec,
                                              get_loop_strategy, register_loop_strategy)
from prodigy.analysis.session import LoopDecision, current_session
from prodigy.distribution import Distribution, State
from prodigy.distribution.generating_function import SympyPGF
from prodigy.pgcl.pgcl_operations import cav_phi
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Analyzer = Callable[
    [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
    Tuple[Distribution, Distribution]
]


class UnsupportedLoopError(ForwardAnalysisError):
    """The chosen strategy cannot handle this loop (e.g. EVT reasoning with conditioning)."""


class InvariantRefutedError(VerificationError):
    """A proposed invariant is not an invariant; ``details`` holds the evidence."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _mass(dist: Distribution) -> sympy.Expr:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import sym
    return sym(dist.get_probability_mass())


def _require_unconditioned(task: LoopTask) -> None:
    if not task.error_prob.is_zero_dist():
        raise UnsupportedLoopError(
            "Currently EVT reasoning does not support conditioning: the strategy "
            f"'{task.strategy.title}' cannot be used after an observe statement.")


def finite_mass(invariant: Any, variables: Sequence[str]) -> Tuple[Optional[bool], Optional[sympy.Expr]]:
    """Decides |I| < ∞ for a generating function I with non-negative coefficients.

    Substitutes t for every variable; f(t) = I(t, …, t) has the same mass. For a rational f
    with f's denominator q(0) ≠ 0, Pringsheim's theorem says the radius of convergence is a
    singularity on the positive axis, so |I| < ∞ iff q has no root in (0, 1]; then
    |I| = f(1). Returns (None, None) when this cannot be decided (parameters, non-rational).
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import sym
    try:
        expr = sym(invariant)
        t = sympy.Symbol("t__mass", positive=True)
        substitution = {s: t for s in expr.free_symbols if s.name in set(variables)}
        if any(s.name not in set(variables) for s in expr.free_symbols):
            return None, None
        f = sympy.cancel(sympy.together(expr.subs(substitution)))
        if not f.is_rational_function(t):
            return None, None
        _, denominator = sympy.fraction(f)
        poly = sympy.Poly(denominator, t)
        if poly.eval(0) == 0:
            return None, None
        if poly.count_roots(0, 1) > 0:
            return False, None
        return True, sympy.nsimplify(f.subs(t, 1))
    except Exception:  # pylint: disable=broad-except
        return None, None


def _positivity(config: ForwardAnalysisConfig, variables, invariant: Any) -> Optional[bool]:
    try:
        strategy: SynthesisStrategy = SynthesisStrategies.make(config.strategy, variables, config.factory)
        verdict = strategy.positivity_heuristics.is_positive(sympy.S(str(invariant).replace("^", "**")))
        return verdict if verdict in (True, False) else None
    except Exception:  # pylint: disable=broad-except
        return None


def _evt_outcome_kind(scope_kind: str, positive: Optional[bool], finite: Optional[bool]) -> Tuple[str, List[str]]:
    """Guarantee for a fixed point I of Ψ_G(H) = G + ⟦body⟧(⟨H⟩_B).

    ``scope_kind`` says how the body was analysed while checking I: exactly, or with inner
    loops over-approximated (then Ψ' ⪰ Ψ, so I = Ψ'(I) ⪰ Ψ(I) and Park still applies, but the
    finite-mass rule for the other direction does not)."""
    messages = []
    if scope_kind not in (Bound.EXACT, Bound.UPPER):
        messages.append("Loops inside the body were under-approximated while checking the invariant; "
                        "no guarantee.")
        return Bound.UNKNOWN, messages
    if positive is False:
        messages.append("The invariant has negative coefficients, so Park's lemma does not apply.")
        return Bound.UNKNOWN, messages
    if positive is None:
        messages.append("Non-negativity of the invariant's coefficients could not be established; "
                        "no guarantee.")
        return Bound.UNKNOWN, messages
    if scope_kind == Bound.UPPER:
        messages.append("Upper bound (Park's lemma): loops inside the body were over-approximated while "
                        "checking the invariant, so exactness cannot be certified.")
        return Bound.UPPER, messages
    if finite is True:
        messages.append("Exact: I is a non-negative fixed point of Ψ_G with finite mass. "
                        "Park's lemma gives ⪯, the finite-mass (lower induction) rule gives ⪰.")
        return Bound.EXACT, messages
    if finite is False:
        messages.append("Upper bound only: I has infinite mass and may be a spurious fixed point "
                        "(Park's lemma gives the result ⪯ ⟨I⟩_¬B).")
        return Bound.UPPER, messages
    messages.append("Upper bound (Park's lemma); finiteness of the mass of I could not be decided, "
                    "so exactness is not certified.")
    return Bound.UPPER, messages


# ---------------------------------------------------------------------------------------
# The six built-in strategies
# ---------------------------------------------------------------------------------------

def _strategy_invariant(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config = task.instruction, task.prog_info, task.config
    inv_src = task.param("invariant")
    inv_prog = parse_pgcl(inv_src)

    prog = Program(declarations=[],
                   variables=prog_info.variables,
                   constants=prog_info.constants,
                   parameters=prog_info.parameters,
                   instructions=[instruction],
                   functions=prog_info.functions)
    print(f"{Style.YELLOW}Verifying invariant...{Style.RESET}")

    # First we create the modified input program in order to fit the premise of Park's Lemma
    phi_inv = cav_phi(prog, inv_prog)

    with task.session.sub_analysis("invariant check") as scope:
        answer, result = check_equivalence(inv_prog, phi_inv, config, task.analyzer)

    certificate: Dict[str, Any] = {
        "type": "invariant",
        "rule": "Park",
        "claim": "Φ(I) = I for all inputs, hence ⟦while⟧ ⪯ ⟦I⟧",
        "invariant": inv_src,
    }
    if answer:
        assert isinstance(result, list)
        constraints = [{str(k): str(v) for k, v in solution.items()} for solution in result]
        certificate["verdict"] = "verified"
        certificate["constraints"] = constraints
        messages = []
        if len(result) == 0:
            print(Style.OKGREEN + "Invariant successfully validated!" + Style.RESET)
        else:
            print(
                f"{Style.OKGREEN}Invariant validated under the following constraints:{Style.RESET} {result}"
            )
            messages.append(f"The invariant holds under the constraints {constraints}; "
                            "the result still contains the parameters.")
        if config.show_intermediate_steps:
            print(Style.YELLOW +
                  "Compute the result using the invariant" +
                  Style.RESET)
        dist, error_prob = task.analyzer(inv_prog.instructions, prog_info, task.distribution, task.error_prob,
                                         config)
        if scope.kind == Bound.EXACT:
            kind = Bound.UPPER
            messages.append("Upper bound by Park's lemma; exact if the loop terminates almost surely.")
        elif scope.kind == Bound.UPPER:
            # an over-approximated body Φ' ⪰ Φ with Φ'(I) = I gives Φ(I) ⪯ I, so Park still applies
            kind = Bound.UPPER
            messages.append("Upper bound by Park's lemma; loops inside the body were over-approximated "
                            "during the check, which keeps the argument sound.")
        else:
            kind = Bound.UNKNOWN
            messages.append("Loops inside the body were under-approximated during the check; no guarantee.")
        return LoopOutcome(dist, error_prob, kind=kind, certificate=certificate, messages=messages)

    if answer is False:
        assert isinstance(result, State)
        print(
            f'{Style.OKRED}Invariant could not be verified.{Style.RESET} Counterexample: {result.valuations}'
        )
        certificate["verdict"] = "refuted"
        certificate["counterexample"] = {str(k): v for k, v in result.valuations.items()}
        raise InvariantRefutedError(
            f"Invariant could not be determined as such. Counterexample: {result.valuations}",
            details=certificate)

    assert answer is None
    assert isinstance(result, Distribution)
    print(
        f'{Style.OKRED}Could not determine whether invariant is valid.{Style.RESET} Phi(I) - I: {result}'
    )
    certificate["verdict"] = "unknown"
    certificate["difference"] = str(result)
    raise InvariantRefutedError("Could not determine whether invariant is valid.", details=certificate)


def _strategy_unroll(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    error_prob = task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations", max_iter)

    sat_part = task.distribution.filter(instruction.cond)
    non_sat_part = task.distribution - sat_part
    fixpoint_after = None
    for i in range(max_iter):
        task.progress(i + 1, max_iter)
        iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        if iterated_non_sat == SympyPGF.zero(
        ) and iterated_sat == sat_part:
            print(
                f"\n{Style.OKGREEN}Terminated already after {i + 1} step(s)!{Style.RESET}"
            )
            fixpoint_after = i + 1
            break
        non_sat_part += iterated_non_sat
        sat_part = iterated_sat

    remaining = _mass(sat_part)
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
    while Fraction(captured_part.get_probability_mass()
                   ) < captured_probability_threshold:
        if max_iterations is not None and iterations >= max_iterations:
            task.note(f"Stopped after {iterations} iterations before reaching the threshold.")
            break
        logger.info("Collected %f of the desired mass", (float(
            (Fraction(captured_part.get_probability_mass()) /
             captured_probability_threshold)) * 100))
        iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
        iterated_sat = iterated_part.filter(instruction.cond)
        iterated_non_sat = iterated_part - iterated_sat
        non_sat_part += iterated_non_sat
        captured_part = non_sat_part + error_prob
        sat_part = iterated_sat
        iterations += 1
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


def _strategy_evt_iterate(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    distribution, error_prob = task.distribution, task.error_prob
    max_iter = task.param("iterations")
    logger.debug("Compute %i iterations of EVT operator", max_iter)

    evt = distribution * 0
    with task.session.sub_analysis("EVT iteration") as scope:
        for i in range(max_iter):
            logger.debug("Iteration %i", i)
            task.progress(i + 1, max_iter)
            evt = distribution + \
                  analyzer(instruction.body, prog_info, evt.filter(instruction.cond), error_prob, config)[0]
    result = evt - evt.filter(instruction.cond)
    remaining = sympy.simplify(_mass(distribution) - _mass(result))
    certificate = {"type": "evt_iteration", "iterations": max_iter, "remaining_mass": str(remaining),
                   "evt": str(evt)}
    if scope.kind not in (Bound.EXACT, Bound.LOWER):
        return LoopOutcome(result, error_prob, kind=Bound.UNKNOWN, certificate=certificate,
                           messages=["Loops inside the body were over-approximated; no guarantee."])
    if remaining == 0:
        return LoopOutcome(result, error_prob, kind=Bound.EXACT, certificate=certificate,
                           messages=[f"Exact: all mass has left the loop within {max_iter} iterations."])
    return LoopOutcome(result, error_prob, kind=Bound.LOWER, residual=remaining, certificate=certificate,
                       messages=[f"Lower bound from {max_iter} iterations of the EVT operator; "
                                 f"mass {remaining} is not accounted for."])


def _evt_certificate(kind_name: str, invariant: Distribution, variables, config, scope_kind: str,
                     loop_cond) -> Tuple[str, Dict[str, Any], List[str]]:
    positive = _positivity(config, variables, invariant)
    finite, mass = finite_mass(str(invariant), list(variables)) if positive else (None, None)
    kind, messages = _evt_outcome_kind(scope_kind, positive, finite)
    certificate = {
        "type": kind_name,
        "rule": "Park + finite-mass lower induction" if kind == Bound.EXACT else "Park",
        "equation": f"I = G + ⟦body⟧(⟨I⟩_({loop_cond}))",
        "invariant": str(invariant),
        "nonnegative": positive,
        "finite_mass": finite,
        "mass": str(mass) if mass is not None else None,
    }
    return kind, certificate, messages


def _strategy_evt_invariant(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    variables = list(prog_info.program.variables.keys())
    evt_inv = config.factory.from_expr(task.param("invariant"), *variables)
    try:
        with task.session.sub_analysis("EVT invariant check") as scope:
            solution, full_invariant, solutions = evt_invariant_check(instruction, prog_info, task.distribution,
                                                                      evt_inv, config, analyzer)
    except VerificationError as exc:
        raise InvariantRefutedError(f"Could not validate the EVT invariant {evt_inv}",
                                    details={"type": "evt_invariant", "invariant": str(evt_inv),
                                             "verdict": "refuted", "reason": str(exc)}) from exc
    print(f"{Style.OKGREEN}Continuing with {solution}{Style.RESET}")
    kind, certificate, messages = _evt_certificate("evt_invariant", full_invariant, variables, config,
                                                   scope.kind, instruction.cond)
    certificate["candidate"] = str(evt_inv)
    certificate["solutions"] = solutions
    return LoopOutcome(solution, task.error_prob, kind=kind, certificate=certificate, messages=messages)


def _strategy_evt_synthesis(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config, analyzer = task.instruction, task.prog_info, task.config, task.analyzer
    _require_unconditioned(task)
    logger.debug("Using invariant synthesis.")

    # Build the strategy.
    strategy: SynthesisStrategy = SynthesisStrategies.make(config.strategy, prog_info.variables.keys(),
                                                           config.factory)

    # generate the invariants using the strategy
    with task.session.sub_analysis("EVT invariant synthesis") as scope:
        invariants = evt_invariant_synthesis(instruction, prog_info, task.distribution, config, strategy,
                                             analyzer)

    def _sorting_key(elem):
        return 1 if elem[1] else 0

    if task.interactive:
        # historic order of the CLI (kept unchanged)
        invariants = sorted(invariants, key=_sorting_key)
    else:
        # certified candidates first
        invariants = sorted(invariants, key=_sorting_key, reverse=True)

    # give associated distributions:
    distributions = [inv - inv.filter(instruction.cond) for inv, inv_type in invariants]
    print("Continuing with the following distributions is possible:")
    for i, d in enumerate(distributions):
        print(f"{i + 1}.\t{d}")

    # Choose one solution
    index = task.session.provider.choose_candidate(task, [str(d) for d in distributions])
    print(f"Continue with distribution: {Style.CYAN}{distributions[index]}{Style.RESET}")

    variables = list(prog_info.program.variables.keys())
    candidates = []
    for i, (inv, positive) in enumerate(invariants):
        finite, mass = finite_mass(str(inv), variables) if positive else (None, None)
        candidates.append({"index": i, "invariant": str(inv), "result": str(distributions[i]),
                           "nonnegative": positive, "finite_mass": finite,
                           "mass": str(mass) if mass is not None else None})
    chosen, positive = invariants[index]
    kind, certificate, messages = _evt_certificate("evt_synthesis", chosen, variables, config, scope.kind,
                                                   instruction.cond)
    certificate["chosen"] = index
    certificate["candidates"] = candidates
    certificate["synthesis_strategy"] = config.strategy.name
    return LoopOutcome(distributions[index], task.error_prob, kind=kind, certificate=certificate,
                       messages=messages)


_INVARIANT_PARAMS = (
    ParamSpec("invariant", "pgcl", "Invariant (loop-free pGCL program)",
              prompt="Invariant file:\t",
              help="A loop-free program I with the loop's variables; checked against one unrolling "
                   "of the loop (Park's lemma)."),
    ParamSpec("invariant_file", "file", "Invariant file", prompt="Invariant file:\t",
              alias_of="invariant", help="Path of a file containing the invariant (CLI only)."),
)

register_loop_strategy(LoopStrategy(
    name="invariant",
    title="Check invariant",
    menu_text="Solve using invariants (Checks whether the invariant over-approximates the loop)",
    legacy_key="1",
    soundness=Bound.UPPER,
    description="Checks that a loop-free program I satisfies I ≡ if (guard) { body; I } on all "
                "inputs and continues with I. By Park's lemma the result over-approximates the "
                "loop; it is exact if the loop terminates almost surely.",
    params=_INVARIANT_PARAMS,
    run=_strategy_invariant,
))

register_loop_strategy(LoopStrategy(
    name="unroll",
    title="Unroll N iterations",
    menu_text="Fix a maximum number of iterations (This results in an under-approximation)",
    legacy_key="2",
    soundness=Bound.LOWER,
    description="Executes at most N iterations. Mass that is still inside the loop afterwards is "
                "reported as residual mass, which turns every query into a sound interval.",
    params=(ParamSpec("iterations", "int", "Iterations N", prompt="Specify a maximum iteration limit: ",
                      default=50, minimum=0, help="Maximal number of loop iterations."),),
    run=_strategy_unroll,
))

register_loop_strategy(LoopStrategy(
    name="mass_threshold",
    title="Iterate until mass captured",
    menu_text="Analyse until a certain probability mass is captured (might not terminate!)",
    legacy_key="3",
    soundness=Bound.LOWER,
    description="Iterates until the terminated (or conditioned-away) mass reaches the threshold. "
                "Does not terminate if the loop does not terminate with at least this probability.",
    params=(ParamSpec("threshold", "probability", "Threshold", prompt="Enter the probability threshold: ",
                      default="0.99", help="Probability mass to capture, e.g. 0.999 or 999/1000."),
            ParamSpec("max_iterations", "int", "Iteration cap", default=None, minimum=1, optional=True,
                      help="Optional safety cap on the number of iterations.")),
    run=_strategy_mass_threshold,
))

register_loop_strategy(LoopStrategy(
    name="evt_iterate",
    title="Approximate expected visiting times",
    menu_text="Compute expected visiting times (might be an approximation)",
    legacy_key="4",
    soundness=Bound.LOWER,
    description="Iterates the expected-visiting-time operator Ψ_G(H) = G + ⟦body⟧(⟨H⟩_guard) N times "
                "from 0; the terminating part is a lower bound.",
    params=(ParamSpec("iterations", "int", "Iterations N", prompt="Enter the number of iterations: ",
                      default=50, minimum=0, help="Number of applications of the EVT operator."),),
    run=_strategy_evt_iterate,
    requires_unconditioned=True,
))

register_loop_strategy(LoopStrategy(
    name="evt_invariant",
    title="Check EVT invariant",
    menu_text="Use EVT Invariant",
    legacy_key="5",
    soundness=Bound.UPPER,
    description="Checks a generating function I against I = G + ⟦body⟧(⟨I⟩_guard), solving for its "
                "parameters. With non-negative coefficients the result is an upper bound (Park), and "
                "exact if I has finite mass.",
    params=(ParamSpec("invariant", "expression", "EVT invariant I", prompt="Enter EVT invariant: ",
                      help="Generating function over the program variables, e.g. 1/(1-c/2)."),),
    run=_strategy_evt_invariant,
    requires_unconditioned=True,
))

register_loop_strategy(LoopStrategy(
    name="evt_synthesis",
    title="Synthesise EVT invariant",
    menu_text="EVT Invariant syntesis",
    legacy_key="6",
    soundness=Bound.UPPER,
    description="Searches rational-function templates for a fixed point of the EVT operator and "
                "checks non-negativity heuristically (synthesis strategy from the settings).",
    params=(ParamSpec("candidate", "int", "Candidate", default=None, minimum=0, optional=True,
                      help="Index of the synthesised candidate to continue with (default: the first "
                           "certified one)."),),
    run=_strategy_evt_synthesis,
    requires_unconditioned=True,
))


# ---------------------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------------------

class WhileHandler(InstructionHandler):

    @staticmethod
    def _run(strategy_name: Optional[str], instruction: WhileInstr, prog_info: ProgramInfo,
             distribution: Distribution, error_prob: Distribution, config: ForwardAnalysisConfig,
             analyzer: Analyzer) -> Tuple[Distribution, Distribution]:
        session = current_session()
        with session.activate():
            loop = session.loops.lookup(instruction)
            decision = session.decide_loop(loop, instruction) if strategy_name is None \
                else LoopDecision(strategy_name)
            strategy = get_loop_strategy(decision.strategy)
            task = LoopTask(instruction=instruction, prog_info=prog_info, distribution=distribution,
                            error_prob=error_prob, config=config, analyzer=analyzer, session=session,
                            loop=loop, decision=decision, strategy=strategy)
            outcome = session.run_loop(task)
        return outcome.distribution, outcome.error_prob

    # The historic entry points, kept for callers that use them directly.

    @staticmethod
    def _analyze_with_invariant(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("invariant", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _compute_iterations(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("unroll", instruction, prog_info, distribution, error_prob, config, analyzer)

    @staticmethod
    def _compute_until_threshold(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("mass_threshold", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _approx_expected_visiting_times(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("evt_iterate", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _evt_invariant(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("evt_invariant", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def _evt_invariant_synthesis(instruction, prog_info, distribution, error_prob, config, analyzer):
        return WhileHandler._run("evt_synthesis", instruction, prog_info, distribution, error_prob, config,
                                 analyzer)

    @staticmethod
    def compute(
            instruction: Instr,
            prog_info: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Analyzer
    ) -> tuple[Distribution, Distribution]:

        _assume(instruction, WhileInstr, 'WhileHandler')
        return WhileHandler._run(None, instruction, prog_info, distribution, error_prob, config, analyzer)
