import logging
from typing import Callable, Dict, List, Sequence, Tuple, Union

import sympy


from prodigy.pgcl import Instr, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution.distribution import Distribution
from prodigy.distribution.equality import sympify_distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Analyzer = Callable[[Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                    Tuple[Distribution, Distribution]]
"""Computes the semantics of instructions (``compute_semantics``)."""


class FixedPointCheckFailed(VerificationError):
    """
    The EVT invariant was not certified. ``verdict`` is ``"refuted"`` if I ≠ Φ(I) is proven (and no
    parameter values are left to try), otherwise ``"unknown"``: equality was neither proven nor refuted,
    so the invariant may still be correct. ``difference`` is Φ(I) − I.
    """

    def __init__(self, message: str, verdict: str, difference: str):
        super().__init__(message)
        self.verdict = verdict
        self.difference = difference


def evt_operator(loop: WhileInstr,
                 prog_info: ProgramInfo,
                 distribution: Distribution,
                 invariant: Distribution,
                 config: ForwardAnalysisConfig,
                 analyzer: Analyzer) -> Distribution:
    """Applies the EVT operator Ψ(I) = distribution + ⟦body⟧(⟨I⟩_guard) to ``invariant``."""
    zero_dist = config.factory.from_expr("0", *prog_info.program.variables.keys())
    return distribution + analyzer(loop.body, prog_info, invariant.filter(loop.cond), zero_dist, config)[0]


def is_evt_fixed_point(loop: WhileInstr,
                       prog_info: ProgramInfo,
                       distribution: Distribution,
                       invariant: Distribution,
                       config: ForwardAnalysisConfig,
                       analyzer: Analyzer) -> bool:
    """
    Whether ``invariant`` is provably a fixed point of the EVT operator. Parameter values found by a solver
    are only proposals; an invariant with these values substituted has to pass this check.
    """
    return invariant.decide_equal(evt_operator(loop, prog_info, distribution, invariant, config, analyzer)) is True


def evt_invariant_check(loop: WhileInstr,
                        prog_info: ProgramInfo,
                        distribution: Distribution,
                        invariant: Distribution,
                        config: ForwardAnalysisConfig,
                        analyzer: Analyzer) -> Tuple[Distribution, Distribution, List[Dict[str, str]]]:
    """
    Checks whether ``invariant`` is a fixed point of the EVT operator
    Ψ(I) = distribution + ⟦body⟧(⟨I⟩_guard), solving for the invariant's parameters if needed.

    :returns: the terminating part ⟨I⟩_¬guard, the invariant I itself (with the first solution
        substituted) and all solutions found.
    :raises FixedPointCheckFailed: if the invariant cannot be validated, with the verdict ``refuted`` or
        ``unknown``.
    """
    logger.debug("Invariant Verification for loop %s with initial distribution %s.", loop, distribution)
    print(f"{Style.YELLOW}Invariant verification initiated...{Style.RESET}")

    # Compute one iteration step.
    phi = evt_operator(loop, prog_info, distribution, invariant, config, analyzer)

    # Validate the invariant.
    logger.debug("Trying to validate invariant: %s", invariant)
    logger.debug("Phi(inv) = %s", phi)
    verdict = invariant.decide_equal(phi)
    if verdict is True:
        print(f"{Style.OKGREEN}Invariant validated!{Style.RESET}")
        print(f"{Style.GREEN}Invariant: {invariant}{Style.RESET}{Style.CLEARTOEND}")
        return invariant - invariant.filter(loop.cond), invariant, []

    # If the invariant does not match, we need to find a solution.
    diff = invariant - phi
    # Without parameters there is nothing to solve for (and sympy.solve rejects an empty set of symbols).
    solution_candidates = sympy.solve(sympify_distribution(diff), invariant.get_parameters(), dict=True) \
        if invariant.get_parameters() else []
    print(f"Solution candidates: {solution_candidates}")
    solutions = []

    # Filter only solutions which are valid for the invariant.
    for candidate in solution_candidates:
        for _, val in candidate.items():
            if not {str(s) for s in val.free_symbols} <= invariant.get_parameters():
                break
        else:
            if not all(map(lambda x: x == 0, candidate.values())):
                solutions.append(candidate)

    # The solver's solutions are only proposals: keep those that make the invariant a proven fixed point.
    variables = prog_info.program.variables.keys()
    verified = []
    for candidate in solutions:
        full = str(sympify_distribution(invariant).subs(candidate)).replace("**", "^")
        full_dist = config.factory.from_expr(full, *variables)
        if is_evt_fixed_point(loop, prog_info, distribution, full_dist, config, analyzer):
            verified.append((candidate, full_dist))
        else:
            logger.info("Discarding the solution %s: the fixed-point equation is not proven.", candidate)

    if len(verified) > 0:
        print(f"All solutions: {[candidate for candidate, _ in verified]}")
        logger.info("Using the first solution to continue.")
        # Use the first solution to continue.
        # This is a heuristic, as there might be multiple solutions.
        full_dist = verified[0][1]
        sol_dist = full_dist - full_dist.filter(loop.cond)
        print(f"{Style.GREEN}Invariant: {sol_dist}{Style.RESET}{Style.CLEARTOEND}")
        return sol_dist, full_dist, [{str(k): str(v) for k, v in candidate.items()} for candidate, _ in verified]

    # No solution found. A proven difference refutes the invariant only if it has no parameters, since the
    # solver's search for parameter values is incomplete.
    difference = str(phi - invariant)
    if verdict is False and not invariant.get_parameters():
        print(f"{Style.OKRED}Invariant refuted: Phi(I) - I = {difference} is not zero.{Style.RESET}")
        raise FixedPointCheckFailed(f"Invariant {invariant} is refuted for loop {loop}: it is not a fixed point, "
                                    f"Φ(I) - I = {difference}.", "refuted", difference)
    reason = "no parameter values were found that make it a fixed point" if invariant.get_parameters() \
        else "the fixed-point equation I = Φ(I) could be neither proven nor refuted"
    print(f"{Style.OKRED}Invariant not proven: {reason}.{Style.RESET}")
    raise FixedPointCheckFailed(f"Invariant {invariant} could not be verified for loop {loop}: {reason}; "
                                "it may still be correct.", "unknown", difference)


def evt_invariant_verification(loop: WhileInstr,
                               prog_info: ProgramInfo,
                               distribution: Distribution,
                               invariant: Distribution,
                               config: ForwardAnalysisConfig,
                               analyzer: Analyzer) -> Distribution:
    """Like :func:`evt_invariant_check`, but returns only the terminating part."""
    return evt_invariant_check(loop, prog_info, distribution, invariant, config, analyzer)[0]
