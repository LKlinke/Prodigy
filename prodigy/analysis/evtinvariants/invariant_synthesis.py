import logging
from typing import List, Callable, Union, Sequence, Tuple

import sympy
from probably.pgcl import WhileInstr, Instr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategy
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


def evt_invariant_synthesis(loop: WhileInstr,
                            prog_info: ProgramInfo,
                            distribution: Distribution,
                            config: ForwardAnalysisConfig,
                            strategy: SynthesisStrategy,
                            analyzer: Callable[
                                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution,
                                 ForwardAnalysisConfig],
                                Tuple[Distribution, Distribution]
                            ]
                            ) -> List[Distribution]:
    logger.debug("Invariant Synthesis for loop %s with initial distribution %s.", loop, distribution)
    zero_dist: Distribution = config.factory.from_expr("0", *prog_info.program.variables)

    # enumerate potential candidates given by a heuristic
    for evt_candidate in strategy.template_heuristics.generate():
        print(f"{Style.YELLOW}Invariant candidate: {evt_candidate}{Style.RESET}")
        # Compute one iteration step.
        evt_inv = evt_candidate
        phi_inv = distribution + analyzer(loop.body,
                                          prog_info,
                                          evt_inv.filter(loop.cond),
                                          zero_dist,
                                          config)[0]
        # Check equality between the iterated expression and the invariant.
        # If they are equal we are done.
        logger.debug("Check Invariant candidate %s", evt_inv)
        if evt_inv == phi_inv:
            logger.debug("Candidate validated.")
            return [evt_inv]

        # otherwise we compute the difference and try to solve for the coefficients
        diff = evt_inv - phi_inv
        coeffs = [sympy.S(p) for p in diff.get_parameters()]
        variables = [sympy.S(v) for v in diff.get_variables()]
        logger.debug("Start solving equation systems for %s", diff)
        solution_candidates = sympy.solve_undetermined_coeffs(sympy.S(str(diff)), coeffs, variables, particular=True)

        # Cleanup of messy sympy output conventions.
        # We now always get back a list of solutions which might be empty.
        if not isinstance(solution_candidates, list):
            solution_candidates = [solution_candidates]
        logger.debug("Filter solutions in: %s", solution_candidates)

        # Some solutions might not be of interest, as they are not purely in terms of our template parameters
        # We filter these out. Also, "all zero" solutions are excluded.
        solutions = []
        for candidate in solution_candidates:
            for _, val in candidate.items():
                if not {str(s) for s in val.free_symbols} <= evt_inv.get_parameters():
                    break
            else:
                if all(map(lambda x: x == 0, candidate.values())):
                    continue

                numerator, denominator = sympy.S(str(diff)).as_numer_denom()
                # we have excluded all zero solutions and now also exclude the solutions
                # which make the denominator zero
                if denominator.subs(candidate).equals(0):
                    continue
                solutions.append(candidate)

        # In case there are still some solutions we check them for actual solutions in the FPS domain with non-negative
        # coefficients. This is in general a hard problem (not known to be decidable), thus we use heuristics.
        if len(solutions) > 0:
            solutions_and_kinds = [
                (sol, strategy.positivity_heuristics.is_positive(sympy.S(str(evt_inv)).subs(sol).ratsimp()))
                for sol in solutions
            ]
            contains_only_bad_invariants = True
            for sol, kind in solutions_and_kinds:
                logger.debug("Possible solution: %s", sol)
                if kind is False and not config.show_all_invs:
                    continue
                message = {None: f"{Style.YELLOW}Possible invariant{Style.RESET}",
                           True: f"{Style.GREEN}Invariant{Style.RESET}",
                           False: f"{Style.RED}Spurious invariant{Style.RESET}"}
                print(f"{message[kind]}: {sympy.S(str(evt_inv)).subs(sol).ratsimp()}")
                contains_only_bad_invariants &= False if kind is None or kind else True

            # Just return if we know that not all invariants are spurious.
            if not contains_only_bad_invariants:
                return [x for (x, y) in filter(lambda x: x[1] is None or x[1], solutions_and_kinds)]

    # We were unable to determine some EVT Invariant using the given heuristic.
    raise VerificationError(
        f"Could not find a rational function inductive invariant using strategy {strategy}")
