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
                            ) -> List[Tuple[Distribution, Union[bool, None]]]:
    logger.debug("Invariant Synthesis for loop %s with initial distribution %s.", loop, distribution)
    zero_dist: Distribution = config.factory.from_expr("0", *prog_info.program.variables)

    # enumerate potential candidates given by a heuristic
    for evt_candidate in strategy.template_heuristics.generate():
        print(f"{Style.YELLOW}Invariant candidate: {evt_candidate}{Style.RESET}{Style.CLEARTOEND}", end="\r")
        # Compute one iteration step.
        evt_inv = evt_candidate
        one_step_dist, one_step_err = analyzer(loop.body, prog_info, evt_inv.filter(loop.cond), zero_dist, config)
        if one_step_err != config.factory.from_expr("0", *prog_info.program.variables):
            raise NotImplementedError(f"Currently invariant synthesis does not support conditioned distributions.")
        phi_inv = distribution + one_step_dist

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

            result: List[Tuple[Distribution, Union[bool, None]]] = []
            inv_found = False

            for sol in solutions:
                sol_inv = sympy.S(str(evt_inv)).subs(sol).ratsimp()
                sol_is_positive = strategy.positivity_heuristics.is_positive(sol_inv)

                logger.debug("Possible solution: %s", sol)
                if sol_is_positive is None:
                    print(f"{Style.CYAN}Possible invariant: {sol_inv}{Style.RESET}{Style.CLEARTOEND}")
                    sol_inv = config.factory.from_expr(str(sol_inv).replace("**", "^"), *prog_info.program.variables)
                    result.append((sol_inv, sol_is_positive))
                elif sol_is_positive is True:
                    print(f"{Style.GREEN}Invariant: {sol_inv}{Style.RESET}{Style.CLEARTOEND}")
                    sol_inv = config.factory.from_expr(str(sol_inv).replace("**", "^"), *prog_info.program.variables)
                    result.append((sol_inv, sol_is_positive))
                    inv_found = True
                else:
                    if config.show_all_invs:
                        print(f"{Style.RED}Spurious invariant: {sol_inv}{Style.RESET}{Style.CLEARTOEND}")

            # Just return if we know that not all invariants are spurious.
            if inv_found:
                return result

    # We were unable to determine some EVT Invariant using the given heuristic.
    raise VerificationError(
        f"Could not find a rational function inductive invariant using strategy {strategy}")
