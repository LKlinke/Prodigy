import logging
from typing import List, Callable, Union, Sequence, Tuple

import sympy
from probably.pgcl import WhileInstr, Instr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategy
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.solver.solver_type import SolverType
from prodigy.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


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
    print(f"{Style.YELLOW}Invariant synthesis initiated...{Style.RESET}")
    # enumerate potential candidates given by a heuristic
    for evt_candidate in strategy.template_heuristics.generate():
        print(f"{Style.YELLOW}Invariant candidate: {evt_candidate}{Style.RESET}{Style.CLEARTOEND}", end="\r")
        # Compute one iteration step.
        evt_inv = evt_candidate
        one_step_dist, one_step_err = analyzer(loop.body, prog_info, evt_inv.filter(loop.cond), zero_dist, config)
        if one_step_err != config.factory.from_expr("0", *prog_info.program.variables):
            raise NotImplementedError("Currently invariant synthesis does not support conditioned distributions.")
        phi_inv = distribution + one_step_dist

        if config.solver_type == SolverType.Z3:
            solver = SolverType.make(config.solver_type, config.factory)
        else:
            solver = SolverType.make(config.solver_type)
        # Check equality between the iterated expression and the invariant.
        logger.debug("Check Invariant candidate %s", evt_inv)
        is_solution, solution_candidates = solver.solve(evt_inv, phi_inv)
        if is_solution is False or is_solution is None:
            continue
        logger.debug("Filter solutions in: %s", solution_candidates)

        # Exclude "all zero" solutions, as well as solutions which make the denominator 0.
        solutions = []
        for candidate in solution_candidates:
            if all(map(lambda x: x == 0, candidate.values())):
                continue

            _, denominator = sympy.S(str(evt_inv - phi_inv)).as_numer_denom()
            # we have excluded all zero solutions and now also exclude the solutions
            # which make the denominator zero
            if denominator.subs(candidate).equals(0):
                continue

            if candidate not in solutions:
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
        f"Could not find a rational function inductive invariant using {strategy}")
