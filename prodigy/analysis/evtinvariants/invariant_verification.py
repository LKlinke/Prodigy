import logging
from typing import Callable, Union, Sequence

import sympy


from probably.pgcl import Instr, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution.distribution import Distribution
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

def evt_invariant_verification(loop: WhileInstr,
                               prog_info: ProgramInfo,
                               distribution: Distribution,
                               invariant: Distribution,
                               config: ForwardAnalysisConfig,
                               analyzer: Callable[
                                            [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                                        tuple[Distribution, Distribution]
                                        ] = None) -> Distribution:
    logger.debug("Invariant Verification for loop %s with initial distribution %s.", loop, distribution)
    print(f"{Style.YELLOW}Invariant verification initiated...{Style.RESET}")
    
    zero_dist = config.factory.from_expr("0", *prog_info.program.variables.keys())
    
    # Compute one iteration step.
    phi = distribution + \
              analyzer(loop.body, prog_info, invariant.filter(loop.cond), zero_dist, config)[0]
    
    
    # Validate the invariant.
    logger.debug("Trying to validate invariant: %s", invariant)
    logger.debug("Phi(inv) = %s", phi)
    if invariant == phi:
        print(f"{Style.OKGREEN}Invariant validated!{Style.RESET}")
        print(f"{Style.GREEN}Invariant: {invariant}{Style.RESET}{Style.CLEARTOEND}")
        return invariant - invariant.filter(loop.cond)
    
    # If the invariant does not match, we need to find a solution.
    diff = invariant - phi
    solution_candidates = sympy.solve(sympy.S(str(diff)), invariant.get_parameters(), dict=True)
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

    if len(solutions) > 0:
        print(f"All solutions: {solutions}")
        logger.info("Using the first solution to continue.")
        # Use the first solution to continue.
        # This is a heuristic, as there might be multiple solutions.
        solution = str(sympy.S(str(invariant - invariant.filter(loop.cond))).subs(solutions[0])).replace("**", "^")
        sol_dist = config.factory.from_expr(
            solution, *prog_info.program.variables.keys()
        )
        print(f"{Style.GREEN}Invariant: {sol_dist}{Style.RESET}{Style.CLEARTOEND}")
        return sol_dist
    
    # No solution found.
    raise VerificationError(f"Invariant {invariant} could not be verified for loop {loop}.")