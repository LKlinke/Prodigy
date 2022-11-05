from __future__ import annotations

import logging
from typing import Dict, Tuple

from probably.pgcl import IfInstr, Program, SkipInstr, VarExpr, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instruction_handler import compute_discrete_distribution
from prodigy.distribution.distribution import Distribution, State
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


def phi(program: Program, invariant: Program) -> Program:
    """
        The characteristic loop functional. It unrolls a loop exactly once.

        .. returns: A new program object equivaelnt to one loop unrolling of :param: program.
    """
    assert isinstance(
        program.instructions[0],
        WhileInstr), "Program can only be one big loop to analyze."
    logger.debug("Create modified invariant program.")
    new_instructions = program.instructions[0].body.copy()

    for instr in invariant.instructions:
        new_instructions.append(instr)

    guarded_instr = IfInstr(cond=program.instructions[0].cond,
                            true=new_instructions,
                            false=[SkipInstr()])

    return Program(declarations=invariant.declarations,
                   variables=invariant.variables,
                   constants=invariant.constants,
                   parameters=invariant.parameters,
                   instructions=[guarded_instr])


def generate_equivalence_test_distribution(
        program: Program,
        config: ForwardAnalysisConfig) -> Tuple[Distribution, Dict[str, str]]:
    """
        Generates a second-order PGF, dependent on the given variables in a program. This SOP can be used to check
        equivalences of two programs.

        .. returns: The SOP expression.
    """
    logger.debug("Generating test distribution.")
    dist = config.factory.one()
    so_vars: Dict[str, str] = {}  # second order variables
    for variable in program.variables:
        new_var = dist.get_fresh_variable()
        dist *= config.factory.from_expr(f"1/(1-{new_var}*{variable})",
                                         VarExpr(var=new_var),
                                         VarExpr(var=variable))
        so_vars[new_var] = variable
    return dist.set_variables(*program.variables.keys(),
                              *so_vars.keys()).set_parameters(), so_vars


def check_equivalence(program: Program, invariant: Program, config: ForwardAnalysisConfig) \
        -> Tuple[bool, Distribution | State]:
    """
    This method uses the fact that we can sometimes determine program equivalence,
    by checking the equality of two parametrized infinite-state Distributions.
    .. param config: The configuration.
    .. param program: The While-Loop program
    .. param invariant: The loop-free invariant
    .. returns: Whether the invariant and the program are equivalent, and either the
        resulting distribution or a state that represents a counterexample demonstrating
        why the given invariant is invalid
    """

    logger.debug("Checking equivalence.")
    # First we create the modified input program in order to fit the premise of Park's Lemma
    if config.show_intermediate_steps:
        print(
            f"{Style.YELLOW} Generate modified invariant program. {Style.RESET}"
        )
    modified_inv = phi(program, invariant)

    # Now we have to generate a infinite state parametrized distribution for every program variable.
    if config.show_intermediate_steps:
        print(
            f"{Style.YELLOW} Generate second order generating function. {Style.RESET}"
        )
    test_dist, new_vars = generate_equivalence_test_distribution(
        program, config)

    # Compute the resulting distributions for both programs
    logger.debug("Compute the modified invariant...")
    if config.show_intermediate_steps:
        print(
            f"\n{Style.YELLOW} Compute the result of the modified invariant. {Style.RESET}"
        )
    modified_inv_result = compute_discrete_distribution(
        modified_inv, test_dist, config)
    logger.debug("modified invariant result:\t%s", modified_inv_result)
    logger.debug("Compute the invariant...")
    if config.show_intermediate_steps:
        print(
            f"\n{Style.YELLOW} Compute the result of the invariant. {Style.RESET}"
        )
    inv_result = compute_discrete_distribution(invariant, test_dist, config)
    logger.debug("invariant result:\t%s", inv_result)
    # Compare them and check whether they are equal.
    logger.debug("Compare results")
    if config.show_intermediate_steps:
        print(
            f"\n{Style.YELLOW} Compare the results. {Style.RESET} \n {modified_inv_result} == {inv_result}"
        )
    if modified_inv_result == inv_result:
        logger.debug("Invariant validated.")
        return True, inv_result
    else:
        logger.debug("Invariant could not be validated.")
        count_ex, _ = (modified_inv_result -
                       inv_result).set_variables(*new_vars.keys()).get_state()
        res = {}
        for var in count_ex:
            res[new_vars[var]] = count_ex[var]
        return False, State(res)
