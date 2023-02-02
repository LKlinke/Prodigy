from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Set, Tuple

import sympy
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

        .. returns: A new program object equivalent to one loop unrolling of :param: program.
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
                   instructions=[guarded_instr],
                   functions=invariant.functions)


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


def check_equivalence(
    program: Program, invariant: Program, config: ForwardAnalysisConfig
) -> Tuple[Literal[True], List[Dict[str, str]]] | Tuple[
        Literal[False], State] | Tuple[None, Distribution]:
    """
    This method uses the fact that we can sometimes determine program equivalence,
    by checking the equality of two parametrized infinite-state Distributions.

    If they are equivalent, also returns a list of constraints under which this holds
    (which may be empty if they are always equal). If not, also returns a counterexample.
    If the solution cannot be determined (returns `None`), also returns the difference of
    the second order distributions generated by the invariant and the once unrolled while
    loop of the program. If this difference can be made equal to 0, the programs are
    equivalent.
    .. param config: The configuration.
    .. param program: The While-Loop program
    .. param invariant: The loop-free invariant
    .. returns: Whether the invariant and the program are equivalent.
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
    return _compute_result(
        inv_result, modified_inv_result, config, new_vars,
        program.parameters.keys() | invariant.parameters.keys())


def _compute_result(
    inv_result: Distribution, modified_inv_result: Distribution,
    config: ForwardAnalysisConfig, new_vars: Dict[str, str], params: Set[str]
) -> Tuple[Literal[True], List[Dict[str, str]]] | Tuple[
        Literal[False], State] | Tuple[None, Distribution]:
    """Evaluates whether the invariant result and the modified invariant result are equal / unifiable"""

    logger.debug("Compare results")
    if config.show_intermediate_steps:
        print(
            f"\n{Style.YELLOW} Compare the results. {Style.RESET} \n {modified_inv_result}\n==\n{inv_result}"
        )

    if modified_inv_result == inv_result:
        logger.debug("Invariant validated.")
        empty: List[Dict[str, str]] = []  # Necessary to satisfy mypy
        return True, empty

    # The results are different, so we check if it's possible to unify them
    logger.debug("Invariant could not be validated.")
    diff = (modified_inv_result - inv_result).set_variables(*new_vars.keys())

    if len(params & diff.get_symbols()) == 0:
        # There are no parameters, so the results can't be unified
        count_ex, _ = diff.get_state()
        res = {}
        for var in count_ex:
            res[new_vars[var]] = count_ex[var]
        logger.debug("Found counterexample: %s", res)
        return False, State(res)

    # First we let sympy try to find a solution. We are only interested
    # in solutions that depend on nothing but parameters
    unify = _solve(sympy.S(str(diff)), params)
    if len(unify) > 0:
        logger.debug(
            "Found constraints under which the invariant can be validated: %s",
            unify)
        return True, [{str(var): str(val)
                       for var, val in x.items()} for x in unify]

    # If we don't find a solution, we try to prove that there is none by comparing coefficients
    logger.debug(
        "Could not find a solution, trying to prove that there is none")
    if config.show_intermediate_steps:
        print(
            f'{Style.YELLOW}Could not find a solution, trying to prove that there is none...{Style.RESET}'
        )
    inv_result_params = inv_result.set_variables(*new_vars.keys())
    modified_inv_result_params = modified_inv_result.set_variables(
        *new_vars.keys())
    count = 0
    finite = diff.is_finite()
    threshold = 1000  # TODO find a good value for the threshold
    # There needs to be at least one solution that can be used to make ALL coefficients equal to 0
    # It doesn't suffice if each coefficient can be made equal to 0 but with different parameter assignments
    all_solutions = None
    for _, state in diff:
        count += 1
        if not finite and count > threshold:
            break

        mass_diff = sympy.S(
            str(
                modified_inv_result_params.filter_state(
                    state).get_probability_mass())
        ) - sympy.S(
            str(inv_result_params.filter_state(state).get_probability_mass()))
        syms = {str(s) for s in mass_diff.free_symbols} & params

        if len(syms) > 0:
            # We are again only interested in solutions that depend wholly on parameters
            sol = _solve(mass_diff, syms)

            # We are only interested in solutions that can also be used for all other coefficients
            if all_solutions is None:
                all_solutions = sol
            else:
                for s in all_solutions.copy():
                    if s not in sol:
                        all_solutions.remove(s)

            if len(all_solutions) == 0:
                # If there are coefficients that cannot be unified, we found a counterexample
                # Here we assume that sympy would have found a solution if there were one
                logger.debug("Found coefficients that cannot be unified: %s",
                             state.valuations)
                if config.show_intermediate_steps:
                    print()
                return False, state
        elif mass_diff != 0:
            # If there are no symbols and the difference of the coefficients isn't 0, there is no solution
            if config.show_intermediate_steps:
                print()
            return False, state

    if config.show_intermediate_steps:
        print()
    # We couldn't prove that the results can be unified, but we also failed to find a counterexample
    return None, diff


def _solve(expr: Any, params: Set[str]) -> List[Dict[Any, Any]]:
    """
    Solves a sympy expression for 0 and filters out any solutions that contain
    a symbol that is not in the provided parameters
    """

    assert len(params) > 0
    sol = []
    for el in sympy.solve(expr, *params, dict=True):
        for _, val in el.items():
            if not {str(s) for s in val.free_symbols} <= params:
                break
        else:
            sol.append(el)

    return sol
