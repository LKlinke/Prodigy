from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Literal, Tuple, Callable, Sequence, Union

import sympy
from prodigy.pgcl import (BoolType, Program, VarExpr,
                           Instr)

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.solver.solver_type import SolverType
from prodigy.distribution.distribution import Distribution, NonFiniteMarginalError, State
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)

Analyzer = Callable[
    [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
    Tuple[Distribution, Distribution]
]
"""The analysis that computes the output of each program (``compute_semantics`` or a loop task's)."""

EquivalenceAnswer = Union[Tuple[Literal[True], List[Dict[str, str]]], Tuple[Literal[False], State],
                          Tuple[None, Distribution]]
"""The raw answer of :func:`check_equivalence`, before any verdict."""


class DifferentVariablesError(ValueError):
    """The two programs of an equivalence check declare different variables, so their output
    distributions cannot be compared state by state. This says nothing about whether they are
    equivalent, so it is not a "not equivalent" (#395)."""


class CheckNotImplemented(NotImplementedError):
    """A step of the second-order check needs a finite measure (e.g. a marginal), but the test
    distribution stands for every input at once and has infinite mass by design: a limitation of the
    check, not a property of the programs (#395)."""


def different_variables(program: Program, other: Program) -> str | None:
    """Why two programs cannot be compared state by state (their variables differ), or None."""
    first, second = set(program.variables), set(other.variables)
    if first == second:
        return None
    parts = []
    if first - second:
        parts.append(f"only program A declares {', '.join(sorted(first - second))}")
    if second - first:
        parts.append(f"only program B declares {', '.join(sorted(second - first))}")
    return ("The programs declare different variables (" + "; ".join(parts) + "). The comparison needs the "
            "same variables in both programs; declare the missing ones (they keep the value 0).")


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
        if isinstance(program.variables[variable], BoolType):
            dist *= config.factory.from_expr(f"(1+{new_var} * {variable})",
                                             VarExpr(var=new_var),
                                             VarExpr(var=variable))
        else:
            dist *= config.factory.from_expr(f"1/(1-{new_var}*{variable})",
                                             VarExpr(var=new_var),
                                             VarExpr(var=variable))
        so_vars[new_var] = variable
    logger.debug("Test distribution is %s with second order variables %s",
                 dist, so_vars)
    return dist.set_variables(*program.variables.keys(),
                              *so_vars.keys()).set_parameters(), so_vars


def check_equivalence(
        program: Program,
        other_program: Program,
        config: ForwardAnalysisConfig,
        analyzer: Analyzer
) -> EquivalenceAnswer:
    """
    Compares the second-order generating functions of two programs: both run on a test distribution
    that stands for every input at once, and a solver decides whether the two outputs are equal.

    The answer is *raw*: it says what holds for the two computed distributions, not for the programs.
    If an analysis inside the check is only a bound (Park's lemma for an invariant, unrolling), equality
    proves one direction only, and "different" proves nothing. A sound verdict needs the guarantees of the
    two analyses and a witness; use :func:`prodigy.analysis.equivalence.verdict.compare` (or
    :func:`~prodigy.analysis.equivalence.verdict.run_check` and
    :func:`~prodigy.analysis.equivalence.verdict.decide`) for it.

    * ``(True, constraints)``: the computed distributions are equal, under the parameter values of
      ``constraints`` (empty if always).
    * ``(False, state)``: the solver found no way to make them equal; ``state`` is an input on which they
      differ (a candidate for a witness, no proof).
    * ``(None, difference)``: undecided; ``difference`` is the difference of the two outputs.

    :raises DifferentVariablesError: if the programs declare different variables.
    :raises CheckNotImplemented: if a step of the check needs a finite measure (a limitation of the check).
    :param program: The first program
    :param other_program: The second program
    :param config: The configuration.
    :param analyzer: The analyzer.
    """

    logger.debug("Checking equivalence.")

    # Programs with different variables cannot be compared state by state; that is no refutation.
    differ = different_variables(program, other_program)
    if differ is not None:
        logger.info("The two programs have different variable sets, prog: %s, other: %s",
                    program.variables.keys(), other_program.variables.keys())
        raise DifferentVariablesError(differ)
    try:
        return _check_equivalence(program, other_program, config, analyzer)
    except NonFiniteMarginalError as exc:
        raise CheckNotImplemented(
            "a step of the check for every input needs a finite measure, but the check works on a generating "
            f"function with infinite mass ({exc})") from exc


def _check_equivalence(
        program: Program,
        other_program: Program,
        config: ForwardAnalysisConfig,
        analyzer: Analyzer
) -> EquivalenceAnswer:
    # Now we have to generate an infinite state parametrized distribution for every program variable.
    test_dist, new_vars = generate_equivalence_test_distribution(program, config)
    # The parameters of both programs are unknown constants of the test distribution: an engine that
    # takes the symbols of a distribution for its variables (GiNaC) needs them, e.g. for `iid` (#403).
    params = program.parameters.keys() | other_program.parameters.keys()
    test_dist = test_dist.set_parameters(*sorted(params))

    # Compute the resulting distributions for both programs
    logger.debug("Compute the other programs posterior...")
    other_program_posterior, other_program_error = analyzer(
        other_program.instructions,
        ProgramInfo(other_program, so_vars=frozenset(new_vars.keys())),
        test_dist,
        config.factory.from_expr("0", *(other_program.variables | new_vars.keys())),
        config
    )
    logger.debug("other programs result:\n%s", other_program_posterior)
    logger.debug("Compute the posterior of the program...")

    if config.show_intermediate_steps:
        print(f"\n{Style.YELLOW} Compute the posterior of the program. {Style.RESET}")
    program_posterior, program_error = analyzer(
        program.instructions,
        ProgramInfo(program, so_vars=frozenset(new_vars.keys())),
        test_dist,
        config.factory.one(*(program.variables | new_vars.keys())) * 0,
        config
    )
    logger.debug("program result:\n%s", program_posterior)

    diff = program_posterior - other_program_posterior

    # Compare them and check whether they are equal.
    solver = SolverType.make(config.solver_type)
    dist_is_solution, dist_candidates = solver.solve(program_posterior.set_parameters(*params),
                                                     other_program_posterior.set_parameters(*params))
    err_is_solution, err_candidates = solver.solve(program_error.set_parameters(*params),
                                                   other_program_error.set_parameters(*params))

    # If there is no solution tell the user why.
    if (dist_is_solution and err_is_solution) is False:
        logger.debug("Equivalence refuted.")
        state, _ = diff.get_state()
        return False, State({var: state[sym] for sym, var in new_vars.items()})

    # If there is a solution, individually, see whether they match.
    if (dist_is_solution and err_is_solution) is True:
        if not dist_candidates:
            logger.debug("Posterior equivalence validated.")
            return True, err_candidates
        if not err_candidates:
            logger.debug("Posterior error validated.")
            return True, dist_candidates
        logger.debug("Matching individual solutions\n%s\n%s", dist_candidates, err_candidates)
        res_both = []
        for dist_sol, err_sol in itertools.product(dist_candidates, err_candidates):
            to_be_solved = []
            for var, constr in dist_sol.items() | err_sol.items():
                to_be_solved.append(var - constr)
            logger.debug("Solving the equation system: %s", to_be_solved)
            res_both += sympy.solve(
                to_be_solved, dict=True) if len(to_be_solved) > 1 else sympy.solve(
                to_be_solved[0], dict=True)
            logger.debug("Current partial solution space: %s", res_both)
        if not res_both:
            # Each of the two equations has solutions, but no parameter values solve both. That is no
            # counterexample (there is no input to show), so the answer stays open (#395).
            logger.debug("No common solution of the distribution and error constraints.")
            return None, diff
        logger.debug("Equivalence validated.")
        return True, res_both
    logger.debug("Equivalence unknown.")
    return None, diff
