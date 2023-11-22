from __future__ import annotations

import itertools
import logging
import sys
from fractions import Fraction
from typing import Iterator, Optional, Callable, Union, Sequence

import sympy
from probably.pgcl import Instr, parse_pgcl, Program, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.exceptions import VerificationError
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution, State
from prodigy.distribution.generating_function import SympyPGF
from prodigy.util.color import Style
from prodigy.util.logger import print_progress_bar, log_setup

logger = log_setup(__name__, logging.DEBUG)


class WhileHandler(InstructionHandler):
    @staticmethod
    def _analyze_with_invariant(
            instruction: Instr, prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        inv_filepath = input("Invariant file:\t")
        with open(inv_filepath, 'r', encoding="utf-8") as inv_file:
            inv_src = inv_file.read()
            inv_prog = parse_pgcl(inv_src)

            prog = Program(declarations=[],
                           variables=prog_info.variables,
                           constants=prog_info.constants,
                           parameters=prog_info.parameters,
                           instructions=[instruction],
                           functions=prog_info.functions)
            print(f"{Style.YELLOW}Verifying invariant...{Style.RESET}")
            answer, result = check_equivalence(
                prog, inv_prog, config, analyzer)
            if answer:
                assert isinstance(result, list)
                if len(result) == 0:
                    print(Style.OKGREEN + "Invariant successfully validated!" +
                          Style.RESET)
                else:
                    print(
                        f"{Style.OKGREEN}Invariant validated under the following constraints:{Style.RESET} {result}"
                    )
                if config.show_intermediate_steps:
                    print(Style.YELLOW +
                          "Compute the result using the invariant" +
                          Style.RESET)
                return analyzer(inv_prog.instructions, prog_info, distribution, error_prob, config)
            elif answer is False:
                assert isinstance(result, State)
                print(
                    f'{Style.OKRED}Invariant could not be verified.{Style.RESET} Counterexample: {result.valuations}'
                )
                raise VerificationError(
                    "Invariant could not be determined as such.")

            else:
                assert answer is None
                assert isinstance(result, Distribution)
                print(
                    f'{Style.OKRED}Could not determine whether invariant is valid.{Style.RESET} Phi(I) - I: {result}'
                )
                raise VerificationError(
                    "Could not determine whether invariant is valid.")

    @staticmethod
    def _compute_iterations(
            instruction: Instr, prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        max_iter = int(input("Specify a maximum iteration limit: "))
        sat_part = distribution.filter(instruction.cond)
        non_sat_part = distribution - sat_part
        for i in range(max_iter):
            print_progress_bar(i + 1, max_iter, length=50)
            iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
            iterated_sat = iterated_part.filter(instruction.cond)
            iterated_non_sat = iterated_part - iterated_sat
            if iterated_non_sat == SympyPGF.zero(
            ) and iterated_sat == sat_part:
                print(
                    f"\n{Style.OKGREEN}Terminated already after {i + 1} step(s)!{Style.RESET}"
                )
                break
            non_sat_part += iterated_non_sat
            sat_part = iterated_sat
        return non_sat_part, error_prob

    @staticmethod
    def _compute_until_threshold(
            instruction: Instr, prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        captured_probability_threshold = float(
            input("Enter the probability threshold: "))
        sat_part = distribution.filter(instruction.cond)
        non_sat_part = distribution - sat_part
        captured_part = non_sat_part + error_prob
        while Fraction(captured_part.get_probability_mass()
                       ) < captured_probability_threshold:
            logger.info("Collected %f of the desired mass", (float(
                (Fraction(captured_part.get_probability_mass()) /
                 captured_probability_threshold)) * 100))
            iterated_part, error_prob = analyzer(instruction.body, prog_info, sat_part, error_prob, config)
            iterated_sat = iterated_part.filter(instruction.cond)
            iterated_non_sat = iterated_part - iterated_sat
            non_sat_part += iterated_non_sat
            captured_part = non_sat_part + error_prob
            sat_part = iterated_sat
            print_progress_bar(int(
                (Fraction(captured_part.get_probability_mass()) /
                 captured_probability_threshold) * 100),
                100,
                length=50)
        return non_sat_part, error_prob

    @staticmethod
    def _approx_expected_visiting_times(
            instruction: Instr,
            prog_info: ProgramInfo,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        assert error_prob.is_zero_dist(), f"Currently EVT reasoning does not support conditioning."
        max_iter = int(input("Enter the number of iterations: "))
        evt = distribution * 0
        for i in range(max_iter):
            print_progress_bar(i + 1, max_iter, length=50)
            evt = distribution + \
                  analyzer(instruction.body, prog_info, evt.filter(instruction.cond), error_prob, config)[0]
        print(evt)
        return (evt - evt.filter(instruction.cond)), error_prob

    @staticmethod
    def _evt_invariant(
            instruction: Instr,
            prog_info: ProgramInfo,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        assert error_prob.is_zero_dist(), f"Currently EVT reasoning does not support conditioning."
        evt_inv = config.factory.from_expr(input("Enter EVT invariant: "), *prog_info.program.variables.keys())
        phi = distribution + \
              analyzer(instruction.body, prog_info, evt_inv.filter(instruction.cond), error_prob, config)[0]

        if evt_inv == phi:
            print(f"{Style.OKGREEN}Invariant validated!{Style.RESET}")
            return evt_inv - evt_inv.filter(instruction.cond), error_prob
        diff = evt_inv - phi
        solution_candidates = sympy.solve(sympy.S(str(diff)), evt_inv.get_parameters(), dict=True)
        print(f"Solution candidates: {solution_candidates}")
        solutions = []
        for candidate in solution_candidates:
            for _, val in candidate.items():
                if not {str(s) for s in val.free_symbols} <= evt_inv.get_parameters():
                    break
            else:
                if not all(map(lambda x: x == 0, candidate.values())):
                    solutions.append(candidate)
        if len(solutions) > 0:
            print(f"All solutions: {solutions}")
            # TODO use a solution to compute the final distribution.
            return evt_inv, error_prob

        raise VerificationError(f"Could not validate the EVT invariant {evt_inv}")

    @staticmethod
    def _evt_invariant_synthesis(
            instruction: Instr,
            prog_info: ProgramInfo,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:

        logger.debug("Using invariant synthesis.")

        def _make_clause(c, variables, powers) -> str:
            c = [f"s_{c}"]
            vp = (['', '{}', '({}^{})'][min(p, 2)].format(v, p) for v, p in zip(variables, powers))
            return '*'.join(c + [s for s in vp if s])

        def _generate_rational_function_stepwise(max_deg: int) -> Iterator[Distribution]:

            for max_powers_denom in sympy.utilities.iterables.iproduct(
                    *[range(max_deg + 1) for _ in prog_info.program.variables]):
                degrees = [range(max_powers_denom[i] + 1) for i, _ in enumerate(prog_info.program.variables.keys())]
                denominator = " + ".join(
                    (_make_clause(f"d_{c}", prog_info.program.variables.keys(), pows) for c, pows in
                     enumerate(itertools.product(*degrees))))
                for max_powers_num in sympy.utilities.iterables.iproduct(
                        *[range(sum(max_powers_denom) + 1) for _ in prog_info.program.variables]):
                    if sum(max_powers_num) > sum(max_powers_denom):
                        continue
                    num_degrees = [range(max_powers_num[i] + 1) for i, _ in
                                   enumerate(prog_info.program.variables.keys())]
                    numerator = " + ".join((_make_clause(c, prog_info.program.variables.keys(), pows) for c, pows in
                                            enumerate(itertools.product(*num_degrees))))
                    yield config.factory.from_expr(f"({denominator}) / ({numerator})",
                                                   *prog_info.program.variables.keys())

        def _check_nonnegative(f: sympy.Expr) -> Optional[bool]:
            """
            Checks heuristically whether a univariate rational function _f_ has a non-negative power series
            expansion.

            ..returns:
                - None: We do not know
                - True: It has a non-negative power series expansion
                - False: There exists at least one negative coefficient in the power series expansion
            """
            logger.debug("check non-negativity for %s", f)

            # TODO  f might be a sum of rational functions (partial fraction decomposition).
            if isinstance(f, sympy.Add):
                sub_results = [_check_nonnegative(rf) for rf in f.args]
                for res in sub_results:
                    if res is None:
                        return None
                return all(sub_results)

            # Getting f as numerator/denominator and check whether it is indeed a rational function.
            numerator, denominator = f.as_numer_denom()

            # Check whether f is univariate (others not supported currently)
            if len(denominator.free_symbols) > 1:
                logger.info("Multivariate functions like %s are currently not supported", f)
                return None

            logger.debug("\nnumerator (non-poly) %s\ndenominator (non-poly) %s", numerator, denominator)
            if not (numerator.is_polynomial() and denominator.is_polynomial()):
                logger.info("%s is not a rational function.", f)
                return None

            # convert numerator and denominator into polynomial objects
            # as_poly() fails converting constant polynomials without specified variables
            # so we give it a _DUMMY_ variable.
            maybe_numerator = numerator.as_poly(*numerator.free_symbols)
            numerator = maybe_numerator if maybe_numerator else numerator.as_poly(sympy.S("DUMMY"))
            maybe_denominator = denominator.as_poly(*denominator.free_symbols)
            denominator = maybe_denominator if maybe_denominator else denominator.as_poly(sympy.S("DUMMY"))
            logger.debug("\nnumerator %s\ndenominator %s", numerator, denominator)

            # Check the coefficients of the denominator
            d_coeffs = denominator.all_coeffs()
            n_coeffs = numerator.all_coeffs()

            # Check the constant coefficient and factor a minus sign if necessary
            if d_coeffs[-1] < 0:
                logger.debug("Try factoring a minus sign in %s", f)
                d_coeffs = [-x for x in d_coeffs]
                n_coeffs = [-x for x in n_coeffs]
            if all(map(lambda x: x <= 0, d_coeffs[:-1])):  # are all other coefficients non-positive?
                if all(map(lambda x: x >= 0, n_coeffs)):  # are all nominator coefficients non-negative?
                    logger.info("Invariant validates as non-negative FPS.")
                    return True

            # Search for bad coefficients
            for i, (coef, state) in enumerate(config.factory.from_expr(str(f).replace("**", "^"), *variables)):
                if i > 10:
                    break
                if sympy.S(coef) < 0:
                    logger.info("Invariant is spurious. Coefficient of state %s is %s", state.valuations, coef)
                    return False

            # We don't know otherwise
            logger.info("Heuristic failed, we dont know!")
            return None

        def _spuriosity_check(f: sympy.Expr) -> bool:
            """
            Here we do check univariate rational functions for positivity, i.e., whether all series coefficients
            of the rational function _f_ are >= 0. This is just a heuristic and not a complete method.
            Currently only univariate, non-parametric rational functions are supported.
            """
            logger.debug("Spuriosity check for %s", f)
            if len(f.free_symbols) > 1:
                logger.info("Heuristic currently does not support multivariate %s", f)
                return False

            # If the denominator is constant, i.e. we already have a polynomial,
            # just compute the result by checking the coefficients.
            if f.is_polynomial():
                # here we need a dummy symbol "" for sympy to handle as_poly correctly.
                return all(map(lambda x: x >= 0, f.as_poly(*f.free_symbols).coeffs()))

            # We indeed have a rational function, so we split this into nominator and denominator
            # and do some edge case handling for constant degree zero polynomials (not theory wise but
            # the implementation crashes otherwise)
            numerator, denominator = f.as_numer_denom()
            if not (numerator.is_polynomial() and denominator.is_polynomial()):
                if numerator.is_constant() and denominator.is_constant():
                    return numerator / denominator > 0
                logger.info("%s is not a rational function", f)
                return False
            n_deg = 0 if numerator.is_constant() else numerator.as_poly().degree()
            d_deg = 0 if denominator.is_constant() else denominator.as_poly().degree()

            # We make sure that we have a true rational function so we try to convert the problem for fractions where
            # the numerator degreee is not smaller than the denominator in the case where we have a "true" fraction
            if n_deg >= d_deg:
                # Filter the polynomial part
                poly_part = sympy.S("0")
                for term in f.apart().as_ordered_terms():
                    numerator, denominator = term.as_numer_denom()
                    if denominator.is_constant():
                        poly_part += numerator / denominator
                fractional_part = f - poly_part
                if poly_part.is_constant():
                    max_deg = 0
                    poly_coeffs = [poly_part]
                else:
                    max_deg = poly_part.as_Poly().degree()
                    poly_coeffs = poly_part.as_Poly().all_coeffs()

                # Now we recursively check whether the fractional part has only non-negative coefficients.
                if not _spuriosity_check(fractional_part):
                    logger.info("The fractional part could not be validated to be strictly positive")
                    return False

                # If this was successfull, we now try to see whether we can annihilate negative coefficients in the
                # polynomial part. This is necessary for the whole invariant to be a true fix point.
                fractional_coeffs = fractional_part.series(n=max_deg).removeO()
                fractional_coeffs = [
                    fractional_coeffs] if fractional_coeffs.is_constant() else fractional_coeffs.as_Poly().all_coeffs()
                if not all(map(lambda x, y: y >= 0 or x + y >= 0, fractional_coeffs, poly_coeffs)):
                    logger.info("This is a spurious invariant!")
                    return False
                return True

            # we are in the true fractional case. Here we can sometimes validate correct solutions by looking at
            # the positivity of the roots of the denominator.
            for root in denominator.as_poly(*denominator.free_symbols, domain="R").real_roots():
                if root <= 0:
                    logger.info("Heuristic is not applicable as the root %s is at most 0.", root)
                    return False
            # All real roots of the denominator are positive!
            logger.debug("Validated non-spuriousity.")
            return True

        assert error_prob.is_zero_dist(), f"Currently EVT reasoning does not support conditioning."
        max_deg = int(input("Enter the maximal degree of the rational function: "))
        logger.debug("Maximal degree for invariant search %i", max_deg)

        for evt_candidate in _generate_rational_function_stepwise(max_deg):
            print(f"{Style.YELLOW}Invariant candidate: {evt_candidate}{Style.RESET}", end="\n")
            # Generate invariant candidate function and compute one iteration step
            evt_inv = evt_candidate
            phi_inv = distribution + \
                      analyzer(instruction.body, prog_info, evt_inv.filter(instruction.cond), error_prob, config)[0]
            # If they are syntactically equal we are done.
            logger.debug("Check Invariant candidate %s", evt_inv)
            if evt_inv == phi_inv:
                logger.debug("Candidate validated.")
                return evt_inv - evt_inv.filter(instruction.cond), error_prob

            # otherwise we compute the difference and try to solve for the coefficients
            diff = evt_inv - phi_inv
            coeffs = [sympy.S(p) for p in diff.get_parameters()]
            variables = [sympy.S(v) for v in diff.get_variables()]
            logger.debug("Start solving equation systems for %s", diff)
            solution_candidates = sympy.solve_undetermined_coeffs(sympy.S(str(diff)), coeffs, variables,
                                                                  particular=True)
            if not isinstance(solution_candidates, list):
                solution_candidates = [solution_candidates]
            logger.debug("Filter solutions in: %s", solution_candidates)
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
            if len(solutions) > 0:
                print()  # generate new line after solutions found.
                solutions_and_kinds = [(sol, _check_nonnegative(sympy.S(str(evt_inv)).subs(sol).ratsimp())) for sol in
                                       solutions]
                distributions = [
                    sympy.S(str(evt_inv - evt_inv.filter(instruction.cond))).subs(sol).ratsimp()
                    if (kind is None) or kind else "-1"
                    for sol, kind in solutions_and_kinds
                ]
                for i, (sol, kind) in enumerate(solutions_and_kinds):
                    logger.debug("Possible solution: %s", sol)
                    message = {None: f"{Style.YELLOW}Possible invariant{Style.RESET}",
                               True: f"{Style.GREEN}Invariant{Style.RESET}",
                               False: f"{Style.RED}Spurious invariant{Style.RESET}"}
                    print(f"{message[kind]}: {sympy.S(str(evt_inv)).subs(sol).ratsimp()}")
                    if kind is None or kind:
                        print(f"{Style.CYAN}Distribution:{Style.RESET} {distributions[i]}")

                # Currently we just take the first solution.
                return config.factory.from_expr(str(distributions[0]).replace("**", "^"),
                                                *evt_inv.get_variables()), error_prob
            print("" * 80, end="\r")  # Clear the current line for the "Degree d" output.
        raise VerificationError(f"Could not find a rational function inductive invariant up to degree {max_deg}")

    @staticmethod
    def compute(
            instruction: Instr,
            prog_info: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:

        _assume(instruction, WhileInstr, 'WhileHandler')

        while True:
            user_choice = input(
                "While Instruction has only limited support. Choose an option:\n"
                "[1]: Solve using invariants (Checks whether the invariant over-approximates the loop)\n"
                "[2]: Fix a maximum number of iterations (This results in an under-approximation)\n"
                "[3]: Analyse until a certain probability mass is captured (might not terminate!)\n"
                "[4]: Compute expected visiting times (might be an approximation)\n"
                "[5]: Use EVT Invariant\n"
                "[6]: EVT Invariant syntesis\n"
                "[q]: Quit.\n")
            logger.info("User chose %s", user_choice)
            if user_choice == "1":
                return WhileHandler._analyze_with_invariant(
                    instruction, prog_info, distribution, error_prob, config, analyzer)
            if user_choice == "2":
                return WhileHandler._compute_iterations(
                    instruction, prog_info, distribution, error_prob, config, analyzer)
            if user_choice == "3":
                return WhileHandler._compute_until_threshold(
                    instruction, prog_info, distribution, error_prob, config, analyzer)
            if user_choice == "4":
                return WhileHandler._approx_expected_visiting_times(instruction, prog_info, distribution, error_prob,
                                                                    config, analyzer)
            if user_choice == "5":
                return WhileHandler._evt_invariant(instruction, prog_info, distribution, error_prob, config, analyzer)
            if user_choice == "6":
                return WhileHandler._evt_invariant_synthesis(instruction, prog_info, distribution, error_prob, config,
                                                             analyzer)
            if user_choice == "q":
                sys.exit()
            print(f"Invalid input: {user_choice}")
