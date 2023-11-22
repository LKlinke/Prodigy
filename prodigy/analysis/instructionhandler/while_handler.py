from __future__ import annotations

import logging
import sys
from fractions import Fraction
from typing import Callable, Union, Sequence

import sympy
from probably.pgcl import Instr, parse_pgcl, Program, WhileInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.evtinvariants.heuristics.strategies import KNOWN_STRATEGIES, SynthesisStrategy
from prodigy.analysis.evtinvariants.invariant_synthesis import evt_invariant_synthesis
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

        # Build the strategy.
        strategy: SynthesisStrategy = KNOWN_STRATEGIES[config.strategy](prog_info.variables.keys(), config.factory)

        # generate the invariants using the strategy
        invariants = evt_invariant_synthesis(instruction, prog_info, distribution, config, strategy, analyzer)

        def _sorting_key(elem):
            return 1 if elem[1] else 0

        invariants = sorted(invariants, key=_sorting_key)

        # give associated distributions:
        distributions = [inv - inv.filter(instruction.cond) for inv, inv_type in invariants]
        print(f"Continuing with the following distributions is possible:")
        for i, d in enumerate(distributions):
            print(f"{i + 1}.\t{d}")

        # Choose one solution
        print(f"Continue with distribution: {Style.CYAN}{distributions[0]}{Style.RESET}")
        return distributions[0], error_prob

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
