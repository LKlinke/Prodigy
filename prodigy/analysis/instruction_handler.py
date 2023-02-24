import functools
import logging
import sys
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Dict, List, Sequence, Set, Tuple, Union, get_args

from probably.pgcl import (AsgnInstr, Binop, BinopExpr, CategoricalExpr,
                           ChoiceInstr, ExpectationInstr, Expr,
                           FunctionCallExpr, IfInstr, Instr, LoopInstr,
                           NatLitExpr, ObserveInstr, OptimizationQuery,
                           OptimizationType, PlotInstr, PrintInstr,
                           ProbabilityQueryInstr, Program, Query, QueryInstr,
                           RealLitExpr, SkipInstr, Unop, UnopExpr, Var,
                           VarExpr, WhileInstr, parse_pgcl)
from probably.pgcl.check import sample_predefined_functions as distr_functions

import prodigy.analysis.equivalence.equivalence_check as equiv_check
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import (ObserveZeroEventError,
                                         VerificationError)
from prodigy.analysis.plotter import Plotter
from prodigy.distribution.distribution import Distribution, MarginalType, State
from prodigy.distribution.generating_function import SympyPGF
from prodigy.util import input_manager
from prodigy.util.color import Style
from prodigy.util.logger import log_setup, print_progress_bar

logger = log_setup(__name__, logging.DEBUG)


def _assume(instruction: Instr, instr_type, clsname: str):
    """ Checks that the instruction is given as the right type."""
    assert isinstance(instruction, instr_type), f"The Instruction handled by a {clsname} must be of type" \
                                                f" {instr_type} got {type(instruction)}"


def compute_discrete_distribution(
        prog: Program,
        dist: Distribution,
        config: ForwardAnalysisConfig,
        so_vars: Set[str] | None = None) -> Distribution:
    # Inject the context from the parsed program into the initial distribution.
    variables = set(prog.variables.keys()).union(dist.get_variables())
    parameters = set(prog.parameters.keys()).union(dist.get_parameters())
    initial_dist = dist.set_variables(*variables).set_parameters(*parameters)
    error_prob = config.factory.one(*variables) * 0

    dist, error_prob = SequenceHandler.compute(prog.instructions, prog,
                                               initial_dist, error_prob,
                                               config, so_vars)
    return condition_distribution(dist, error_prob, config)


def condition_distribution(dist: Distribution, error_prob: Distribution,
                           config: ForwardAnalysisConfig) -> Distribution:
    one = config.factory.one(*error_prob.get_variables())
    zero = one * "0"
    if dist == zero and error_prob == one:
        raise ObserveZeroEventError(
            "Undefined semantics: Probability of observing a valid run is 0.")
    result = dist / (one - error_prob)
    return result


class InstructionHandler(ABC):
    """ Abstract class that defines a strategy for handling a specific program instruction. """
    @staticmethod
    @abstractmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]],
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        """Computes the updated distribution after executing the instruction `instruction` on input `distribution`"""


class SequenceHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]],
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config=ForwardAnalysisConfig(),
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        def _show_steps(inp: tuple[Distribution, Distribution],
                        instr: Instr) -> tuple[Distribution, Distribution]:
            dist, error_prob = inp
            res = SequenceHandler.compute(instr, program, dist, error_prob,
                                          config, so_vars)
            if isinstance(instr, (WhileInstr, IfInstr, LoopInstr)):
                print("\n")
            output = f"\n{Style.BLUE}Instruction:{Style.RESET} {instr}\t {Style.GREEN}Result:{Style.RESET} {res}"
            print(output, so_vars)
            return res

        def _dont_show_steps(
                inp: tuple[Distribution, Distribution],
                instr: Instr) -> tuple[Distribution, Distribution]:
            dist, error_prob = inp
            return SequenceHandler.compute(instr, program, dist, error_prob,
                                           config, so_vars)

        if isinstance(instruction, list):
            func = _show_steps if config.show_intermediate_steps else _dont_show_steps
            return functools.reduce(func, instruction,
                                    (distribution, error_prob))

        elif isinstance(instruction, SkipInstr):
            return distribution, error_prob

        elif isinstance(instruction, WhileInstr):
            logger.info("%s gets handled", instruction)
            return WhileHandler.compute(instruction, program, distribution,
                                        error_prob, config, so_vars)

        elif isinstance(instruction, IfInstr):
            return ITEHandler.compute(instruction, program, distribution,
                                      error_prob, config, so_vars)

        elif isinstance(instruction, AsgnInstr):
            return AssignmentHandler.compute(instruction, program,
                                             distribution, error_prob, config,
                                             so_vars)

        elif isinstance(instruction, ChoiceInstr):
            return PChoiceHandler.compute(instruction, program, distribution,
                                          error_prob, config, so_vars)

        elif isinstance(instruction, ObserveInstr):
            logger.info("%s gets handled", instruction)
            return ObserveHandler.compute(instruction, program, distribution,
                                          error_prob, config, so_vars)

        elif isinstance(instruction, get_args(Query)):
            logger.info("%s gets handled", instruction)
            distribution = condition_distribution(
                distribution, error_prob,
                config)  # evaluate queries on conditioned distribution
            error_prob *= "0"
            return QueryHandler.compute(instruction, program, distribution,
                                        error_prob, config, so_vars)

        elif isinstance(instruction, LoopInstr):
            logger.info("%s gets handled", instruction)
            return LoopHandler.compute(instruction, program, distribution,
                                       error_prob, config, so_vars)

        elif isinstance(instruction, QueryInstr):
            logger.info("entering query block")
            return QueryBlockHandler.compute(instruction, program,
                                             distribution, error_prob, config,
                                             so_vars)

        raise TypeError("illegal instruction")


class QueryHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, get_args(Query), 'QueryHandler')

        # User wants to compute an expected value of an expression
        if isinstance(instruction, ExpectationInstr):
            expression = instruction.expr
            if isinstance(expression, (VarExpr, NatLitExpr, RealLitExpr)):
                result = distribution.get_expected_value_of(expression)
            elif isinstance(expression, BinopExpr):
                if expression.operator in (Binop.PLUS, Binop.MINUS,
                                           Binop.TIMES):
                    result = distribution.get_expected_value_of(expression)
                else:
                    raise SyntaxError("Expression cannot be a condition.")
            elif isinstance(expression,
                            UnopExpr) and expression.operator == Unop.NEG:
                result = distribution.get_expected_value_of(expression)
            else:
                raise SyntaxError("Expression has wrong format.")

            print(f"Expected value: {result}")
            return distribution, error_prob

        # User wants to compute a marginal, or the probability of a condition.
        elif isinstance(instruction, ProbabilityQueryInstr):
            return QueryHandler.__query_probability_of(instruction.expr,
                                                       distribution), \
                   error_prob

        # User wants to Plot something
        elif isinstance(instruction, PlotInstr):
            return QueryHandler.__query_plot(instruction, distribution), \
                   error_prob

        # User wants to print the current distribution.
        elif isinstance(instruction, PrintInstr):
            print(distribution)
            return distribution, error_prob

        elif isinstance(instruction, OptimizationQuery):
            return QueryHandler.__query_optimization(instruction, distribution,
                                                     config), \
                   error_prob

        else:
            raise SyntaxError("This should not happen.")

    @staticmethod
    def __query_optimization(instr: OptimizationQuery, dist: Distribution,
                             config: ForwardAnalysisConfig):
        logger.debug(
            "Computing the optimal value for parameter %s in order to %s the distribution %s with respect to %s",
            instr.parameter, 'maximize' if instr.type
            == OptimizationType.MAXIMIZE else 'minimize', dist, instr.expr)
        result = config.optimizer.optimize(instr.expr,
                                           dist,
                                           instr.parameter,
                                           method=instr.type)
        if not result:
            print("No solutions could be found.")
        elif len(result) == 1:
            print(f"The optimal value is at {instr.parameter}={result[0]}.")
        else:
            print(
                f"The maximal values can be achieved choosing the parameter {instr.parameter} in {result}."
            )
        return dist

    @staticmethod
    def __query_plot(instr: PlotInstr, dist: Distribution) -> Distribution:
        # Gather the variables to plot.
        variables = [instr.var_1.var]
        variables += [instr.var_2.var] if instr.var_2 else []

        # User can specify either a probability threshold or a number of terms which are shown in the histogram.
        if instr.prob:
            # User chose a probability or oo (meaning, try to compute the whole histogram).
            if instr.prob.is_infinite():
                Plotter.plot(dist, *variables, threshold=str(1))
            else:
                Plotter.plot(dist, *variables, threshold=str(instr.prob))

        elif instr.term_count:
            # User has chosen a term limit.
            Plotter.plot(dist, *variables, threshold=instr.term_count.value)
            # User did neither specify a term limit nor a threshold probability. We try an iterative approach...
        else:
            p = .1
            inc = .1
            while True:
                Plotter.plot(dist, *variables, threshold=str(p))
                p = p + inc if p + inc < 1 else 1
                if input_manager.reader.read_yn(f"Continue with p={p}?"):
                    break
        return dist

    @staticmethod
    def __query_probability_of(expression: Expr,
                               dist: Distribution) -> Distribution:
        # Marginal computation
        if isinstance(expression, VarExpr):
            marginal = dist.marginal(expression, method=MarginalType.INCLUDE)
            print(f"Marginal distribution of {expression}: {marginal}")
        # Probability of condition computation.
        else:
            sat_part = dist.filter(expression)
            prob = sat_part.get_probability_mass()
            print(f"Probability of {expression}: {prob}")
        return dist


class SampleHandler(InstructionHandler):
    @staticmethod
    @abstractmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, AsgnInstr, 'SampleHandler')
        assert isinstance(instruction.rhs, FunctionCallExpr), "The Instruction handled by a SampleHandler must be of" \
                                                                 f" type FunctionCallExpr, got {type(instruction)}"
        assert instruction.rhs.function in distr_functions, f"Not a pre-defined function: {instruction.rhs.function}"

        logger.info("Computing distribution sampling update.\n%s", instruction)
        variable: Var = instruction.lhs
        marginal = distribution.marginal(variable, method=MarginalType.EXCLUDE)
        factory = config.factory
        distr = instruction.rhs.function

        # rhs is a uniform distribution
        if distr in {"unif_d", "unif"}:
            start, end = instruction.rhs.params[0]
            return marginal * factory.uniform(variable, start, end), error_prob

        # rhs is binomial distribution
        if distr == "binomial":
            n, p = instruction.rhs.params[0]
            return marginal * factory.binomial(variable, n, p), error_prob

        # rhs is an iid sampling expression
        if distr == "iid":
            dist, count = instruction.rhs.params[0]
            return distribution.update_iid(dist, count, instruction.lhs), \
                   error_prob

        # all remaining distributions have only one parameter
        [param] = instruction.rhs.params[0]

        # rhs is geometric distribution
        if distr == "geometric":
            return marginal * factory.geometric(variable, param), error_prob

        # rhs is poisson distribution
        if distr == "poisson":
            return marginal * factory.poisson(variable, param), error_prob

        # rhs is bernoulli distribution
        if distr == "bernoulli":
            return marginal * factory.bernoulli(variable, param), error_prob

        # rhs is logarithmic distribution
        if distr == "logdist":
            return marginal * factory.log(variable, param), error_prob

        raise ValueError("Unknown distribution type!")


class FunctionHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, AsgnInstr, 'FunctionCallHandler')
        assert isinstance(instruction.rhs, FunctionCallExpr)

        if instruction.rhs.function in distr_functions:
            return SampleHandler.compute(instruction, program, distribution,
                                         error_prob, config, so_vars)

        function = program.functions[instruction.rhs.function]
        input_distr = distribution.set_variables(*distribution.get_variables(),
                                                 *function.variables)

        new_names: Dict[str, str] = {}
        for var, val in function.params_to_dict(
                instruction.rhs.params).items():
            new_name = input_distr.get_fresh_variable()
            new_names[new_name] = var
            # TODO add a more efficient add_variable function for cases like this?
            input_distr = input_distr.set_variables(
                new_name, *input_distr.get_variables()).update(
                    BinopExpr(Binop.EQ, VarExpr(new_name), val))
        for new, old in new_names.items():
            input_distr = input_distr.update(
                BinopExpr(Binop.EQ, VarExpr(old), VarExpr(new)))
        input_distr = input_distr.marginal(*new_names.values()).set_variables(
            *function.variables)

        returned = compute_discrete_distribution(
            Program.from_function(function, program), input_distr, config,
            so_vars).evaluate_expression(function.returns, instruction.lhs)
        return distribution.marginal(
            instruction.lhs,
            method=MarginalType.EXCLUDE) * returned, error_prob


class AssignmentHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, AsgnInstr, 'AssignmentHandler')

        # rhs is a categorical expression (explicit finite distr)
        if isinstance(instruction.rhs, CategoricalExpr):
            raise NotImplementedError(
                "Categorical expression are currently not supported.")

        if isinstance(instruction.rhs, FunctionCallExpr):
            return FunctionHandler.compute(instruction, program, distribution,
                                           error_prob, config, so_vars)

        if not isinstance(instruction.rhs, get_args(Expr)):
            raise SyntaxError(
                f"Assignment {instruction} is ill-formed. right-hand-side must be an expression."
            )
        logger.info("Computing distribution update.\n%s", instruction)
        return distribution.update(
            BinopExpr(operator=Binop.EQ,
                      lhs=VarExpr(instruction.lhs),
                      rhs=instruction.rhs)), \
               error_prob


class ObserveHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, ObserveInstr, 'ObserverHandler')

        sat_part = distribution.filter(instruction.cond)
        unsat_part = distribution - sat_part
        if so_vars is None or len(so_vars) == 0:
            unsat_prob = unsat_part.get_probability_mass()
        else:
            unsat_prob = unsat_part.marginal(*so_vars)
        res_error_prob = error_prob + unsat_prob
        return sat_part, res_error_prob


class PChoiceHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, ChoiceInstr, 'PChoiceHandlerGF')

        lhs_block, lhs_error_prob = SequenceHandler.compute(
            instruction.lhs, program, distribution, error_prob, config,
            so_vars)
        rhs_block, rhs_error_prob = SequenceHandler.compute(
            instruction.rhs, program, distribution, error_prob, config,
            so_vars)
        logger.info("Combining PChoice branches.\n%s", instruction)
        res_error_prob = (
            lhs_error_prob * str(instruction.prob) +
            rhs_error_prob * f"1-{instruction.prob}").set_variables(
                *error_prob.get_variables())
        return lhs_block * str(
            instruction.prob) + rhs_block * f"1-({instruction.prob})", \
               res_error_prob


class ITEHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, IfInstr, 'ITEHandler')

        logger.info("Filtering the guard %s", instruction.cond)
        sat_part = distribution.filter(instruction.cond)
        non_sat_part = distribution - sat_part
        zero = error_prob * "0"
        if config.show_intermediate_steps:
            print(
                f"\n{Style.YELLOW}Filter:{Style.RESET} {instruction.cond} \t "\
                    f"{Style.GREEN}Result:{Style.RESET} {sat_part}"
            )
            print(
                f"\n{Style.YELLOW} If-branch: ({instruction.cond}){Style.RESET}"
            )
            if_branch, if_error_prob = SequenceHandler.compute(
                instruction.true, program, sat_part, zero, config, so_vars)
            print(f"\n{Style.YELLOW} Else-branch:{Style.RESET}")
            else_branch, else_error_prob = SequenceHandler.compute(
                instruction.false, program, non_sat_part, zero, config,
                so_vars)
            print(f"\n{Style.YELLOW}Combined:{Style.RESET}")
        else:
            if_branch, if_error_prob = SequenceHandler.compute(
                instruction.true, program, sat_part, zero, config, so_vars)
            else_branch, else_error_prob = SequenceHandler.compute(
                instruction.false, program, non_sat_part, zero, config,
                so_vars)
        result = if_branch + else_branch
        res_error_prob = error_prob + if_error_prob + else_error_prob
        logger.info("Combining if-branches.\n%s", instruction)
        return result, res_error_prob


class LoopHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, LoopInstr, 'LoopHandler')
        for i in range(instruction.iterations.value):
            logger.info("Computing iteration %d out of %d", i + 1,
                        instruction.iterations.value)
            distribution, error_prob = SequenceHandler.compute(
                instruction.body, program, distribution, error_prob, config,
                so_vars)
        return distribution, error_prob


class WhileHandler(InstructionHandler):
    @staticmethod
    def _analyze_with_invariant(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        inv_filepath = input_manager.reader.read_file('Invariant file: ')
        with open(inv_filepath, 'r', encoding="utf-8") as inv_file:
            inv_src = inv_file.read()
            inv_prog = parse_pgcl(inv_src)

            prog = Program(declarations=[],
                           variables=program.variables,
                           constants=program.constants,
                           parameters=program.parameters,
                           instructions=[instruction],
                           functions=program.functions)
            print(f"{Style.YELLOW}Verifying invariant...{Style.RESET}")
            answer, result = equiv_check.check_equivalence(
                prog, inv_prog, config)
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
                return SequenceHandler.compute(inv_prog.instructions, program,
                                               distribution, error_prob,
                                               config, so_vars)
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
            instruction: Instr, program: Program, distribution: Distribution,
            error_prob: Distribution, config: ForwardAnalysisConfig
    ) -> tuple[Distribution, Distribution]:
        max_iter = int(
            input_manager.reader.read_text(
                "Specify a maximum iteration limit: "))
        sat_part = distribution.filter(instruction.cond)
        non_sat_part = distribution - sat_part
        for i in range(max_iter + 1):
            print_progress_bar(i, max_iter, length=50)
            iterated_part, error_prob = SequenceHandler.compute(
                instruction.body, program, sat_part, error_prob, config)
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
            instruction: Instr, program: Program, distribution: Distribution,
            error_prob: Distribution, config: ForwardAnalysisConfig
    ) -> tuple[Distribution, Distribution]:
        captured_probability_threshold = input_manager.reader.read_float(
            "Enter the probability threshold: ")
        sat_part = distribution.filter(instruction.cond)
        non_sat_part = distribution - sat_part
        captured_part = non_sat_part + error_prob
        while Fraction(captured_part.get_probability_mass()
                       ) < captured_probability_threshold:
            logger.info("Collected %f of the desired mass", (float(
                (Fraction(captured_part.get_probability_mass()) /
                 captured_probability_threshold)) * 100))
            iterated_part, error_prob = SequenceHandler.compute(
                instruction.body, program, sat_part, error_prob, config)
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
    def compute(
            instruction: Instr,
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:

        _assume(instruction, WhileInstr, 'WhileHandler')

        options: List[str | Tuple[str, str]] = [
            "Solve using invariants (Checks whether the invariant over-approximates the loop)",
            "Fix a maximum number of iterations (This results in an under-approximation)",
            "Analyse until a certain probability mass is captured (might not terminate!)",
            ('q', 'Quit')
        ]
        user_choice = input_manager.reader.read_option(
            *options,
            prompt=
            "While Instruction has only limited support. Choose an option:")
        logger.info("User chose %s", options[user_choice])
        if user_choice == 0:
            return WhileHandler._analyze_with_invariant(
                instruction, program, distribution, error_prob, config,
                so_vars)
        elif user_choice == 1:
            return WhileHandler._compute_iterations(instruction, program,
                                                    distribution, error_prob,
                                                    config)
        elif user_choice == 2:
            return WhileHandler._compute_until_threshold(
                instruction, program, distribution, error_prob, config)
        else:
            sys.exit()


class QueryBlockHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]],
            program: Program,
            distribution: Distribution,
            error_prob: Distribution,
            config: ForwardAnalysisConfig,
            so_vars: Set[str] | None = None
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, QueryInstr, "QueryBlockHandler")
        instrs = instruction.instrs  # type: ignore
        assert len(instrs) == 2
        assert isinstance(instrs[0], AsgnInstr)
        assert isinstance(instrs[1], ObserveInstr)
        assert distribution.is_finite()
        assert error_prob.is_zero_dist(
        )  # this means the sum of all probabilities used below is 1 respectively

        dist, _ = AssignmentHandler.compute(instrs[0], program, distribution,
                                            error_prob, config, so_vars)
        marginal = dist.marginal(instrs[0].lhs)
        assert marginal.is_finite()

        # this assumes that the updated variable is independent from the rest of the distribution
        zero: Distribution = config.factory.one(*dist.get_variables()) * '0'
        for rest_prob, rest_state in dist.marginal(
                instrs[0].lhs, method=MarginalType.EXCLUDE):
            if rest_prob == "0":
                continue
            new = zero
            for me_prob, me_state in marginal:
                if me_prob == "0":
                    continue
                state = rest_state.extend(me_state)
                if dist.evaluate_condition(instrs[1].cond, state):
                    new += dist.filter_state(state)
            new /= str(new.get_probability_mass())
            new *= rest_prob
            dist -= dist.filter_state(rest_state)
            dist += new

        return dist, error_prob
