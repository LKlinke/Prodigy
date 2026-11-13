from __future__ import annotations

import logging
from abc import abstractmethod
from fractions import Fraction
from typing import Callable, Dict, List, Sequence, Set, Tuple, Union

from prodigy.pgcl import (Instr, AsgnInstr, ChoiceInstr, Expr, FunctionCallExpr, NatLitExpr, Var, VarExpr,
                          sample_predefined_functions as distr_functions)

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.instructionhandler.counter_operators import counter_probability
from prodigy.analysis.instructionhandler.state_dependent import (expression_variables, instantiate,
                                                                 split_states, state_variables)
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.distribution import State, constant_value
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


class SampleHandler(InstructionHandler):
    @staticmethod
    @abstractmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]], prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        instruction = _assume(instruction, AsgnInstr, 'SampleHandler')
        assert isinstance(instruction.rhs, FunctionCallExpr), "The Instruction handled by a SampleHandler must be of" \
                                                              f" type FunctionCallExpr, got {type(instruction)}"
        assert instruction.rhs.function in distr_functions, f"Not a pre-defined function: {instruction.rhs.function}"

        variables = state_variables(_arguments(instruction.rhs), distribution)
        if variables:
            return _sample_state_dependent(instruction, variables, prog_info, distribution,
                                           error_prob, config, analyzer)

        _check_arguments(instruction.rhs)
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

        # rhs is a negative binomial distribution
        if distr == "negative_binomial":
            r, p = instruction.rhs.params[0]
            return marginal * factory.negative_binomial(variable, r, p), error_prob

        # rhs is an iid sampling expression
        if distr == "iid":
            dist, count = instruction.rhs.params[0]
            # the type checker only admits a program variable or a natural number as count
            if not isinstance(count, (VarExpr, NatLitExpr)):
                raise ValueError(f"The number of samples in {instruction.rhs} must be a program "
                                 f"variable or a natural number, got {count}.")
            updated = distribution.update_iid(dist, count, instruction.lhs)
            return updated, error_prob

        if distr == "sample_pgf":
            dist_expr, dist_var = instruction.rhs.params[0]
            # the type checker only admits sampling in the assigned variable
            if dist_var != VarExpr(variable):
                raise ValueError(f"{instruction.rhs} samples a generating function in {dist_var}, "
                                 f"but only sampling in the assigned variable {variable} is supported.")
            sampled_dist = factory.from_expr(dist_expr, variable)
            return marginal * sampled_dist, error_prob

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


def _natural(value: Fraction) -> bool:
    return value.denominator == 1 and value >= 0


# For each distribution, its arguments with the condition on their values and its description.
_ADMISSIBLE: Dict[str, List[Tuple[str, Callable[[Fraction], bool], str]]] = {
    "bernoulli": [("p", lambda p: 0 <= p <= 1, "in [0, 1]")],
    "geometric": [("p", lambda p: 0 < p <= 1, "in (0, 1]")],
    "poisson": [("lambda", lambda lam: lam >= 0, "at least 0")],
    "logdist": [("p", lambda p: 0 < p < 1, "in (0, 1)")],
    "binomial": [("n", _natural, "a natural number"), ("p", lambda p: 0 <= p <= 1, "in [0, 1]")],
    "negative_binomial": [("r", _natural, "a natural number"), ("p", lambda p: 0 < p <= 1, "in (0, 1]")],
    "unif": [("a", _natural, "a natural number"), ("b", _natural, "a natural number")],
}
_ADMISSIBLE["unif_d"] = _ADMISSIBLE["unif"]


def _check_arguments(call: FunctionCallExpr) -> None:
    """
    Raises ``ValueError`` if a constant argument of a sampling call is not admissible, e.g.
    ``bernoulli(3/2)``, which is not a distribution. The check is the same on every engine;
    arguments with parameters are left symbolic.
    """
    if call.function == "iid":
        dist = call.params[0][0]
        if isinstance(dist, FunctionCallExpr):
            _check_arguments(dist)
        return
    values: List[Fraction] = []
    for arg, (name, admissible, description) in zip(call.params[0], _ADMISSIBLE.get(call.function, [])):
        if expression_variables([arg]):
            continue
        value = constant_value(arg)
        if value is None:
            raise ValueError(f"The argument {name} = {arg} of {call} is undefined.")
        if not admissible(value):
            raise ValueError(f"The argument {name} of {call} is {value}; it must be {description}.")
        values.append(value)
    if call.function in ("unif", "unif_d") and len(values) == 2 and values[0] > values[1]:
        raise ValueError(f"The bounds of {call} are {values[0]} > {values[1]}; the lower bound must "
                         f"not exceed the upper bound.")


def _arguments(call: FunctionCallExpr) -> List[Expr]:
    """
    The arguments of a sampling call that are parameters of the sampled distribution, e.g. ``p``
    of ``bernoulli(p)``. The count of ``iid(d, n)`` is not one: it may be a program variable (the
    distribution of the sum of ``n`` samples is computed for all values of ``n`` at once). A
    generating function given to ``sample_pgf`` or ``iid`` is not one either: its variables are
    the variables of the generating function, not program variables.
    """
    if call.function == "sample_pgf":
        return []
    if call.function == "iid":
        dist = call.params[0][0]
        return _arguments(dist) if isinstance(dist, FunctionCallExpr) else []
    return list(call.params[0])


def _instantiate_call(call: FunctionCallExpr, state: State) -> FunctionCallExpr:
    """``call`` with the program variables of ``state`` in its distribution arguments replaced by
    their values."""
    if call.function == "iid":
        dist, count = call.params[0]
        if isinstance(dist, FunctionCallExpr):
            dist = _instantiate_call(dist, state)
        return FunctionCallExpr(call.function, ([dist, count], call.params[1]))
    return FunctionCallExpr(call.function,
                            ([instantiate(arg, state) for arg in call.params[0]], call.params[1]))


def _sample_state_dependent(
        instruction: AsgnInstr, variables: Set[str], prog_info: ProgramInfo,
        distribution: Distribution, error_prob: Distribution, config: ForwardAnalysisConfig,
        analyzer: Callable[
            [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
            tuple[Distribution, Distribution]
        ]) -> tuple[Distribution, Distribution]:
    """
    A sampling instruction whose distribution arguments depend on program variables, e.g.
    ``x = bernoulli(1/k)``: a case split over the values of these variables
    (:mod:`~prodigy.analysis.instructionhandler.state_dependent`), sampling with the arguments'
    values in each part.
    """
    assert isinstance(instruction.rhs, FunctionCallExpr)
    if instruction.rhs.function == "bernoulli":
        probability = instruction.rhs.params[0][0]
        operator = counter_probability(probability, variables)
        if operator is not None and not distribution.is_zero_dist() \
                and distribution.finite_support(operator.variable) is None:
            # x = bernoulli(p) is {x = 1} [p] {x = 0}; p is evaluated before the assignment, as in the choice, so
            # the choice applies p as an operator in the counter (#323)
            choice = ChoiceInstr(probability, [AsgnInstr(instruction.lhs, NatLitExpr(1))],
                                 [AsgnInstr(instruction.lhs, NatLitExpr(0))])
            return analyzer(choice, prog_info, distribution, error_prob, config)
    logger.info("Splitting the states for the arguments of %s", instruction.rhs)
    construct = f"The distribution {instruction.rhs}"
    result = distribution * "0"
    for part in split_states(distribution, variables, construct):
        concrete = AsgnInstr(instruction.lhs, _instantiate_call(instruction.rhs, part.state))
        try:
            sampled, _ = SampleHandler.compute(concrete, prog_info, part.distribution, error_prob, config,
                                               analyzer)
        except ValueError as error:
            raise ValueError(f"{construct} is not defined in {part.describe()}: {error}") from error
        result += sampled
    return result, error_prob
