from __future__ import annotations

import logging
from abc import abstractmethod
from fractions import Fraction
from typing import Callable, Dict, List, Sequence, Set, Tuple, Union

from prodigy.language.ast.types import NatType
from prodigy.language import (Instr, AsgnInstr, ChoiceInstr, Expr, FunctionCallExpr, NatLitExpr, Var, VarExpr,
                          sample_predefined_functions as distr_functions)

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.instructionhandler.counter_operators import (counter_probability, ratio_probability,
                                                                   uniform_up_to)
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
        _check_generating_functions(instruction.rhs, instruction.lhs, prog_info)
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


def _generating_functions(call: FunctionCallExpr, assigned: Var) -> List[Tuple[FunctionCallExpr, Expr, Var]]:
    """The generating functions given to ``sample_pgf`` and ``iid`` in ``call``, each with the call it is an
    argument of and its variable: ``sample_pgf(G, x)`` reads ``G`` in ``x``, ``iid(G, n)`` in the assigned
    variable (other program variables in ``G`` are variables of the generating function as well)."""
    if call.function == "sample_pgf":
        generating_function, variable = call.params[0][:2]
        return [(call, generating_function, variable.var if isinstance(variable, VarExpr) else assigned)]
    if call.function == "iid":
        dist = call.params[0][0]
        if isinstance(dist, FunctionCallExpr):
            return _generating_functions(dist, assigned) if dist.function in distr_functions else []
        return [(call, dist, assigned)]
    return []


def _check_generating_functions(call: FunctionCallExpr, assigned: Var, prog_info: ProgramInfo) -> None:
    """
    Checks that the generating functions given to ``sample_pgf`` and ``iid`` are probability generating
    functions: power series at 0 with non-negative coefficients and mass 1 (#424), with the rules of input
    distributions (:func:`~prodigy.distribution.finite_measure.check_probability_generating_function`).

    Refuted: ``ValueError`` naming the generating function. Not proven: if the generating function depends on
    parameters, the result is a formula in them that holds for the parameter values for which it is a
    probability generating function, as for parametric arguments of the other distributions (e.g.
    ``geometric(p)``), and a warning says so; otherwise the result has no guarantee
    (:meth:`~prodigy.analysis.session.AnalysisSession.unproven`).
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.finite_measure import check_probability_generating_function
    from prodigy.analysis.session import current_session
    session = current_session()
    parameters = {str(name): typ for name, typ in prog_info.parameters.items()}
    natural = tuple(sorted(name for name, typ in parameters.items() if isinstance(typ, NatType)))
    for sampling, generating_function, variable in _generating_functions(call, assigned):
        names = expression_variables([generating_function])
        variables = tuple(sorted((names - set(parameters)) | {str(variable)}))
        verdict, reason = check_probability_generating_function(str(generating_function), variables, natural)
        if verdict is True:
            continue
        text, construct = session.spell(generating_function), session.spell(sampling)
        if verdict is False:
            raise ValueError(f"{construct} needs a probability generating function (non-negative coefficients, "
                             f"mass 1), but {text} is not one: {reason}.")
        used = sorted(names & set(parameters))
        if used:
            session.warn(f"Prodigy could not prove that {text} in {construct} is a probability generating function "
                         f"({reason}). The result holds for the values of the parameter"
                         f"{'s' if len(used) > 1 else ''} {', '.join(used)} for which it is one.")
        else:
            session.unproven(f"Prodigy could not prove that {text} in {construct} is a probability generating "
                             f"function ({reason}), so the result has no guarantee.")


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


def _uniform_infinite_bound(instruction: AsgnInstr, variables: Set[str], distribution: Distribution) -> Distribution:
    """
    ``x = unif(0, n)`` for a program variable n with infinitely many values, by the operator ``1/(θ + 1)`` in n
    (:func:`~prodigy.analysis.instructionhandler.counter_operators.uniform_up_to`, #385). Other bounds that depend
    on such variables are not implemented.
    """
    assert isinstance(instruction.rhs, FunctionCallExpr)
    construct = f"The distribution {instruction.rhs}"
    start, end = instruction.rhs.params[0]
    if expression_variables([start]) or constant_value(start) != 0 or not isinstance(end, VarExpr):
        distribution.checked_marginal(*sorted(variables))
        names = ", ".join(sorted(variables))
        raise NotImplementedError(
            f"{construct} depends on the program variable(s) {names}, which can take infinitely many values "
            f"here. For infinitely many values, Prodigy computes only unif(0, n) with a program variable n as "
            f"upper bound; other bounds (a lower bound other than 0, an expression as bound) are not implemented.")
    logger.info("Applying %s as an operator in %s", instruction.rhs, end.var)
    return uniform_up_to(distribution, instruction.lhs, end.var, construct)


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
        infinite = not distribution.is_zero_dist() and (
            operator is not None and distribution.finite_support(operator.variable) is None
            or ratio_probability(probability, variables) is not None
            and distribution.finite_support(*sorted(variables)) is None)
        if infinite:
            # x = bernoulli(p) is {x = 1} [p] {x = 0}; p is evaluated before the assignment, as in the choice, so
            # the choice applies p as an operator in the counter (#323) or in the two variables of x/n (#237)
            choice = ChoiceInstr(probability, [AsgnInstr(instruction.lhs, NatLitExpr(1))],
                                 [AsgnInstr(instruction.lhs, NatLitExpr(0))])
            return analyzer(choice, prog_info, distribution, error_prob, config)
    if instruction.rhs.function in ("unif", "unif_d") and not distribution.is_zero_dist() \
            and distribution.finite_support(*sorted(variables)) is None:
        return _uniform_infinite_bound(instruction, variables, distribution), error_prob
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
