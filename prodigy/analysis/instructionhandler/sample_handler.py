from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Callable, Sequence, Union

from prodigy.pgcl import (Instr, AsgnInstr, FunctionCallExpr, NatLitExpr, Var, VarExpr,
                          sample_predefined_functions as distr_functions)

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution, MarginalType
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
