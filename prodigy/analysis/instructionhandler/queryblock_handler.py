from __future__ import annotations

import logging
from typing import Union, Sequence, Callable

from probably.pgcl import Instr, QueryInstr, AsgnInstr, ObserveInstr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.assignment_handler import AssignmentHandler
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution, MarginalType
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


class QueryBlockHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]], prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, QueryInstr, "QueryBlockHandler")
        instrs = instruction.instrs  # type: ignore
        assert len(instrs) == 2
        assert isinstance(instrs[0], AsgnInstr)
        assert isinstance(instrs[1], ObserveInstr)
        assert distribution.is_finite()
        assert error_prob.is_zero_dist(
        )  # this means the sum of all probabilities used below is 1 respectively

        dist, _ = AssignmentHandler.compute(instrs[0], prog_info, distribution,
                                            error_prob, config, analyzer)
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
