from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Callable, Union, Sequence

from probably.pgcl import Instr, AsgnInstr, FunctionCallExpr, sample_predefined_functions as distr_functions, BinopExpr, \
    Binop, VarExpr, Program

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.instructionhandler.sample_handler import SampleHandler
from prodigy.distribution import Distribution, MarginalType
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


class FunctionHandler(InstructionHandler):
    @staticmethod
    def compute(
            instruction: Instr, prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, AsgnInstr, 'FunctionCallHandler')
        assert isinstance(instruction.rhs, FunctionCallExpr)

        if instruction.rhs.function in distr_functions:
            return SampleHandler.compute(instruction, prog_info, distribution,
                                         error_prob, config, analyzer)

        function = prog_info.functions[instruction.rhs.function]
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

        sub_porgram = Program.from_function(function, prog_info.program)
        sub_result, sub_error = analyzer(sub_porgram.instructions,
                                         replace(prog_info, program=sub_porgram),
                                         input_distr,
                                         config.factory.from_expr("0"),
                                         config)
        result = sub_result.evaluate_expression(function.returns, instruction.lhs)
        return distribution.marginal(
            instruction.lhs,
            method=MarginalType.EXCLUDE) * result, error_prob
