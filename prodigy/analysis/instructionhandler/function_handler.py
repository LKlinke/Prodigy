from __future__ import annotations

import logging
from typing import Callable, Dict, Sequence, Union

from prodigy.pgcl import (AsgnInstr, Binop, BinopExpr, Expr, FunctionCallExpr, Instr, NatLitExpr,
                          VarExpr, sample_predefined_functions as distr_functions)
from prodigy.pgcl.ast.types import DistributionType

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.instructionhandler.sample_handler import SampleHandler
from prodigy.distribution import Distribution
from prodigy.distribution.distribution import constant_value
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def _is_constant(expr: Expr) -> bool:
    """Whether ``expr`` has the same value in every state (a literal, or literals with operators)."""
    if isinstance(expr, NatLitExpr):
        return True
    return isinstance(expr, BinopExpr) and constant_value(expr) is not None


class FunctionHandler(InstructionHandler):
    """
    A call ``x = f(args)``. The body of ``f`` runs on its own variables, which start at the
    arguments (0 for the others). If every argument is a constant, the result does not depend
    on the caller's state: the body runs once on a point distribution, and the caller's
    distribution is multiplied by the distribution of the returned value. Otherwise the body
    runs on the joint distribution of the caller's and the function's variables, so the result
    keeps its dependence on the arguments
    (:func:`~prodigy.analysis.nested_inference.call_in_joint`). Observations inside the body
    condition the whole program, as anywhere else.
    """
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
        # pylint: disable=import-outside-toplevel
        from prodigy.analysis.nested_inference import (assign_return_value, call_in_joint,
                                                       function_program_info, without)
        instruction = _assume(instruction, AsgnInstr, 'FunctionCallHandler')
        assert isinstance(instruction.rhs, FunctionCallExpr)

        if instruction.rhs.function in distr_functions:
            return SampleHandler.compute(instruction, prog_info, distribution,
                                         error_prob, config, analyzer)

        function = prog_info.functions[instruction.rhs.function]
        assignments: Dict[str, Expr] = function.params_to_dict(instruction.rhs.params)
        types = {decl.var: decl.typ for decl in function.declarations}
        # dist arguments that are bound while a loop runs are bound in the body, too
        bindings = {}
        for var, expr in list(assignments.items()):
            if isinstance(types.get(var), DistributionType) and isinstance(expr, VarExpr) \
                    and expr.var in prog_info.dist_bindings:
                bindings[var] = prog_info.dist_bindings[expr.var]
                del assignments[var]

        if prog_info.so_vars or not all(_is_constant(expr) for expr in assignments.values()):
            return call_in_joint(function, assignments, bindings, instruction.lhs, prog_info,
                                 distribution, error_prob, config, analyzer)

        start = config.factory.one(*function.variables)
        for var, expr in assignments.items():
            start = start.update(BinopExpr(Binop.EQ, VarExpr(var), expr))
        sub_info = function_program_info(prog_info, function, bindings)
        zero = config.factory.from_expr("0")
        sub_result, sub_error = analyzer(function.instructions, sub_info, start, zero, config)
        sub_result, sub_error = assign_return_value(sub_result, function, instruction.lhs, sub_info,
                                                    config, analyzer, sub_error)
        result = sub_result.marginal(instruction.lhs)
        output = without(distribution, [instruction.lhs]) * result
        if sub_error.is_zero_dist():
            return output, error_prob
        # the observations of the body fail with this probability in every state of the caller
        return output, error_prob + sub_error * distribution.get_probability_mass()
