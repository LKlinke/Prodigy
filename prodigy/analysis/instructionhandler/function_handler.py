from __future__ import annotations

import logging
from typing import Dict

from prodigy.language import (AsgnInstr, Binop, BinopExpr, Expr, FunctionCallExpr, NatLitExpr,
                          VarExpr, sample_predefined_functions as distr_functions)
from prodigy.language.ast.types import DistributionType

from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.instructionhandler import sample_handler
from prodigy.distribution.distribution import constant_value
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def _is_constant(expr: Expr) -> bool:
    """Whether ``expr`` has the same value in every state (a literal, or literals with operators)."""
    if isinstance(expr, NatLitExpr):
        return True
    return isinstance(expr, BinopExpr) and constant_value(expr) is not None


def handle(frame: Frame, instruction: AsgnInstr, state: ProgramState) -> ProgramState:
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
    prog_info, config = frame.program, frame.config
    distribution, error_prob = state
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.nested_inference import (assign_return_value, call_in_joint,
                                                   function_program_info, without)
    assert isinstance(instruction.rhs, FunctionCallExpr)

    if instruction.rhs.function in distr_functions:
        return sample_handler.handle(frame, instruction, state)

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
        return call_in_joint(frame, function, assignments, bindings, instruction.lhs, state)

    start = config.factory.one(*function.variables)
    for var, expr in assignments.items():
        start = start.update(BinopExpr(Binop.EQ, VarExpr(var), expr))
    body = frame.for_program(function_program_info(prog_info, function, bindings))
    sub_state = body.run(function.instructions, ProgramState(start, config.factory.from_expr("0")))
    sub_result, sub_error = assign_return_value(body, function, instruction.lhs, sub_state)
    result = sub_result.marginal(instruction.lhs)
    output = without(distribution, [instruction.lhs]) * result
    if sub_error.is_zero_dist():
        return ProgramState(output, error_prob)
    # the observations of the body fail with this probability in every state of the caller
    return ProgramState(output, error_prob + sub_error * distribution.get_probability_mass())
