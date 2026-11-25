"""
-----------------------------------
Checks of a program before its run
-----------------------------------

Checks that need the proof rules of :mod:`prodigy.distribution` and therefore cannot be part of the type checker
in :mod:`prodigy.language` (which imports nothing above it, ``docs/source/architecture.md``, T3). Every entry
path runs them after the type checker: the analysis session, the CLI commands and the GUI's check of the
source.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, List, Optional

from prodigy.distribution.finite_measure import check_probability_generating_function
from prodigy.language import (CategoricalExpr, Expr, FunctionCallExpr, Instr, NatType, Program, SampleExpr, VarExpr,
                              substitute)
from prodigy.language.ast.walk import Walk, walk_expr, walk_instr_exprs, walk_instrs
from prodigy.language.check import CheckFail, check_program
from prodigy.language.diagnostics import check_failure_diagnostic
from prodigy.language.ppy_printer import spell
from prodigy.language.util.ref import Mut
from prodigy.language.engine_text import engine_text_of

__all__ = ["first_failure", "refute_sampling_pgfs", "type_error"]


def _calls(instructions: List[Instr], returns: Optional[Expr] = None) -> Iterator[FunctionCallExpr]:
    for instr_ref in walk_instrs(Walk.DOWN, instructions):
        for expr_ref in walk_instr_exprs(Walk.DOWN, instr_ref.val):
            if isinstance(expr_ref.val, FunctionCallExpr):
                yield expr_ref.val
    if returns is not None:
        yield from (ref.val for ref in walk_expr(Walk.DOWN, Mut.alloc(returns))
                    if isinstance(ref.val, FunctionCallExpr))


def refute_sampling_pgfs(program: Program) -> Optional[CheckFail]:
    """The first ``sample_pgf(G, x)`` or ``iid(G, n)`` of a type-correct ``program`` whose generating function
    ``G`` is refuted to be a probability generating function (non-negative coefficients, mass 1, #424), e.g.
    ``sample_pgf(2*x, x)``, as a failure at ``G``; the functions are checked first, then the instructions (the
    order of the type checker). Only refutations are reported: ``G`` with names that are neither program
    variables nor parameters (e.g. variables of a function) and ``G`` that are not proven are left to the
    analysis, which checks every sampling it executes (``prodigy.analysis.instructionhandler.sample_handler``).
    """
    for function in program.functions.values():
        failure = _refute_in(Program.from_function(function, program, copy_functions=True), function.instructions,
                             function.returns)
        if failure is not None:
            return failure
    return _refute_in(program, program.instructions)


def _refute_in(program: Program, instructions: List[Instr], returns: Optional[Expr] = None) -> Optional[CheckFail]:
    for call in _calls(instructions, returns):
        if call.function not in program.functions:
            failure = _refute_call(program, call)
            if failure is not None:
                return failure
    return None


def _refute_call(program: Program, expr: FunctionCallExpr) -> Optional[CheckFail]:
    if expr.function not in ('iid', 'sample_pgf') or len(expr.params[0]) < 2:
        return None
    generating_function, variable = expr.params[0][:2]
    if isinstance(generating_function, (FunctionCallExpr, CategoricalExpr, SampleExpr)):
        return None  # a distribution expression; its arguments are checked on their own
    names = {ref.val.var for ref in walk_expr(Walk.DOWN, Mut.alloc(generating_function))
             if isinstance(ref.val, VarExpr)}
    if expr.function == 'sample_pgf' and isinstance(variable, VarExpr):
        names.add(variable.var)
    parameters = {str(name): typ for name, typ in program.parameters.items()}
    if not names <= set(program.variables) | set(parameters):
        return None
    natural = tuple(sorted(name for name, typ in parameters.items() if isinstance(typ, NatType)))
    try:
        verdict, reason = check_probability_generating_function(
            engine_text_of(generating_function), tuple(sorted(names - set(parameters))), natural)
    except Exception:  # pylint: disable=broad-except
        return None  # the analysis checks it again and reports what goes wrong
    if verdict is False:
        return CheckFail(generating_function,
                         f"{spell(expr)} needs a probability generating function (non-negative coefficients, "
                         f"mass 1), but {spell(generating_function)} is not one: {reason}.")
    return None


def type_error(program: Any, source: str) -> Optional[Dict[str, Any]]:
    """The diagnostic (:mod:`prodigy.language.diagnostics`) of the first failure of ``program`` (:func:`first_failure`,
    after substituting the constants), or None; the session, the CLI and the GUI report it."""
    checked_program = copy.deepcopy(program)
    substitute.substitute_constants(checked_program)
    failure = first_failure(checked_program)
    return check_failure_diagnostic(source, failure) if isinstance(failure, CheckFail) else None


def first_failure(checked_program: Any) -> Optional[CheckFail]:
    """The type checker's first failure, else the first refuted generating function of ``sample_pgf``/``iid``
    (:func:`~prodigy.analysis.checks.refute_sampling_pgfs`)."""
    return check_program(checked_program) or refute_sampling_pgfs(checked_program)
