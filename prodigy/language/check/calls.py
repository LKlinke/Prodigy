"""
Calls: the signatures of the predefined distributions (:data:`sample_predefined_functions`) and the check of a
call of a distribution or of a function of the program (:func:`check_function_call`).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Optional, Set

# mypy complains here, but this isn't actually wrong
# the next version of frozendict will likely fix this issue
# https://github.com/Marco-Sulla/python-frozendict/issues/62
from frozendict import frozendict

from prodigy.language.ast import (Binop, BinopExpr, CategoricalExpr, Expr, FunctionCallExpr, NatLitExpr, NatType,
                                  Program, RealLitExpr, RealType, SampleExpr, Unop, UnopExpr, Var, VarDecl, VarExpr)

from .context import CheckFail, Ctx, expect, spell_node

sample_predefined_functions: Mapping[Var, List[VarDecl]] = frozendict({
    'unif_d': [VarDecl('start', NatType(None)),
               VarDecl('end', NatType(None))],
    'unif': [VarDecl('start', NatType(None)),
             VarDecl('end', NatType(None))],
    'geometric': [VarDecl('p', RealType())],
    'bernoulli': [VarDecl('p', RealType())],
    'poisson': [VarDecl('p', RealType())],
    'logdist': [VarDecl('p', RealType())],
    'binomial': [VarDecl('n', NatType(None)),
                 VarDecl('p', RealType())],
    'negative_binomial': [VarDecl('r', NatType(None)),
                          VarDecl('p', RealType())],
    # the distribution of iid/sample_pgf is a distribution expression or a generating function,
    # whose coefficients are real
    'iid': [VarDecl('dist', RealType()),
            VarDecl('var', NatType(None))],
    'sample_pgf': [VarDecl("dist", RealType()),
                   VarDecl("var", NatType(None))]
})



def _check_sample_count(program: Program,
                        expr: FunctionCallExpr) -> Optional[CheckFail]:
    """
    Check the second argument of ``iid`` and ``sample_pgf``.

    ``iid(D, n)`` sums ``n`` independent samples of ``D``; ``n`` must be a program variable, a
    natural number (constants are substituted before) or a draw such as ``geometric(1/2)``, the
    analysis cannot handle arbitrary expressions or parameters there. ``sample_pgf(G, x)`` reads ``G`` as a
    generating function in the program variable ``x``.
    """
    if expr.function not in ('iid', 'sample_pgf') or len(expr.params[0]) < 2:
        return None
    arg = expr.params[0][1]
    if isinstance(arg, VarExpr) and arg.var in program.variables:
        return None
    if expr.function == 'iid':
        # a draw is drawn into a helper variable first (prodigy.language.draws)
        if isinstance(arg, (NatLitExpr, FunctionCallExpr, CategoricalExpr, SampleExpr)):
            return None
        return CheckFail(
            arg, f"The number of samples in {spell_node(expr)} must be a program variable "
            f"or a natural number, got {spell_node(arg)}.")
    return CheckFail(
        arg, f"The second argument of {spell_node(expr)} must be the program variable "
        f"of the generating function, got {spell_node(arg)}.")


_LITERAL_ARITHMETIC: Dict[Binop, Callable[[Fraction, Fraction], Optional[Fraction]]] = {
    Binop.PLUS: lambda lhs, rhs: lhs + rhs,
    Binop.MINUS: lambda lhs, rhs: lhs - rhs,
    Binop.TIMES: lambda lhs, rhs: lhs * rhs,
    Binop.DIVIDE: lambda lhs, rhs: None if rhs == 0 else lhs / rhs,
}


def _literal_value(expr: Expr) -> Optional[Fraction]:
    """The value of an expression built from number literals with ``+``, ``-``, ``*`` and ``/``,
    or None if it contains anything else (variables, parameters, other operators)."""
    if isinstance(expr, NatLitExpr):
        return Fraction(expr.value)
    if isinstance(expr, RealLitExpr):
        return None if expr.is_infinite() else expr.to_fraction()
    if isinstance(expr, UnopExpr) and expr.operator == Unop.NEG:
        value = _literal_value(expr.expr)
        return None if value is None else -value
    if isinstance(expr, BinopExpr) and expr.operator in _LITERAL_ARITHMETIC:
        lhs, rhs = _literal_value(expr.lhs), _literal_value(expr.rhs)
        return None if lhs is None or rhs is None else _LITERAL_ARITHMETIC[expr.operator](lhs, rhs)
    return None


def _check_negative_binomial(expr: FunctionCallExpr) -> Optional[CheckFail]:
    """
    Check the arguments of ``negative_binomial(r, p)``.

    ``r`` must be a natural number (constants are substituted before), because the analysis takes
    the ``r``-th power of the generating function of ``geometric(p)``. A literal ``p`` must lie in
    ``(0, 1]``; otherwise it is not a distribution. A ``p`` with parameters or program variables is
    checked by the analysis (in every state, for program variables).
    """
    if expr.function != 'negative_binomial' or len(expr.params[0]) != 2:
        return None
    r, p = expr.params[0]
    if not isinstance(r, NatLitExpr):
        return CheckFail(
            r, f"The number of successes in {spell_node(expr)} must be a natural number, got {spell_node(r)}.")
    value = _literal_value(p)
    if value is not None and not 0 < value <= 1:
        return CheckFail(
            p, f"The success probability in {spell_node(expr)} must be in (0, 1], got {spell_node(p)}.")
    return None


def check_function_call(ctx: Ctx, expr: FunctionCallExpr) -> Optional[CheckFail]:
    """
    Check a call of a function of the program or of a predefined distribution: the number of its arguments,
    their names and types, and the extra rules of ``iid``, ``sample_pgf`` and ``negative_binomial``.
    """
    ctx = ctx.checked()
    param_list: List[VarDecl]
    if expr.function in ctx.program.functions:
        param_list = ctx.program.functions[expr.function].declarations
    else:
        if expr.function not in ctx.predefined_functions:
            return CheckFail(expr, f"Unknown function: {expr.function}")
        param_list = ctx.predefined_functions[expr.function]
        failure = _check_distribution_call(ctx.program, expr, param_list)
        if failure is not None:
            return failure
    if len(expr.params[0]) > len(param_list):
        return CheckFail(expr, "Too many parameters")
    # whether a generating function of sample_pgf or iid is one is checked by the analysis before it runs
    # (prodigy.analysis.checks), which has the proof rules
    return _check_arguments(ctx, expr, param_list)


def _check_distribution_call(program: Program, expr: FunctionCallExpr,
                             param_list: List[VarDecl]) -> Optional[CheckFail]:
    # the analysis reads the parameters of the predefined distributions by position, and they
    # have no default values
    if expr.params[1]:
        return CheckFail(expr, f"{expr.function} takes no named parameters")
    if len(expr.params[0]) < len(param_list):
        return CheckFail(expr, "Too few parameters")
    return _check_sample_count(program, expr) or _check_negative_binomial(expr)


def _check_arguments(ctx: Ctx, expr: FunctionCallExpr, param_list: List[VarDecl]) -> Optional[CheckFail]:
    covered_names: Set[Var] = set()
    # the positional arguments; the parameters after them may be given by name (below)
    for decl, param_expr in zip(param_list, expr.params[0], strict=False):
        failure = expect(ctx, param_expr, decl.typ)
        if failure is not None:
            return failure
        covered_names.add(decl.var)

    param_dict = {decl.var: decl.typ for decl in param_list}
    for var, param_expr in expr.params[1].items():
        if var not in param_dict:
            return CheckFail(
                expr, f"Unknown variable in named function parameters: {var}")
        if var in covered_names:
            return CheckFail(
                expr, f"Variable {var} is covered by multiple parameters")
        failure = expect(ctx, param_expr, param_dict[var])
        if failure is not None:
            return failure
        covered_names.add(var)
    return None
