"""
``Pr[d …]`` and ``Ex[d]``: the numbers of a ``dist`` value, computed from its generating function and substituted into
the instruction that uses them.
"""
from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sympy

from prodigy.analysis.exceptions import UnsupportedFeatureError
from prodigy.analysis.context import Frame, ProgramState, Semantics
from prodigy.distribution import Distribution
from prodigy.language import (Binop, BinopExpr, BoolLitExpr, Expr, ExprClass, Instr, NatLitExpr, RealLitExpr, Unop,
                              UnopExpr, VarExpr, WhileInstr)
from prodigy.language.ast.expressions import ExpectationExpr, ProbabilityExpr
from prodigy.language.ast.walk import Walk, mut_expr_children, walk_expr
from prodigy.language.util.ref import Mut
from prodigy.analysis.nested_inference.sites import (ClosedDist, DistSite, VALUE_VARIABLE, _spell, _sum, _sym,
                                                     _unassigned, dist_variables, split_by_values)
from prodigy.analysis.nested_inference.functions import _constant_argument, closed_form

def _dist_queries(instr: Instr) -> List[Union[ProbabilityExpr, ExpectationExpr]]:
    """The ``Pr[d …]`` and ``Ex[d]`` in the instruction's own expressions (not in its blocks)."""
    found: List[Union[ProbabilityExpr, ExpectationExpr]] = []
    for value in instr.__dict__.values():
        if isinstance(value, ExprClass):
            for ref in walk_expr(Walk.DOWN, Mut.alloc(value.cast())):
                if isinstance(ref.val, (ProbabilityExpr, ExpectationExpr)):
                    found.append(ref.val)
    return found


def _queried_variable(node: Union[ProbabilityExpr, ExpectationExpr], dist_vars: Set[str]) -> str:
    """The ``dist`` variable of ``Pr[d …]`` or ``Ex[d]`` (the type checker ensures the form)."""
    if isinstance(node, ExpectationExpr):
        assert isinstance(node.expr, VarExpr)
        return node.expr.var
    cond = node.cond
    assert isinstance(cond, BinopExpr)
    if isinstance(cond.lhs, VarExpr) and cond.lhs.var in dist_vars:
        return cond.lhs.var
    assert isinstance(cond.rhs, VarExpr)
    return cond.rhs.var


def _constant_argument_or_none(expr: Expr) -> Optional[int]:
    try:
        return _constant_argument(expr)
    except UnsupportedFeatureError:
        return None


def query_value(frame: Frame, node: Union[ProbabilityExpr, ExpectationExpr], var: str, site: DistSite) -> Any:
    """The value of ``Pr[d …]`` or ``Ex[d]`` for the closed value ``site`` of ``d`` (sympy)."""
    closed = closed_form(frame, site)
    key = repr(node)  # the structure of the query, not its spelling
    if key in closed.values:
        return closed.values[key]
    value_var = VarExpr(VALUE_VARIABLE)
    try:
        try:
            value = _engine_value(node, var, closed, value_var)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            if not isinstance(node, ExpectationExpr):
                raise
            # e.g. GiNaC cannot take the limit of G'(z) at 1 if Ex[d] = ∞; SymPy can
            function = _sym(str(closed.distribution))
            points = [symbol for symbol in function.free_symbols if str(symbol) == VALUE_VARIABLE]
            if len(points) != 1 or len(function.free_symbols) != 1:
                raise
            value = sympy.limit(sympy.diff(function, points[0]), points[0], 1, "-")
    except UnsupportedFeatureError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise UnsupportedFeatureError(
            f"{_spell(node)} for {site.describe()} could not be computed exactly: {exc}") from exc
    value = sympy.nsimplify(value) if not value.has(sympy.oo) else value
    closed.values[key] = value
    return value


def _engine_value(node: Union[ProbabilityExpr, ExpectationExpr], var: str, closed: ClosedDist,
                  value_var: VarExpr) -> Any:
    if isinstance(node, ExpectationExpr):
        text = str(closed.distribution.get_expected_value_of(value_var)).strip()
        if text.lower() in ("infinity", "oo", "inf", "∞"):
            return sympy.oo
        value = _sym(text)
    else:
        cond = node.cond
        assert isinstance(cond, BinopExpr)
        dist_left = isinstance(cond.lhs, VarExpr) and cond.lhs.var == var
        other = cond.rhs if dist_left else cond.lhs
        number = _constant_argument_or_none(other)
        if number is None:
            raise UnsupportedFeatureError(
                f"{_spell(node)}: comparing a dist value with a number that depends on the program state "
                "is not implemented.")
        rewritten = BinopExpr(cond.operator, value_var, NatLitExpr(number)) if dist_left \
            else BinopExpr(cond.operator, NatLitExpr(number), value_var)
        value = _sym(closed.distribution.get_probability_of(rewritten))
    if value.free_symbols or value.has(sympy.zoo, sympy.nan):
        raise UnsupportedFeatureError(
            f"{_spell(node)} = {value} is not a number; Pr and Ex that depend on parameters are "
            "not implemented inside a program.")
    return value


def _to_sympy(expr: Expr, values: Dict[int, Any]) -> Optional[Any]:
    """``expr`` as a sympy value if it has no variables (with the values of ``Pr``/``Ex``)."""
    # pylint: disable=too-many-return-statements
    if id(expr) in values:
        return values[id(expr)]
    if isinstance(expr, NatLitExpr):
        return sympy.Integer(expr.value)
    if isinstance(expr, RealLitExpr):
        return sympy.oo if expr.is_infinite() else sympy.Rational(expr.to_fraction())
    if isinstance(expr, BoolLitExpr):
        return sympy.true if expr.value else sympy.false
    if isinstance(expr, UnopExpr) and expr.operator == Unop.NEG:
        inner = _to_sympy(expr.expr, values)
        return None if inner is None else sympy.Not(inner)
    if not isinstance(expr, BinopExpr):
        return None
    lhs, rhs = _to_sympy(expr.lhs, values), _to_sympy(expr.rhs, values)
    if lhs is None or rhs is None:
        return None
    operations = {
        Binop.PLUS: lambda a, b: a + b, Binop.MINUS: lambda a, b: a - b,
        Binop.TIMES: lambda a, b: a * b, Binop.DIVIDE: lambda a, b: a / b,
        Binop.POWER: lambda a, b: a ** b,
        Binop.EQ: sympy.Eq, Binop.LT: sympy.Lt, Binop.LEQ: sympy.Le,
        Binop.GT: sympy.Gt, Binop.GEQ: sympy.Ge,
        Binop.AND: sympy.And, Binop.OR: sympy.Or,
    }
    operation = operations.get(expr.operator)
    return None if operation is None else operation(lhs, rhs)


def _from_sympy(value: Any, where: Expr) -> Expr:
    if value is sympy.true or value is sympy.false:
        return BoolLitExpr(bool(value))
    if isinstance(value, (sympy.logic.boolalg.Boolean, sympy.core.relational.Relational)):
        raise UnsupportedFeatureError(f"{_spell(where)}: the comparison {value} cannot be decided exactly.")
    if value == sympy.oo:
        return RealLitExpr.infinity()
    if value.is_Rational:
        return RealLitExpr(Fraction(int(value.p), int(value.q)))
    if value is sympy.nan or value.has(sympy.nan, sympy.zoo):
        raise UnsupportedFeatureError(f"{_spell(where)} is undefined (nan).")
    raise UnsupportedFeatureError(
        f"{_spell(where)} = {value} is not a rational number; using it as a probability or in arithmetic "
        "of the program is not implemented (comparisons work).")


def _fold(expr: Expr, values: Dict[int, Any]) -> Expr:
    """Replaces the subexpressions with a resolved ``Pr``/``Ex`` by their value."""
    def contains_value(node: Expr) -> bool:
        return any(id(ref.val) in values for ref in walk_expr(Walk.DOWN, Mut.alloc(node)))

    root = Mut.alloc(expr)

    def visit(ref: Mut) -> None:
        node = ref.val
        if not contains_value(node):
            return
        value = _to_sympy(node, values)
        if value is not None:
            ref.val = _from_sympy(value, node)
            return
        for child in mut_expr_children(ref):
            visit(child)

    visit(root)
    return root.val


def resolve_queries(frame: Frame, instr: Instr, state: ProgramState,
                    untraced: Semantics) -> Optional[ProgramState]:
    """Runs ``instr`` with its ``Pr[d …]`` and ``Ex[d]`` replaced by their values, split by the
    values of the ``dist`` variables they read; ``None`` if it has none. The instruction with the values
    (the shell) runs with ``untraced``: it stands for ``instr``, whose step is traced already."""
    nodes = _dist_queries(instr)
    if not nodes:
        return None
    distribution, error_prob = state
    dist_vars = dist_variables(frame.program)
    variables = sorted({_queried_variable(node, dist_vars) for node in nodes})
    results: List[Tuple[Distribution, Distribution]] = []
    for part, sites in split_by_values(distribution, variables, frame.program, frame.context.dist_registry):
        values: Dict[int, Any] = {}
        for node in nodes:
            var = _queried_variable(node, dist_vars)
            site = sites[var]
            if site is None:
                if _sym(part.get_probability_mass()) == 0:
                    break
                raise _unassigned(var, _spell(node))
            if not site.closed:
                raise UnsupportedFeatureError(
                    f"{_spell(node)}: the value of {var} depends on the program state with infinite support "
                    f"({site.describe()}); Pr and Ex of such a dist are not implemented.")
            values[id(node)] = query_value(frame, node, var, site)
        else:
            shell = copy.copy(instr)
            for name, value in list(instr.__dict__.items()):
                if isinstance(value, ExprClass):
                    setattr(shell, name, _fold(_copy_keeping_ids(value.cast(), values), values))
            results.append(_run_shell(frame, shell, instr, ProgramState(part, error_prob * "0"), untraced))
    return _sum(results, distribution, error_prob)


def _copy_keeping_ids(expr: Expr, values: Dict[int, Any]) -> Expr:
    """A copy of ``expr`` whose resolved nodes keep their identity (they are replaced anyway)."""
    memo = {id(node): node for node in _nodes(expr) if id(node) in values}
    return copy.deepcopy(expr, memo)


def _nodes(expr: Expr) -> List[Expr]:
    return [ref.val for ref in walk_expr(Walk.DOWN, Mut.alloc(expr))]


def _run_shell(frame: Frame, shell: Instr, original: Instr, state: ProgramState,
               untraced: Semantics) -> ProgramState:
    if isinstance(shell, WhileInstr) and isinstance(original, WhileInstr):
        with frame.context.loop_alias(shell, original):
            return untraced(frame, shell, state)
    return untraced(frame, shell, state)
