"""
-------------------------------
Draws inside expressions
-------------------------------

A *draw* is an expression that samples a value: a call of a distribution (``geometric(1/2)``,
``binomial(n, p)``, ``iid(D, n)``, …) or of a function of the program, a categorical choice, or
``sample(d)`` of a ``dist`` variable. The analysis handles a draw as the whole right-hand side of an
assignment (``x = geometric(1/2)``). A draw inside a larger expression, e.g.
``x = geometric(1/2) + 1``, is lowered by :func:`lower_draws`: every draw gets a fresh helper
variable, the helpers are drawn first (in the state before the assignment, from left to right,
inner draws before the draw whose argument contains them), then the expression is assigned with
the helpers in place of the draws, and the helpers are marginalised. Every occurrence is a draw of
its own, so ``geometric(1/2) + geometric(1/2)`` is the sum of two independent draws.

Arguments are lowered where they are values: all arguments of a distribution or a function of
the program, except the distribution argument of ``iid`` and both arguments of ``sample_pgf``
(a distribution expression and a generating function, no values). ``sample_pgf(G, x)`` reads
``G`` as a generating function in ``x``; drawn into a helper ``h``, it becomes
``sample_pgf(G[x := h], h)``. The values of a categorical
choice are evaluated in their own branch, so draws inside them are lowered there.

.. doctest::

    >>> from prodigy.pgcl.parser import parse_expr
    >>> names = iter(["h0", "h1", "h2"])
    >>> helpers, expr = lower_draws(parse_expr("geometric(1/2) + geometric(1/2) * 2"), set(),
    ...                             lambda: next(names))
    >>> [(name, str(draw)) for name, draw in helpers], str(expr)
    ([('h0', 'geometric(1/2)'), ('h1', 'geometric(1/2)')], 'h0 + (h1 * 2)')
    >>> lower_draws(parse_expr("x + 1"), set(), lambda: "h")
    ([], BinopExpr(operator=Binop.PLUS, lhs=VarExpr('x'), rhs=NatLitExpr(1)))
"""
from __future__ import annotations

import copy
from typing import Callable, Collection, List, Optional, Set, Tuple

from prodigy.pgcl.ast import CategoricalExpr, Expr, FunctionCallExpr, VarExpr
from prodigy.pgcl.ast.expressions import InferExpr, SampleExpr
from prodigy.pgcl.ast.walk import Walk, mut_expr_children, walk_expr
from prodigy.pgcl.util.ref import Mut

Helpers = List[Tuple[str, Expr]]
"""The helper variables of a lowering with the draws assigned to them, in the order of drawing."""

_DRAWS = (FunctionCallExpr, CategoricalExpr, SampleExpr)


def find_draw(expr: Expr) -> Optional[Expr]:
    """The first draw in ``expr`` (top-down, ``expr`` itself included), or ``None``. The call
    inside ``infer`` is no draw: ``infer`` computes a distribution, it draws no value."""
    if isinstance(expr, _DRAWS):
        return expr
    if isinstance(expr, InferExpr):
        return None
    for child in mut_expr_children(Mut.alloc(expr)):
        found = find_draw(child.val)
        if found is not None:
            return found
    return None


def _value_arguments(call: FunctionCallExpr, functions: Collection[str]) -> Set[int | str]:
    """The positions (index or name) of the arguments of ``call`` that are values."""
    positions: Set[int | str] = set(range(len(call.params[0]))) | set(call.params[1])
    if call.function in functions:
        return positions
    if call.function == "sample_pgf":
        return set()
    if call.function == "iid":
        return positions - {0, "dist"}
    return positions


def lower_draws(expr: Expr, functions: Collection[str],
                fresh: Callable[[], str]) -> Tuple[Helpers, Expr]:
    """
    Lowers the draws inside ``expr`` (see the module documentation). Returns the helpers with their
    draws and the expression that remains; a draw that is ``expr`` itself stays (with its arguments
    lowered). Without draws to lower, the helpers are empty and ``expr`` is returned unchanged.

    :param functions: the names of the program's functions (they hide distributions of the same
        name).
    :param fresh: returns a fresh variable name on every call.
    """
    helpers: Helpers = []

    def emit(draw: Expr) -> Expr:
        name = fresh()
        helpers.append((name, draw))
        return VarExpr(name)

    def lower(node: Expr, top: bool) -> Expr:
        if isinstance(node, FunctionCallExpr):
            call = _lower_arguments(node, functions, lambda arg: lower(arg, False))
            if top:
                return call
            if call.function == "sample_pgf" and call.function not in functions:
                return _emit_sample_pgf(call)
            return emit(call)
        if isinstance(node, (CategoricalExpr, SampleExpr)):
            return node if top else emit(node)
        if find_draw(node) is None:
            return node
        result = copy.deepcopy(node)
        for child in mut_expr_children(Mut.alloc(result)):
            child.val = lower(child.val, False)
        return result

    def _emit_sample_pgf(call: FunctionCallExpr) -> Expr:
        # sample_pgf(G, x) reads G as a generating function in x; drawn into a helper h, it is
        # sample_pgf(G[x := h], h)
        name = fresh()
        positional = call.params[0]
        if len(positional) == 2 and not call.params[1] and isinstance(positional[1], VarExpr):
            pgf = _rename(positional[0], positional[1].var, name)
            call = FunctionCallExpr(call.function, ([pgf, VarExpr(name)], {}))
        helpers.append((name, call))
        return VarExpr(name)

    lowered = lower(expr, True)
    return helpers, (lowered if helpers else expr)


def _rename(expr: Expr, old: str, new: str) -> Expr:
    """``expr`` with the variable ``old`` renamed to ``new``."""
    result = Mut.alloc(copy.deepcopy(expr))
    for ref in walk_expr(Walk.DOWN, result):
        if isinstance(ref.val, VarExpr) and ref.val.var == old:
            ref.val = VarExpr(new)
    return result.val


def _lower_arguments(call: FunctionCallExpr, functions: Collection[str],
                     lower: Callable[[Expr], Expr]) -> FunctionCallExpr:
    values = _value_arguments(call, functions)
    positional = [lower(arg) if index in values else arg for index, arg in enumerate(call.params[0])]
    named = {name: lower(arg) if name in values else arg for name, arg in call.params[1].items()}
    if positional == call.params[0] and named == call.params[1]:
        return call
    return FunctionCallExpr(call.function, (positional, named))
