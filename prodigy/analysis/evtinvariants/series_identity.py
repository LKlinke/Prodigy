"""
When does a fixed point of the EVT operator on closed forms say something about power series (#276)?

The EVT strategies check ``I = G + ⟦body⟧(⟨I⟩_guard)`` on the *closed form* of a candidate I (a rational
function), but Park's lemma needs the equation for the *power series* of I: the semantics of a program acts on
measures, i.e. on the coefficients. Both agree when every step of the computation is an operation on formal
power series. They need not agree when a step evaluates the closed form at a point where the series diverges:
marginalising a variable (substituting 1) sums infinitely many coefficients, and if the radius of convergence is
below 1 the closed form still has a finite value there (``1/(1 - 2x)`` at ``x = 1`` is ``-1``) while the series
gives ``∞``. A closed-form fixed point is then no fixed point of the series semantics, and the "upper bound" from
it can be wrong (``tests/analysis/test_evt_series_identity.py`` has such a loop: Pr[x = 5] is 1/2, a spurious
closed-form fixed point gives the upper bound 0).

For a candidate I with non-negative coefficients, either of the two criteria below proves that the closed-form
equation is the equation of the power series, so that I is a pre-fixed point of the series semantics and Park's
lemma applies. With finite mass (the result ``exact``) the first one holds anyway.

**(A) I converges on the open unit polydisc** U = ``{|x_i| < 1}``. For non-negative coefficients this holds iff the
diagonal f(t) = I(t, …, t) has radius of convergence ≥ 1, and by Pringsheim's theorem iff the reduced
denominator of f has no root in (0, 1) (:func:`converges_on_unit_polydisc`). Induction over the steps of
the computation: each intermediate closed form equals its (non-negative) series on U. The engines compute with
linear combinations, products with generating functions that converge on the closed polydisc (distributions),
coefficient extraction (Taylor coefficients at 0, formal), and substitutions that map U into its closure (monomials,
roots of unity times variables, ``y ↦ y·D(x)`` for a distribution D). A substitution that maps U into U keeps
the equality. The only one that reaches the boundary is evaluating a variable at 1 (marginal): for real
y ∈ [0, 1)^k and non-negative coefficients, monotone convergence (Abel) gives Σ_n a_n(y) = lim_{x→1⁻} I(x, y),
which is the closed form I(1, y) wherever its denominator does not vanish; so the marginal series converges on a
dense set of real points of [0, 1)^k, hence (domination by non-negative coefficients) on all of U, and by the
identity theorem equals the closed form there. If the denominator vanishes identically at x = 1, the closed form
is infinite or undefined and the engines stop with an error (``checked_marginal``, #242). At the end, both sides of
the equation are rational functions that equal convergent power series on U, so the series are equal.

**(B) The body is formal** (:func:`formal_loop`): the guard compares variables with natural-number literals, the
probabilities of choices do not depend on the state, and the only assignments are ``x := x + c``,
``x := x - c`` (``c`` a literal) and ``x := x + y``. For these, every step the engines take is an operation on
formal power series: Taylor coefficients at 0 for the guards, substitutions of monomials, and marginals only of
variables whose value is determined by the remaining ones up to finitely many choices (the copies the engines
make while evaluating ``x + 1``, and the old value of x afterwards). Such a marginal ``F ↦ F|_{v=1}`` is a ring
homomorphism of K[v][[w]], the power series in the other variables w with polynomial coefficients in v; from
F·Q = N it gives ``F|_{v=1}·Q(1, w) = N(1, w)``, so the closed form N(1, w)/Q(1, w) is the series ``F|_{v=1}``
(and if Q(1, w) vanishes identically the closed form is undefined, which stops the analysis). The radius of
convergence of I does not matter then: the biased random walk ``while (x > 0) {x := x - 1} [1/3] {x := x + 1}``
has the spurious invariant ``-1/2 + 3/2/(1 - 2x)``, which is a fixed point of the series semantics, and Park
gives a correct upper bound.

If neither criterion holds, the result is ``unknown``.
"""
from __future__ import annotations

from typing import Any, Collection, Optional, Sequence

import sympy

from prodigy.pgcl import (AsgnInstr, Binop, BinopExpr, BoolLitExpr, ChoiceInstr, Expr, IfInstr, Instr,
                          LoopInstr, NatLitExpr, RealLitExpr, SkipInstr, Unop, UnopExpr, VarExpr, WhileInstr)

_COMPARISONS = (Binop.LT, Binop.LEQ, Binop.GT, Binop.GEQ, Binop.EQ)


def converges_on_unit_polydisc(invariant: Any, variables: Sequence[str]) -> Optional[bool]:
    """Decides whether a generating function I with non-negative coefficients converges on the open unit
    polydisc, criterion (A) of this module.

    Substitutes t for every variable; f(t) = I(t, …, t) has non-negative coefficients, and its radius of
    convergence is the smallest positive root of its reduced denominator q (Pringsheim). I converges on the open
    polydisc iff f converges on [0, 1), i.e. iff q has no root in (0, 1). A root at 1 is allowed (infinite mass,
    e.g. ``(1 + x)/(1 - x)``). Returns None when this cannot be decided (parameters, not rational, q(0) = 0).
    """
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import sym
    try:
        expr = sym(invariant)
        names = set(variables)
        if any(s.name not in names for s in expr.free_symbols):
            return None
        t = sympy.Symbol("t__radius", positive=True)
        f = sympy.cancel(sympy.together(expr.subs({s: t for s in expr.free_symbols})))
        if not f.is_rational_function(t):
            return None
        _, denominator = sympy.fraction(f)
        poly = sympy.Poly(denominator, t)
        if poly.eval(0) == 0:
            return None
        one = sympy.Poly(t - 1, t)
        while not poly.is_zero and poly.degree() > 0 and poly.eval(1) == 0:
            poly = poly.quo(one)
        return poly.count_roots(0, 1) == 0
    except Exception:  # pylint: disable=broad-except
        return None


def _constant(expr: Expr, variables: Collection[str]) -> bool:
    """``expr`` contains no program variable (parameters are constants)."""
    if isinstance(expr, VarExpr):
        return expr.var not in variables
    if isinstance(expr, BinopExpr):
        return _constant(expr.lhs, variables) and _constant(expr.rhs, variables)
    if isinstance(expr, UnopExpr):
        return _constant(expr.expr, variables)
    return isinstance(expr, (NatLitExpr, BoolLitExpr, RealLitExpr))


def _formal_guard(cond: Expr, variables: Collection[str]) -> bool:
    if isinstance(cond, BoolLitExpr):
        return True
    if isinstance(cond, VarExpr):
        return cond.var in variables
    if isinstance(cond, UnopExpr):
        return cond.operator == Unop.NEG and _formal_guard(cond.expr, variables)
    if not isinstance(cond, BinopExpr):
        return False
    if cond.operator in (Binop.AND, Binop.OR):
        return _formal_guard(cond.lhs, variables) and _formal_guard(cond.rhs, variables)
    if cond.operator in _COMPARISONS:
        sides = (cond.lhs, cond.rhs)
        return any(isinstance(a, VarExpr) and a.var in variables and isinstance(b, NatLitExpr)
                   for a, b in (sides, sides[::-1]))
    return False


def _formal_assignment(instr: AsgnInstr, variables: Collection[str]) -> bool:
    target = str(instr.lhs)
    rhs = instr.rhs

    def is_target(expr: Expr) -> bool:
        return isinstance(expr, VarExpr) and expr.var == target

    if is_target(rhs):
        return True
    if not isinstance(rhs, BinopExpr):
        return False
    if rhs.operator == Binop.PLUS:
        return any(is_target(a) and (isinstance(b, NatLitExpr) or (isinstance(b, VarExpr) and b.var in variables))
                   for a, b in ((rhs.lhs, rhs.rhs), (rhs.rhs, rhs.lhs)))
    if rhs.operator == Binop.MINUS:
        return is_target(rhs.lhs) and isinstance(rhs.rhs, NatLitExpr)
    return False


def _formal_instructions(instructions: Sequence[Instr], variables: Collection[str]) -> bool:
    for instr in instructions:
        if isinstance(instr, SkipInstr):
            continue
        if isinstance(instr, AsgnInstr):
            if not _formal_assignment(instr, variables):
                return False
        elif isinstance(instr, ChoiceInstr):
            if not (_constant(instr.prob, variables) and _formal_instructions(instr.lhs, variables)
                    and _formal_instructions(instr.rhs, variables)):
                return False
        elif isinstance(instr, IfInstr):
            if not (_formal_guard(instr.cond, variables) and _formal_instructions(instr.true, variables)
                    and _formal_instructions(instr.false, variables)):
                return False
        elif isinstance(instr, LoopInstr):
            if not _formal_instructions(instr.body, variables):
                return False
        else:
            return False
    return True


def formal_loop(loop: WhileInstr, variables: Collection[str]) -> bool:
    """Whether the guard and the body of ``loop`` only use operations that act on formal power series,
    criterion (B) of this module. ``variables`` are the program variables; other names are parameters."""
    names = {str(v) for v in variables}
    return _formal_guard(loop.cond, names) and _formal_instructions(loop.body, names)
