"""
Probabilities that are rational in a counter, applied as operators on the generating function (#323).

A probabilistic choice ``{A} [p(k)] {B}`` whose probability depends on a program variable ``k`` with infinitely
many values cannot be computed by a case split over the values of k
(:mod:`~prodigy.analysis.instructionhandler.state_dependent`). For a probability that is *rational in k*, the
weighting has a closed form: weighting the coefficient ``g_n`` of ``k^n`` with ``p(n)`` is the operator ``p(θ)``
with the Euler operator ``θ = k·∂/∂k`` (``θ k^n = n k^n``). The forward semantics is
``⟦A⟧(p(θ) G) + ⟦B⟧((1 - p)(θ) G)``.

This module covers the probabilities rational in k with rational coefficients whose partial fractions are
``p(k) = α + Σ_i β_i/(k + a_i)`` with distinct natural numbers ``a_i ≥ 1`` (simple poles at negative integers),
e.g. ``1/(k + 1)``, ``k/(k + 1) = 1 - 1/(k + 1)``, ``1/2 + 1/(2k + 2)`` or ``2/((k + 1)(k + 2))``. Whether p is a
probability for every value of k is decided exactly: with p = N/D and D > 0 on the naturals, ``N ≥ 0`` and
``D - N ≥ 0`` on all naturals is checked directly up to Cauchy's root bound and by the leading coefficient beyond
(:func:`prodigy.analysis.evtinvariants.exp_polynomial.eventually_nonneg`). Then

    p(θ) G = α·G + Σ_i β_i·H_(a_i)   with   H_a = k^(-a) ∫_0^k t^(a-1) G(t) dt,

since ``∫_0^k t^(n+a-1) dt = k^(n+a)/(n+a)``. For ``1/(k + 1)`` followed by ``k := k + 1`` this is just
``∫_0^k G``. The integral is proposed by a formula for exponential polynomials in k, otherwise by SymPy's
``integrate`` (:func:`prodigy.analysis.evtinvariants.exp_polynomial.integral_from_zero`); the proposal is certified
by differentiation and evaluation at 0, so no symbolic integration has to be trusted. A given EVT invariant of a
loop is checked with the same operator: the check needs the integral of an exponential polynomial only, whose
formula needs no integration. The computation works on the
SymPy form of the distribution and gives the result back to the distribution's engine, so it is the same on all
engines. Anything else (another shape of p, e.g. a polynomial or a double pole, or an integral that is not certified) is
not implemented. ``x = bernoulli(p)`` with such a p is the choice ``{x = 1} [p] {x = 0}`` and uses the same operator.
``x = unif(0, n)`` uses the operator ``1/(θ + 1)`` in n after a substitution (:func:`uniform_up_to`, #385).
The probability ``x/n`` of two program variables is ``θ_x`` followed by ``1/θ_n`` (:func:`apply_ratio`, #237).

**Removable singularities.** The closed form of ``H_a`` is ``F/k^a`` with ``F = ∫_0^k t^(a-1) G``, whose series
starts at ``k^a``; so the closed form has a removable singularity at ``k = 0``, while its series is an ordinary power
series. The engines treat a closed form as its Taylor series at the origin (coefficients by series expansion or
derivatives); where an engine instead substitutes ``k = 0``, both numerator and denominator vanish, so it gets an
undefined value (``nan``, ``zoo`` or a division by zero), never a wrong finite one, which Prodigy reports as
undefined instead of as a result (#259). The tests filter and query at ``k = 0`` on all engines.
"""
from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Optional, Set, Tuple

import sympy

from prodigy.pgcl import Binop, BinopExpr, Expr, NatLitExpr, RealLitExpr, VarExpr

from prodigy.analysis.evtinvariants.exp_polynomial import eventually_nonneg, integral_from_zero
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.equality import decide_equal, sympify_distribution, sympy_text


class CounterProbability(NamedTuple):
    """A probability ``α + Σ_i β_i/(k + a_i)`` in the counter ``k`` (module docstring); ``poles`` lists
    ``(β_i, a_i)``."""
    variable: str
    alpha: Fraction
    poles: Tuple[Tuple[Fraction, int], ...]

    def __str__(self) -> str:
        return " + ".join([str(self.alpha)] + [f"{beta}/({self.variable} + {shift})" for beta, shift in self.poles])


def _to_sympy(expr: Expr, counter: str) -> Optional[sympy.Expr]:
    """The arithmetic expression ``expr`` over the counter (rationals, ``+ - * /``) as a SymPy expression, or
    None for anything else. A subtraction is accepted only if it can never be negative (in ``nat`` arithmetic it
    would be a monus otherwise)."""
    if isinstance(expr, NatLitExpr):
        return sympy.Integer(expr.value)
    if isinstance(expr, RealLitExpr):
        value = expr.to_fraction()
        return sympy.Rational(value.numerator, value.denominator)
    if isinstance(expr, VarExpr):
        return sympy.Symbol(counter) if expr.var == counter else None
    if not isinstance(expr, BinopExpr):
        return None
    lhs, rhs = _to_sympy(expr.lhs, counter), _to_sympy(expr.rhs, counter)
    if lhs is None or rhs is None:
        return None
    if expr.operator == Binop.PLUS:
        return lhs + rhs
    if expr.operator == Binop.TIMES:
        return lhs * rhs
    if expr.operator == Binop.DIVIDE:
        return lhs / rhs if rhs != 0 else None
    if expr.operator == Binop.MINUS:
        difference = lhs - rhs
        if _nonneg_for_naturals(difference, sympy.Symbol(counter)) is not True:
            return None
        return difference
    return None


def _nonneg_for_naturals(expr: sympy.Expr, k: sympy.Symbol) -> Optional[bool]:
    """Decides ``expr(n) ≥ 0`` for all naturals n, for a rational function with rational coefficients whose
    denominator has no root in the naturals; None for other expressions."""
    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(expr)))
    try:
        num, den = sympy.Poly(numerator, k), sympy.Poly(denominator, k)
    except sympy.PolynomialError:
        return None
    if not all(c.is_Rational for c in num.all_coeffs() + den.all_coeffs()):
        return None
    # the denominator must have one sign on all naturals (and no root there): then p ≥ 0 iff ±numerator ≥ 0
    for sign in (1, -1):
        if eventually_nonneg([Fraction(str(sign * c)) for c in den.all_coeffs()], -1, strict=True) is True:
            return eventually_nonneg([Fraction(str(sign * c)) for c in num.all_coeffs()], -1)
    return None


Poles = Tuple[Tuple[Fraction, int], ...]


def _partial_fractions(expr: sympy.Expr, k: sympy.Symbol) -> Optional[Tuple[Fraction, Poles]]:
    """``(α, ((β_i, a_i), …))`` if ``expr = α + Σ_i β_i/(k + a_i)`` with rationals and distinct naturals a_i ≥ 1."""
    expr = sympy.cancel(sympy.together(expr))
    if not expr.has(k):
        return (Fraction(str(expr)), ()) if expr.is_Rational else None
    numerator, denominator = sympy.fraction(expr)
    try:
        num, den = sympy.Poly(numerator, k), sympy.Poly(denominator, k)
    except sympy.PolynomialError:
        return None
    if not all(c.is_Rational for c in num.all_coeffs() + den.all_coeffs()) or num.degree() > den.degree():
        return None
    roots = sympy.roots(den, multiple=True)
    if len(roots) != den.degree() or len(set(roots)) != len(roots) \
            or not all(r.is_Integer and r <= -1 for r in roots):
        return None
    alpha, remainder = sympy.div(num, den)
    poles = []
    derivative = den.diff(k)
    for root in sorted(roots, reverse=True):
        beta = remainder.eval(root) / derivative.eval(root)  # residue at a simple pole
        if beta != 0:
            poles.append((Fraction(str(beta)), int(-root)))
    return Fraction(str(alpha.as_expr())), tuple(poles)


def counter_probability(probability: Expr, variables: Set[str]) -> Optional[CounterProbability]:
    """The probability as ``α + Σ_i β_i/(k + a_i)`` in its only program variable k, if it has this shape and is
    a probability for every value of k; otherwise None."""
    if len(variables) != 1:
        return None
    counter = next(iter(variables))
    k = sympy.Symbol(counter)
    expr = _to_sympy(probability, counter)
    if expr is None:
        return None
    shape = _partial_fractions(expr, k)
    if shape is None or not shape[1]:
        return None
    if _nonneg_for_naturals(expr, k) is not True or _nonneg_for_naturals(1 - expr, k) is not True:
        return None
    return CounterProbability(counter, shape[0], shape[1])


def _rational(value: Fraction) -> sympy.Rational:
    return sympy.Rational(value.numerator, value.denominator)


def apply_operator(probability: CounterProbability, distribution: Distribution,
                   construct: str) -> tuple[Distribution, Distribution]:
    """The two parts ``p(θ) G`` and ``(1 - p)(θ) G`` of ``distribution`` G (module docstring). Raises
    ``NotImplementedError`` (naming ``construct``) if an integral cannot be computed and certified."""
    function = sympify_distribution(distribution)
    names = distribution.get_variables()
    k = sympy.Symbol(probability.variable)
    left = _rational(probability.alpha) * function
    for beta, shift in probability.poles:
        integral = integral_from_zero(function * k ** (shift - 1), k, [sympy.Symbol(v) for v in names])
        if integral is None:
            integrand = str(function * k ** (shift - 1))
            integrand = integrand if len(integrand) <= 200 else integrand[:200] + " …"
            raise NotImplementedError(
                f"{construct} depends on the program variable {probability.variable}, which can take infinitely "
                f"many values here. Prodigy applies a probability α + Σ β/({probability.variable} + a) as an "
                f"operator on the generating function, which needs the integral from 0 of "
                f"{integrand} in {probability.variable}. No antiderivative was found that passes the "
                f"check (derivative, value 0 at {probability.variable} = 0, analytic at 0): Prodigy integrates sums "
                f"of terms {probability.variable}^n·exp(c·{probability.variable}) (e.g. a Poisson distribution) by "
                f"a formula and other integrands without parameters with SymPy; this one is not implemented.")
        left += _rational(beta) * integral / k ** shift
    right = function - left
    return _back(left, distribution), _back(right, distribution)


def _back(expr: sympy.Expr, distribution: Distribution) -> Distribution:
    """The SymPy expression ``expr`` over the variables of ``distribution`` as a distribution of its engine."""
    # exp factors merged again (exp(k - 1), not exp(-1)*exp(k)); Euler's number as exp(1), never "E"
    text = sympy_text(sympy.powsimp(sympy.together(sympy.expand(expr)), combine="exp")).replace("**", "^")
    result = distribution.factory().from_expr(text, *sorted(distribution.get_variables()))
    parameters = distribution.get_parameters()
    return result.set_parameters(*parameters) if parameters else result


def uniform_up_to(distribution: Distribution, variable: str, bound: str, construct: str) -> Distribution:
    """
    ``variable = unif(0, bound)`` on ``distribution`` G, for a program variable ``bound`` that takes infinitely many
    values (#385). Raises ``NotImplementedError`` (naming ``construct``) if the integral below cannot be computed
    and certified.

    With x = ``variable``, n = ``bound`` and G with x marginalised, a term ``n^k`` becomes
    ``n^k (1 + x + … + x^k)/(k + 1)``. Appending the geometric sum is the substitution
    ``H = (G(n) - x·G(n·x))/(1 - x)``, and the weight ``1/(k + 1)`` is the operator ``1/(θ + 1)`` of the module
    docstring: ``F = (1/n) ∫_0^n H(t) dt``. Since ``∫_0^n x·G(t·x) dt = I(n·x)`` for ``I = ∫_0^n G``, only the
    integral I is needed: ``F = (I(n) - I(n·x)) / (n (1 - x))``. I is proposed and certified by
    :func:`~prodigy.analysis.evtinvariants.exp_polynomial.integral_from_zero` (``I' = G``, ``I(0) = 0``, I analytic
    at the origin); in addition, the derivative of ``n·F`` in n is checked to be H. As ``n·F`` vanishes at n = 0,
    the division by n is exact: the term ``n^0`` of F is the term ``n^0`` of G (``unif(0, 0)`` is ``x = 0``).

    The closed form has removable singularities at n = 0 and x = 1 (module docstring); its series is an ordinary
    power series. The result is computed on the SymPy form and given back to the distribution's engine, as
    :func:`apply_operator` does.
    """
    if variable == bound:
        raise NotImplementedError(
            f"{construct} assigns to its own upper bound {bound}, which can take infinitely many values here; "
            f"Prodigy computes unif(0, n) for infinitely many values of n only if n is another variable.")
    names = distribution.get_variables() | {variable}
    marginal = distribution.marginal(variable, method=MarginalType.EXCLUDE)
    function = sympify_distribution(marginal)
    n, x = sympy.Symbol(bound), sympy.Symbol(variable)
    symbols = [sympy.Symbol(v) for v in names]
    integral = integral_from_zero(function, n, symbols)
    proposal = None
    if integral is not None:
        proposal = (integral - integral.subs(n, n * x)) / (1 - x)
        integrand = (function - x * function.subs(n, n * x)) / (1 - x)
        if decide_equal(sympy.diff(proposal, n), integrand, use_simplify=False) is not True \
                or decide_equal(proposal.subs(n, 0), sympy.Integer(0), use_simplify=False) is not True:
            proposal = None
    if proposal is None:
        text = str(function)
        text = text if len(text) <= 200 else text[:200] + " …"
        raise NotImplementedError(
            f"{construct} depends on the program variable {bound}, which can take infinitely many values here. "
            f"Prodigy computes unif(0, {bound}) with the operator 1/({bound} + 1) on the generating function, "
            f"which needs the integral from 0 of {text} in {bound}. No antiderivative was found that passes the "
            f"check (derivative, value 0 at {bound} = 0, analytic at 0): Prodigy integrates sums of terms "
            f"{bound}^k·exp(c·{bound}) by a formula and other integrands without parameters with SymPy; this one "
            f"is not implemented.")
    text = sympy_text(sympy.powsimp(sympy.together(proposal / n), combine="exp")).replace("**", "^")
    result = distribution.factory().from_expr(text, *sorted(names))
    parameters = distribution.get_parameters()
    return result.set_parameters(*parameters) if parameters else result


class RatioProbability(NamedTuple):
    """The probability ``x/n`` of two program variables x (``numerator``) and n (``denominator``), #237."""
    numerator: str
    denominator: str

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


def ratio_probability(probability: Expr, variables: Set[str]) -> Optional[RatioProbability]:
    """The probability as ``x/n`` with two different program variables x and n (``variables``), or None."""
    if not (isinstance(probability, BinopExpr) and probability.operator == Binop.DIVIDE):
        return None
    lhs, rhs = probability.lhs, probability.rhs
    if isinstance(lhs, VarExpr) and isinstance(rhs, VarExpr) and lhs.var != rhs.var \
            and {lhs.var, rhs.var} == variables:
        return RatioProbability(lhs.var, rhs.var)
    return None


def _proven_empty(distribution: Distribution, condition: Expr, construct: str, requirement: str) -> None:
    """Raises ``NotImplementedError`` (naming ``construct`` and the ``requirement``) unless no state of
    ``distribution`` satisfies ``condition``, as proven by the engine's filter and equality."""
    try:
        empty = distribution.filter(condition).is_provably_zero()
        reason = f"the part where {condition} is not provably zero"
    except NotImplementedError as error:
        empty, reason = False, f"filtering {condition} is not implemented here: {error}"
    if not empty:
        raise NotImplementedError(
            f"{construct} acts as an operator on the generating function only if {requirement} in every state "
            f"that reaches it; Prodigy could not prove this ({reason}). This is not implemented.")


def apply_ratio(probability: RatioProbability, distribution: Distribution,
                construct: str) -> tuple[Distribution, Distribution]:
    """
    The two parts of ``distribution`` G for the probability ``x/n`` (#237): the part weighted with ``x/n`` and the
    rest. Raises ``NotImplementedError`` (naming ``construct``) unless ``n > 0`` and ``x ≤ n`` are proven for
    every state of G and the integral below is computed and certified.

    Weighting the coefficient of ``x^a n^b`` with ``a/b`` is the Euler operator ``θ_x`` followed by ``1/θ_n``
    (module docstring): ``L = ∫_0^n (θ_x G)(t)/t dt``. It is defined when G has no terms ``n^0``; then
    ``(θ_x G)/n`` is a power series. ``x ≤ n`` makes ``x/n`` a probability. Both are proven by filtering G
    (``n = 0`` and ``x > n`` must give the zero measure, e.g. after the guard ``n > 0 and x < n``). The integral
    is proposed and certified by :func:`~prodigy.analysis.evtinvariants.exp_polynomial.integral_from_zero`
    (``∂L/∂n = (θ_x G)/n``, ``L(n = 0) = 0``, L analytic at the origin); as ``θ_n`` is injective on the power
    series without ``n^0`` terms, L is the weighted series. The other part is ``G - L``. The computation works on
    the SymPy form, as :func:`apply_operator` does, so it is the same on all engines.
    """
    x, n = probability.numerator, probability.denominator
    _proven_empty(distribution, BinopExpr(Binop.EQ, VarExpr(n), NatLitExpr(0)), construct, f"{n} > 0")
    _proven_empty(distribution, BinopExpr(Binop.GT, VarExpr(x), VarExpr(n)), construct, f"{x} <= {n}")
    function = sympify_distribution(distribution)
    symbols = [sympy.Symbol(v) for v in distribution.get_variables()]
    x_symbol, n_symbol = sympy.Symbol(x), sympy.Symbol(n)
    integrand = sympy.cancel(sympy.together(x_symbol * sympy.diff(function, x_symbol) / n_symbol))
    integral = integral_from_zero(integrand, n_symbol, symbols)
    if integral is None:
        text = str(integrand)
        text = text if len(text) <= 200 else text[:200] + " …"
        raise NotImplementedError(
            f"{construct} depends on the program variables {x} and {n}, which can take infinitely many values "
            f"here. Prodigy applies the probability {probability} as the operator ∫_0^{n} {x}·∂/∂{x} G(t)/t dt on "
            f"the generating function G, which needs the integral from 0 of {text} in {n}. No antiderivative was "
            f"found that passes the check (derivative, value 0 at {n} = 0, analytic at 0): Prodigy integrates sums "
            f"of terms {n}^k·exp(c·{n}), rational functions whose denominators split into linear factors in {n}, "
            f"and other integrands without parameters with SymPy; this one is not implemented.")
    return _back(integral, distribution), _back(function - integral, distribution)
