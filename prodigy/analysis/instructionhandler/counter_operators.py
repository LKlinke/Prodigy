"""
Probabilities that are rational in a counter, applied as operators on the generating function (#323).

A probabilistic choice ``{A} [p(k)] {B}`` whose probability depends on a program variable ``k`` with infinitely
many values cannot be computed by a case split over the values of k
(:mod:`~prodigy.analysis.instructionhandler.state_dependent`). For a probability that is *rational in k*, the
weighting has a closed form: weighting the coefficient ``g_n`` of ``k^n`` with ``p(n)`` is the operator ``p(θ)``
with the Euler operator ``θ = k·∂/∂k`` (``θ k^n = n k^n``). The forward semantics is
``⟦A⟧(p(θ) G) + ⟦B⟧((1 - p)(θ) G)``.

This module covers the probabilities ``p(k) = α + β/(k + a)`` with rational constants α, β and a natural
number ``a ≥ 1`` (e.g. ``1/(k + 1)``, ``k/(k + 1) = 1 - 1/(k + 1)``, ``1/2 + 1/(2k + 2)``). They are probabilities
for every value of k iff ``p(0)`` and ``α = lim p(n)`` are in [0, 1], because p is monotone in n. Then

    p(θ) G = α·G + β·H   with   H = k^(-a) ∫_0^k t^(a-1) G(t) dt,

since ``∫_0^k t^(n+a-1) dt = k^(n+a)/(n+a)``. For ``1/(k + 1)`` followed by ``k := k + 1`` this is just
``∫_0^k G``. The integral is computed for exponential polynomials in k only
(:func:`prodigy.analysis.evtinvariants.exp_polynomial.integral_from_zero`); its result is certified by
differentiation and evaluation at 0, so no symbolic integration has to be trusted. The computation works on the
SymPy form of the distribution and gives the result back to the distribution's engine, so it is the same on all
engines. Anything else (another shape of p, an integrand that is no exponential polynomial) is not implemented.
"""
from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Optional, Set

import sympy

from prodigy.pgcl import Binop, BinopExpr, Expr, NatLitExpr, RealLitExpr, VarExpr

from prodigy.analysis.evtinvariants.exp_polynomial import integral_from_zero
from prodigy.distribution import Distribution
from prodigy.distribution.equality import sympify_distribution


class CounterProbability(NamedTuple):
    """A probability ``α + β/(k + a)`` in the counter ``k`` (module docstring)."""
    variable: str
    alpha: Fraction
    beta: Fraction
    shift: int

    def __str__(self) -> str:
        return f"{self.alpha} + {self.beta}/({self.variable} + {self.shift})"


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
        shape = _shape(difference, sympy.Symbol(counter))
        if shape is None or min(_value_at_zero(*shape), shape[0]) < 0:
            return None
        return difference
    return None


def _shape(expr: sympy.Expr, k: sympy.Symbol) -> Optional[tuple[Fraction, Fraction, int]]:
    """``(α, β, a)`` if ``expr = α + β/(k + a)`` with rationals α, β and a natural number ``a ≥ 1``."""
    expr = sympy.cancel(sympy.together(expr))
    if not expr.has(k):
        return (Fraction(str(expr)), Fraction(0), 1) if expr.is_Rational else None
    numerator, denominator = sympy.fraction(expr)
    try:
        num, den = sympy.Poly(numerator, k), sympy.Poly(denominator, k)
    except sympy.PolynomialError:
        return None
    if den.degree() != 1 or num.degree() > 1 or not all(c.is_Rational for c in num.all_coeffs() + den.all_coeffs()):
        return None
    lead = den.all_coeffs()[0]
    shift = den.all_coeffs()[1] / lead
    if not (shift.is_Integer and shift >= 1):
        return None
    alpha, remainder = sympy.div(num, den)
    beta = remainder.as_expr() / lead
    return Fraction(str(alpha.as_expr())), Fraction(str(beta)), int(shift)


def _value_at_zero(alpha: Fraction, beta: Fraction, shift: int) -> Fraction:
    return alpha + beta / shift


def counter_probability(probability: Expr, variables: Set[str]) -> Optional[CounterProbability]:
    """The probability as ``α + β/(k + a)`` in its only program variable k, if it has this shape and is a
    probability for every value of k; otherwise None."""
    if len(variables) != 1:
        return None
    counter = next(iter(variables))
    expr = _to_sympy(probability, counter)
    if expr is None:
        return None
    shape = _shape(expr, sympy.Symbol(counter))
    if shape is None or shape[1] == 0:
        return None
    alpha, beta, shift = shape
    # p(n) is monotone in n, so it lies between p(0) and α
    if not (0 <= alpha <= 1 and 0 <= _value_at_zero(alpha, beta, shift) <= 1):
        return None
    return CounterProbability(counter, alpha, beta, shift)


def apply_operator(probability: CounterProbability, distribution: Distribution,
                   construct: str) -> tuple[Distribution, Distribution]:
    """The two parts ``p(θ) G`` and ``(1 - p)(θ) G`` of ``distribution`` G (module docstring). Raises
    ``NotImplementedError`` (naming ``construct``) if the integral cannot be computed and certified."""
    function = sympify_distribution(distribution)
    names = distribution.get_variables()
    k = sympy.Symbol(probability.variable)
    shift = probability.shift
    integral = integral_from_zero(function * k ** (shift - 1), k, [sympy.Symbol(v) for v in names])
    if integral is None:
        raise NotImplementedError(
            f"{construct} depends on the program variable {probability.variable}, which can take infinitely many "
            f"values here. Prodigy applies a probability of the form α + β/({probability.variable} + a) as an "
            f"operator on the generating function, which needs the integral of {function} in "
            f"{probability.variable}; this is implemented only for sums of terms "
            f"{probability.variable}^n·exp(c·{probability.variable}) (e.g. a Poisson distribution), not here.")
    alpha = sympy.Rational(probability.alpha.numerator, probability.alpha.denominator)
    beta = sympy.Rational(probability.beta.numerator, probability.beta.denominator)
    left = alpha * function + beta * integral / k ** shift
    right = function - left

    def back(expr: sympy.Expr) -> Distribution:
        text = str(sympy.together(sympy.expand(expr))).replace("**", "^")
        result = distribution.factory().from_expr(text, *sorted(names))
        parameters = distribution.get_parameters()
        return result.set_parameters(*parameters) if parameters else result

    return back(left), back(right)
