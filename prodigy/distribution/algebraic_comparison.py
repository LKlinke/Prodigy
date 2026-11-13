"""
Exact filters `x < y + c` and `x = y + c` on rational generating functions where x and y are not
independent, by the residue (kernel) method. The result is algebraic in general, not rational
(Prodigy#243): e.g. the part with x < y of (1/2) / (1 - x/4 - y/4) contains sqrt(1 - x*y/4).

Let F(x, y, r) be rational with F(0, 0, r) defined (r: the other variables). With i the exponent of
x and j that of y, H(s) = F(x/s, y*s) = sum_k h_k s^k, h_k = sum_{j - i = k} f_ij x^i y^j, is a
Laurent series in s converging on an annulus that contains |s| = 1 for small x and y. So

    sum_{j - i >= c} f_ij x^i y^j = sum_{k >= c} h_k = (1/2 pi i) oint_{|s| = rho} H(s) s^(-c) / (s - 1) ds,
    sum_{j - i = c}  f_ij x^i y^j = h_c        = (1/2 pi i) oint_{|s| = rho} H(s) s^(-c-1) ds

with rho slightly larger than 1, i.e. the sum of the residues inside |s| < rho: at s = 0, at s = 1
and at the "small" roots of the kernel K, the denominator of H without the factors s and s - 1. At
x = y = 0 the kernel must be c0 * s^n (c0 a nonzero rational number); then, for small x and y,
exactly n roots of K are small (they tend to 0) and the others are large (they tend to infinity).

- no small root: the residues at 0 and 1 are rational, and so is the result;
- no large root: the result is minus the (rational) residue at infinity;
- K quadratic, k2 s^2 + k1 s + k0, with one small root: k0 and k2 vanish at x = y = 0 and k1 is c0
  there, and the small root is rho0 = -2 k0 / (k1 + W) with W = c0 * sqrt(R), R = (k1^2 - 4 k0 k2) / c0^2
  and the principal square root. Its residue is N(rho0) / (rho0^a (rho0 - 1)^b W) for the integrand
  N / (s^a (s - 1)^b K), as K'(rho0) = W.

The branch of the square root is proven, not guessed: R is a polynomial with R = 1 at x = y = 0 and
rational coefficients whose non-constant ones have absolute values summing to less than 1. So
|R - 1| < 1 on the closed unit polydisk of all variables, the principal square root is analytic
there and W = c0 at x = y = 0, which makes rho0 the small root. The result equals the power series
near 0 and therefore on the whole open unit polydisk, where the filtered series converges as well
(it is a part of the series of F), so substituting values of modulus at most 1 (marginals,
probabilities, expected values) gives the values of the series. Everything else (parameters,
kernels of higher degree, other square roots) raises :class:`ComparisonNotImplemented`.
"""
from __future__ import annotations

from typing import Iterable, Set, Tuple

import sympy

from prodigy.distribution.independent_comparison import ComparisonNotImplemented
from prodigy.distribution.rational_series import rational_taylor_coefficients

PARAMETERS = "the compared variables are not independent, and the generating function has parameters"
NOT_RATIONAL = "the compared variables are not independent, and the generating function is not rational"


def _multiplicity(poly: sympy.Poly, factor: sympy.Poly) -> Tuple[int, sympy.Poly]:
    """How often `factor` divides `poly`, and the quotient."""
    count = 0
    while True:
        quotient, remainder = poly.div(factor)
        if not remainder.is_zero:
            return count, poly
        poly, count = quotient, count + 1


def _residue(numerator: sympy.Expr, denominator: sympy.Expr, s: sympy.Symbol, point, order: int) -> sympy.Expr:
    """The residue at `point` of numerator / (denominator * (s - point)^order); denominator(point) != 0."""
    if order == 0:
        return sympy.Integer(0)
    u = sympy.Dummy("u")
    shifted = sympy.cancel((numerator / denominator).subs(s, u + point))
    coefficients = rational_taylor_coefficients(shifted, u, order)
    if coefficients is None:  # pragma: no cover - the denominator has no root at the point
        raise AssertionError("pole in the residue")
    return coefficients[order - 1] if len(coefficients) >= order else sympy.Integer(0)


def _residue_at_infinity(integrand: sympy.Expr, s: sympy.Symbol) -> sympy.Expr:
    """Res_{s = oo} f = -Res_{t = 0} f(1/t) / t^2."""
    t = sympy.Dummy("t")
    numerator, denominator = sympy.fraction(sympy.cancel(integrand.subs(s, 1 / t) / t ** 2))
    order, rest = _multiplicity(sympy.Poly(denominator, t), sympy.Poly(t, t))
    return -_residue(numerator, rest.as_expr(), t, 0, order)


def _sum_below_one(poly: sympy.Poly) -> bool:
    """Whether the non-constant coefficients have absolute values summing to less than 1."""
    total = sum(abs(coefficient) for monomial, coefficient in poly.terms() if any(monomial))
    return bool(total < 1)


def filter_difference_algebraic(function: sympy.Expr, x: str, y: str, relation: str, shift: int,
                                variables: Iterable[str], condition: str) -> sympy.Expr:
    """
    The part of the rational `function` whose exponents i of the variable x and j of y satisfy
    `j - i >= shift` (relation `"ge"`) or `j - i = shift` (`"eq"`), possibly algebraic (see the
    module documentation). `variables` are the variables of the distribution; other symbols are
    parameters, which are not supported.

    :raises ComparisonNotImplemented: if the result cannot be computed with a proven branch.
    """
    # pylint: disable=too-many-locals
    function = sympy.sympify(function)
    names: Set[str] = set(variables)
    if any(str(symbol) not in names for symbol in function.free_symbols):
        raise ComparisonNotImplemented(condition, PARAMETERS)
    plain = {symbol: sympy.Symbol(str(symbol)) for symbol in function.free_symbols}
    function = function.xreplace(plain)
    if not function.is_rational_function(*function.free_symbols):
        raise ComparisonNotImplemented(condition, NOT_RATIONAL)
    x_symbol, y_symbol = sympy.Symbol(x), sympy.Symbol(y)
    s = sympy.Dummy("s")
    kernel_function = function.subs({x_symbol: x_symbol / s, y_symbol: y_symbol * s}, simultaneous=True)
    integrand = kernel_function * s ** (-shift) / (s - 1) if relation == "ge" \
        else kernel_function * s ** (-shift - 1)
    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(integrand)))
    symbols = [sympy.Symbol(name) for name in sorted(names)]

    def _back(result: sympy.Expr) -> sympy.Expr:
        return result.xreplace({plain_symbol: original for original, plain_symbol in plain.items()})

    def _not_implemented(reason: str) -> ComparisonNotImplemented:
        return ComparisonNotImplemented(condition, "the compared variables are not independent, and " + reason)

    origin = {x_symbol: 0, y_symbol: 0}
    poles = []  # (factor, multiplicity, inside: None for a quadratic factor with one small root)
    large = 0
    for factor, multiplicity in sympy.factor_list(denominator)[1]:
        poly = sympy.Poly(factor, s)
        degree = poly.degree()
        if degree <= 0:
            continue
        coefficients = [sympy.cancel(c.subs(origin)) for c in poly.all_coeffs()]  # highest first
        if any(not c.is_Rational for c in coefficients):
            raise _not_implemented("the roots of the kernel cannot be told apart at 0")
        if degree == 1:
            if coefficients[0] == 0:
                large += 1  # the root tends to infinity
                continue
            limit = -coefficients[1] / coefficients[0]
            root = sympy.cancel(-poly.all_coeffs()[1] / poly.all_coeffs()[0])
            if abs(limit) == 1 and root != limit:
                raise _not_implemented("a root of the kernel tends to the unit circle")
            if abs(limit) > 1:
                large += 1
                continue
            poles.append((poly, multiplicity, root))
            continue
        nonzero = [k for k, c in enumerate(reversed(coefficients)) if c != 0]
        if len(nonzero) != 1:
            raise _not_implemented("the roots of the kernel cannot be told apart at 0")
        small = nonzero[0]
        large += degree - small
        if small == 0:
            continue
        if degree != 2 or small != 1 or multiplicity != 1:
            poles.append((poly, multiplicity, None))
            continue
        poles.append((poly, multiplicity, "quadratic"))

    if large == 0:
        return _back(sympy.cancel(-_residue_at_infinity(integrand, s)))
    if any(inside is None for _, _, inside in poles):
        raise _not_implemented("the filter needs the roots of a polynomial of degree more than 2")

    rational = sympy.Integer(0)
    algebraic = sympy.Integer(0)
    for poly, multiplicity, inside in poles:
        others = sympy.cancel(denominator / poly.as_expr() ** multiplicity)
        if inside != "quadratic":
            leading = poly.all_coeffs()[0]
            rational += _residue(numerator, others * leading ** multiplicity, s, inside, multiplicity)
            continue
        k2, k1, k0 = poly.all_coeffs()
        constant = k1.subs(origin)
        radicand = sympy.Poly(sympy.expand((k1 ** 2 - 4 * k0 * k2) / constant ** 2), *symbols)
        if radicand.coeff_monomial(1) != 1 or not _sum_below_one(radicand):
            raise _not_implemented("the branch of the square root in the result is not proven")
        root_term = constant * sympy.sqrt(radicand.as_expr())
        small_root = -2 * k0 / (k1 + root_term)
        algebraic += (numerator / others).subs(s, small_root) / root_term
    return _back(sympy.cancel(rational) + sympy.together(algebraic))
