"""
When is the marginal of a generating function the sum of its series (Prodigy#393)?

Summing out variables x = (x_1, ..., x_k) of a generating function F(x, y) evaluates the closed form at
x = 1. This is the marginal only if, for every monomial y^j of the variables y that are kept, the series
sum_i a_(i, j) of the coefficients converges, and if the closed form F(1, y) is the generating function of
these sums. Evaluating the closed form alone does not show this: for ``1/((1 - 2x)(1 - y))`` the series in
x diverges at x = 1 (radius 1/2), but the closed form gives the finite value ``-1/(1 - y)``.

**Criterion.** Let K be the closed unit polydisc in x times the point y = 0. If F is analytic on a
neighbourhood of K, then (K is compact) F is analytic on a polydisc ``|x_i| < 1 + e, |y_l| < d`` and equals its
power series there, which converges absolutely on it. So every sum over x of the coefficients of a monomial
y^j converges absolutely, and F(1, y) is the generating function of these sums for small y; the closed form
gives the marginal as a formal power series in y. This holds for signed coefficients as well.

**Deciding analyticity on K** (sufficient rules; everything else is "not proven", never the evaluated value):

- a rational function P/Q in lowest terms is analytic on a neighbourhood of K if Q(x, 0) has no zero on the
  closed polydisc. Q(x, 0) is factored over the rationals, and each factor is checked exactly:

  - a factor in one variable by the Schur–Cohn test (all roots of modulus greater than 1, in exact arithmetic;
    a root of modulus 1, where the series need not converge, is never accepted),
  - a factor in several variables if its constant term dominates: ``|c_0| > sum of |c_m|`` over the other
    monomials. For ``c - R`` with R having non-negative coefficients (the denominators of distributions) this is
    exact: if ``R(1, ..., 1) >= c``, then ``c - R`` has a zero on ``[0, 1]^k`` by the intermediate value theorem.

  Coefficients with parameters are not decided.
- sums, products and non-negative integer powers of analytic functions, and ``exp`` of one;
- ``log(g)`` and ``g^r`` (r not an integer, principal branch) of an analytic g whose real part is positive on K;
  this is proven for a polynomial g(x, 0) with a positive real constant term c that dominates its other
  coefficients (e.g. the logarithmic distribution ``log(1 - p x)``, p < 1).

Inputs of the analysis are validated to have finite mass and non-negative coefficients (#300, #314), and the
program semantics keeps this, so their marginals pass the rules above whenever the closed form is of these
kinds; candidate measures (invariants, the second-order generating functions of the equivalence check, signed
differences inside checks) are where the check matters.
"""
from __future__ import annotations

import functools
from typing import Collection, Dict, FrozenSet, List, Optional, Sequence, Tuple

import sympy

# the reason why a marginal is not proven to be the sum of the series, or None if it is proven
Obstacle = Optional[str]

_NON_FINITE = (sympy.nan, sympy.zoo, sympy.oo, -sympy.oo)


def _text(expression: sympy.Expr) -> str:
    return str(expression).replace("**", "^")


def marginal_obstacle(function: sympy.Expr, summed: Collection[str], kept: Collection[str]) -> Obstacle:
    """
    None if the power series of `function` provably converges absolutely when the variables `summed` are set
    to 1, coefficient by coefficient in the variables `kept` (see the module documentation); then the closed
    form at 1 is the marginal. Otherwise the reason why this is not proven. Other symbols are parameters.
    """
    return _cached_obstacle(sympy.sympify(function), frozenset(summed), frozenset(kept))


@functools.lru_cache(maxsize=1024)
def _cached_obstacle(function: sympy.Expr, summed: FrozenSet[str], kept: FrozenSet[str]) -> Obstacle:
    by_name = {symbol.name: symbol for symbol in function.free_symbols if isinstance(symbol, sympy.Symbol)}
    xs = tuple(sorted((by_name[name] for name in summed if name in by_name), key=lambda s: s.name))
    if not xs:
        return None
    at_zero = {by_name[name]: sympy.Integer(0) for name in kept if name in by_name}
    variables = xs + tuple(at_zero)
    try:
        return _Analyticity(xs, variables, at_zero).obstacle(function)
    except (sympy.PolynomialError, sympy.polys.polyerrors.PolificationFailed, TypeError, ValueError,
            NotImplementedError, RecursionError) as exc:
        return f"SymPy could not decide it ({type(exc).__name__})"


class _Analyticity:
    """Analyticity on a neighbourhood of K = {|x_i| <= 1} x {y = 0} (module documentation)."""

    def __init__(self, xs: Tuple[sympy.Symbol, ...], variables: Tuple[sympy.Symbol, ...],
                 at_zero: Dict[sympy.Symbol, sympy.Integer]):
        self.xs = xs
        self.variables = variables
        self.at_zero = at_zero

    def obstacle(self, expression: sympy.Expr) -> Obstacle:
        # pylint: disable=too-many-return-statements  # one obstacle per case of the walk over the expression
        if not expression.has(*self.variables):
            if expression.has(*_NON_FINITE) or expression.is_finite is False:
                return f"it contains the non-finite value {_text(expression)}"
            return None
        if expression.is_polynomial(*self.variables):
            return None
        if expression.is_rational_function(*self.variables):
            return self._rational(expression)
        if isinstance(expression, (sympy.Add, sympy.Mul)):
            # the rational part as one fraction (its factors may cancel), the other parts one by one
            rational = [a for a in expression.args if a.is_rational_function(*self.variables)]
            others = [a for a in expression.args if not a.is_rational_function(*self.variables)]
            parts = ([expression.func(*rational)] if rational else []) + others
            for part in parts:
                reason = self.obstacle(part)
                if reason is not None:
                    return reason
            return None
        if isinstance(expression, sympy.Pow):
            return self._power(expression)
        if isinstance(expression, sympy.exp):
            return self.obstacle(expression.args[0])
        if isinstance(expression, sympy.log):
            return self.obstacle(expression.args[0]) or self._positive_real_part(expression.args[0], expression)
        return (f"{_text(expression)} is not a closed form whose singularities Prodigy knows (rational functions, "
                "exp, log and powers)")

    def _power(self, expression: sympy.Pow) -> Obstacle:
        base, exponent = expression.args
        if exponent.has(*self.variables):
            if base.has(*self.variables) or not (base.is_positive or isinstance(base, sympy.exp)):
                return f"{_text(expression)} has a variable in the exponent"
            # c^g = exp(g log c) with c > 0
            return self.obstacle(exponent)
        if exponent.is_Integer:
            if exponent >= 0 or isinstance(base, sympy.exp):
                # exp(g)^(-n) = exp(-n g)
                return self.obstacle(base)
            return self.obstacle(base) or self._nonzero(base, expression)
        if exponent.is_real:
            return self.obstacle(base) or self._positive_real_part(base, expression)
        return f"{_text(expression)} has an exponent that is not a real number"

    def _rational(self, expression: sympy.Expr) -> Obstacle:
        _, denominator = sympy.fraction(sympy.cancel(sympy.together(expression)))
        on_k = sympy.expand(denominator.subs(self.at_zero))
        parameters = on_k.free_symbols - set(self.xs)
        if parameters:
            return (f"its denominator {_text(denominator)} depends on the parameters "
                    f"{', '.join(sorted(str(p) for p in parameters))}, so its zeros are not located")
        if on_k.is_zero:
            return f"its denominator {_text(denominator)} is 0 at 0 (no power series)"
        if not on_k.has(*self.xs):
            return None
        _, factors = sympy.factor_list(on_k, *[x for x in self.xs if on_k.has(x)])
        for factor, _ in factors:
            reason = _factor_obstacle(sympy.Poly(factor, *[x for x in self.xs if factor.has(x)]))
            if reason is not None:
                return f"its denominator {_text(sympy.factor(denominator))} has the factor {_text(factor)}, {reason}"
        return None

    def _nonzero(self, divisor: sympy.Expr, expression: sympy.Expr) -> Obstacle:
        """None if `divisor`, analytic on a neighbourhood of K, has no zero on K (then so is `expression`, a
        negative power of it): proven if `divisor` at y = 0 is a rational function in x whose numerator has no
        zero on the closed polydisc."""
        on_k = sympy.cancel(sympy.together(divisor.subs(self.at_zero)))
        reason = (f"{_text(expression)} divides by {_text(divisor)}, which is not proven to be non-zero for "
                  f"|{', '.join(map(str, self.xs))}| <= 1")
        if on_k.free_symbols - set(self.xs) or not on_k.is_rational_function(*self.xs):
            return reason
        numerator, _ = sympy.fraction(on_k)
        if not numerator.has(*self.xs):
            return None if numerator.is_nonzero else reason
        _, factors = sympy.factor_list(numerator, *[x for x in self.xs if numerator.has(x)])
        for factor, _ in factors:
            if _factor_obstacle(sympy.Poly(factor, *[x for x in self.xs if factor.has(x)])) is not None:
                return reason
        return None

    def _positive_real_part(self, argument: sympy.Expr, expression: sympy.Expr) -> Obstacle:
        """None if the real part of `argument` is provably positive on K (the principal branch of `expression`
        is then analytic there)."""
        on_k = sympy.expand(argument.subs(self.at_zero))
        reason = (f"{_text(expression)} is not proven to stay off its branch cut: the real part of "
                  f"{_text(argument)} is not proven to be positive for |{', '.join(map(str, self.xs))}| <= 1")
        if not on_k.is_polynomial(*self.xs) or on_k.free_symbols - set(self.xs):
            return reason
        poly = sympy.Poly(on_k, *self.xs)
        constant = poly.coeff_monomial(1)
        if not (constant.is_real and constant.is_positive):
            return reason
        rest = sum((abs(c) for m, c in poly.terms() if any(m)), sympy.Integer(0))
        return None if (constant - rest).is_positive else reason


def _factor_obstacle(poly: sympy.Poly) -> Obstacle:
    """None if the polynomial `poly` (numeric coefficients) has no zero on the closed unit polydisc."""
    terms = poly.terms()
    constant = poly.coeff_monomial(1)
    rest = sum((abs(c) for m, c in terms if any(m)), sympy.Integer(0))
    if (abs(constant) - rest).is_positive:
        return None
    if len(poly.gens) == 1:
        if _roots_outside_unit_disc(poly.all_coeffs()):
            return None
        return f"which has a zero with |{poly.gens[0]}| <= 1"
    return (f"which is not proven to be non-zero for |{', '.join(map(str, poly.gens))}| <= 1 (its constant term "
            "does not dominate its other coefficients)")


def _roots_outside_unit_disc(coefficients: Sequence[sympy.Expr]) -> bool:
    """
    Whether all roots of the polynomial p with `coefficients` (exact numbers, highest degree first) have modulus
    greater than 1. Equivalently, the reversed polynomial f(z) = z^n p(1/z) has all its roots in the open unit
    disc, which the Schur–Cohn recursion decides exactly: f = a_n z^n + ... + a_0 with a_n != 0 has all roots in
    |z| < 1 iff |a_n| > |a_0| and (conj(a_n) f - a_0 f*) / z has, where f* = z^n conj(f(1/conj(z))). A root of
    modulus 1 fails the strict comparison. Returns False when a comparison is not decided.
    """
    # f, highest degree first: the coefficients of p, lowest degree first
    f: List[sympy.Expr] = [sympy.nsimplify(c) if c.is_Float else sympy.sympify(c) for c in reversed(coefficients)]
    if f[0] == 0:
        return False  # p(0) = 0
    while len(f) > 1:
        lead, constant = f[0], f[-1]
        if not (sympy.Abs(lead) - sympy.Abs(constant)).is_positive:
            return False
        # f* has the coefficients of f reversed and conjugated
        star = [sympy.conjugate(c) for c in reversed(f)]
        combined = [sympy.expand(sympy.conjugate(lead) * a - constant * b) for a, b in zip(f, star, strict=True)]
        f = combined[:-1]  # the constant coefficient is 0: divide by z
    return True
