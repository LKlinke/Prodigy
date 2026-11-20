"""
Rational generating functions over a product denominator, with exact arithmetic.

A rational generating function in product form is ``P / D`` with a product denominator::

    D = (1 - r_1 X_1^{p_1}) * ... * (1 - r_k X_k^{p_k})       (r_i >= 0 rational, p_i >= 1; r_i = 0: no factor)

and a polynomial ``P`` with rational coefficients, stored as a dict mapping exponent tuples to
:class:`fractions.Fraction`. The coefficients of ``P / D`` are *eventually geometric*: along variable ``i`` they
satisfy ``c_{n + p_i e_i} = r_i c_n`` as soon as ``n_i`` exceeds the degree of ``P`` in ``X_i``. Hence
coefficient-wise non-negativity of ``P / D`` is decided exactly by finitely many coefficients: ``P / D >= 0`` iff all
coefficients in the box ``prod_i [0, deg_i P + p_i - 1]`` are non-negative (:meth:`ProductFormSeries.is_nonneg`).

These proof rules are shared by the check of input distributions (:mod:`prodigy.distribution.finite_measure`) and
the lower bounds of the analysis (:mod:`prodigy.analysis.lowerbounds.egcalc` adds the transformer semantics of
loop-free statements).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import sympy

from prodigy.distribution.equality import sympify_symbolic_string

Exp = Tuple[int, ...]
Num = Dict[Exp, Fraction]
Value = Union[Fraction, float]

ONE = Fraction(1)
ZERO = Fraction(0)

_Series = TypeVar("_Series", bound="ProductFormSeries")


# ------------------------------------------------------------------------------------------- denominators
class Denominator:
    """The fixed product denominator ``prod_i (1 - r_i X_i^{p_i})``."""

    def __init__(self, rates: Sequence, periods: Sequence[int]):
        self.r: List[Fraction] = [Fraction(r) for r in rates]
        self.p: List[int] = [int(p) for p in periods]
        if len(self.r) != len(self.p):
            raise ValueError("rates and periods must have the same length")
        if any(r < 0 for r in self.r) or any(p < 1 for p in self.p):
            raise ValueError("rates must be nonnegative and periods positive")
        self.d = len(self.r)

    def value_at_one(self) -> Fraction:
        val = ONE
        for r in self.r:
            val *= 1 - r
        return val

    def factor(self, i: int) -> Num:
        """The factor ``1 - r_i X_i^{p_i}`` as a numerator."""
        zero = tuple([0] * self.d)
        if self.r[i] == 0:
            return {zero: ONE}
        e = [0] * self.d
        e[i] = self.p[i]
        return {zero: ONE, tuple(e): -self.r[i]}

    def poly(self) -> Num:
        """``D`` itself as a numerator (i.e. the generating function ``1``)."""
        out = {tuple([0] * self.d): ONE}
        for i in range(self.d):
            out = pmul(out, self.factor(i))
        return out

    def without(self, i: int) -> "Denominator":
        rates = list(self.r)
        rates[i] = ZERO
        return Denominator(rates, self.p)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Denominator) and self.r == other.r and self.p == other.p

    def __repr__(self) -> str:
        return f"Denominator(rates={[str(r) for r in self.r]}, periods={self.p})"


# ------------------------------------------------------------------------------------------- polynomials
def padd(*polys: Num) -> Num:
    out: Num = {}
    for poly in polys:
        for e, c in poly.items():
            out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c != 0}


def pscale(poly: Num, a: Union[int, Fraction]) -> Num:
    if a == 0:
        return {}
    return {e: c * a for e, c in poly.items() if c != 0}


def pmul(p1: Num, p2: Num) -> Num:
    out: Num = {}
    for e1, c1 in p1.items():
        for e2, c2 in p2.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            out[e] = out.get(e, 0) + c1 * c2
    return {e: c for e, c in out.items() if c != 0}


def pshift(poly: Num, i: int, k: int) -> Num:
    """Multiply by ``X_i^k``; ``k`` may be negative if all exponents stay nonnegative."""
    out: Num = {}
    for e, c in poly.items():
        e2 = list(e)
        e2[i] += k
        if e2[i] < 0:
            raise ValueError("negative exponent in shift")
        out[tuple(e2)] = c
    return out


def degrees(poly: Num, d: int) -> List[int]:
    deg = [0] * d
    for e in poly:
        for i in range(d):
            deg[i] = max(deg[i], e[i])
    return deg


def monomial(d: int, e: Iterable[int], c: Union[int, Fraction] = ONE) -> Num:  # pylint: disable=unused-argument
    return {tuple(e): Fraction(c)} if c != 0 else {}


# ------------------------------------------------------------------------------------------- series
class ProductFormSeries:
    """The series ``P / D`` over a fixed product denominator ``D``: coefficients, masses and moments (exact)."""

    def __init__(self, variables: Sequence[str], den: Denominator):
        self.vars = list(variables)
        self.idx = {v: i for i, v in enumerate(self.vars)}
        self.den = den
        self.d = len(self.vars)
        if den.d != self.d:
            raise ValueError("denominator does not match the number of variables")

    def with_den(self: _Series, den: Denominator) -> _Series:
        return type(self)(self.vars, den)

    # ---------------------------------------------------------------- coefficient slices along one variable
    def coeffs_along(self, poly: Num, i: int, upto: int) -> List[Num]:
        """The ``X_i``-coefficients ``c_0..c_upto`` of ``P / D``, each as a numerator over ``D`` without the
        ``X_i`` factor (exponent ``i`` set to 0)."""
        r, p = self.den.r[i], self.den.p[i]
        slices: Dict[int, Num] = {}
        for e, c in poly.items():
            e2 = list(e)
            e2[i] = 0
            sl = slices.setdefault(e[i], {})
            sl[tuple(e2)] = sl.get(tuple(e2), 0) + c
        out: List[Num] = []
        for v in range(upto + 1):
            cur = dict(slices.get(v, {}))
            if r != 0 and v - p >= 0:
                cur = padd(cur, pscale(out[v - p], r))
            out.append(cur)
        return out

    def _over_full(self, poly_minus: Num, i: int) -> Num:
        """Numerator over ``D`` without the ``X_i`` factor -> numerator over ``D``."""
        return pmul(poly_minus, self.den.factor(i)) if self.den.r[i] != 0 else poly_minus

    def eliminate(self, poly: Num, i: int) -> Optional[Num]:
        """``F(X_i = 1)`` as a numerator over ``D`` without the ``X_i`` factor (exponent ``i`` set to 0), or None if
        the series diverges at ``X_i = 1`` (only possible if ``r_i >= 1``)."""
        r, p = self.den.r[i], self.den.p[i]
        if r != 0 and r >= 1:
            deg = degrees(poly, self.d)[i]
            sl = self.coeffs_along(poly, i, deg + p)
            if any(sl[v] for v in range(deg + 1, deg + p + 1)):
                return None
            return padd(*sl[: deg + 1]) if sl else {}
        out: Num = {}
        for e, c in poly.items():
            e2 = list(e)
            e2[i] = 0
            out[tuple(e2)] = out.get(tuple(e2), 0) + c
        if r != 0:
            out = pscale(out, ONE / (1 - r))
        return {e: c for e, c in out.items() if c != 0}

    # ---------------------------------------------------------------- coefficients
    def expand(self, poly: Num, box: Sequence[int]) -> np.ndarray:
        """The coefficients of ``P / D`` on the box ``prod_i [0, box_i]`` (object array of Fractions)."""
        shape = [b + 1 for b in box]
        arr = np.empty(shape, dtype=object)
        arr.fill(ZERO)
        for e, c in poly.items():
            if all(e[j] <= box[j] for j in range(self.d)):
                arr[e] += c
        for i in range(self.d):
            r, p = self.den.r[i], self.den.p[i]
            if r == 0:
                continue
            for v in range(p, shape[i]):
                sl_v: List[Union[slice, int]] = [slice(None)] * self.d
                sl_v[i] = v
                sl_w: List[Union[slice, int]] = [slice(None)] * self.d
                sl_w[i] = v - p
                arr[tuple(sl_v)] = arr[tuple(sl_v)] + r * arr[tuple(sl_w)]
        return arr

    def positivity_box(self, poly: Num) -> List[int]:
        deg = degrees(poly, self.d)
        return [deg[i] + self.den.p[i] - 1 if self.den.r[i] != 0 else deg[i] for i in range(self.d)]

    def is_nonneg(self, poly: Num) -> bool:
        """Exact decision of coefficient-wise nonnegativity of ``P / D``."""
        if not poly:
            return True
        return all(x >= 0 for x in self.expand(poly, self.positivity_box(poly)).flat)

    def first_negative(self, poly: Num) -> Optional[Tuple[Exp, Fraction]]:
        """A coefficient of ``P / D`` that is negative (state and value), or None."""
        if not poly:
            return None
        arr = self.expand(poly, self.positivity_box(poly))
        for idx in np.ndindex(*arr.shape):
            if arr[idx] < 0:
                return tuple(int(v) for v in idx), cast(Fraction, arr[idx])
        return None

    def coefficient(self, poly: Num, state: Sequence[int]) -> Fraction:
        return self.expand(poly, list(state))[tuple(state)]

    # ---------------------------------------------------------------- functionals
    def mass(self, poly: Num) -> Union[Fraction, float]:
        """Total mass ``(P / D)(1, ..., 1)`` in ``[0, inf]`` (exact); ``math.inf`` if the series diverges."""
        cur, calc = dict(poly), self
        for i in range(self.d):
            if not cur:
                return ZERO
            elim = calc.eliminate(cur, i)
            if elim is None:
                return math.inf
            cur = elim
            calc = calc.with_den(calc.den.without(i))
        return sum(cur.values(), ZERO) if cur else ZERO

    def moment(self, poly: Num, i: int) -> Union[Fraction, float]:
        """First moment ``sum_n n_i * coeff_n(P / D)`` of variable ``i`` (exact), ``math.inf`` if it diverges.
        Requires ``P / D >= 0`` for the value ``inf`` to be meaningful."""
        cur, calc = dict(poly), self
        for j in range(self.d):
            if j == i or not cur:
                continue
            elim = calc.eliminate(cur, j)
            if elim is None:
                return math.inf
            cur = elim
            calc = calc.with_den(calc.den.without(j))
        if not cur:
            return ZERO
        r, p = calc.den.r[i], calc.den.p[i]
        if r != 0 and r >= 1:
            deg = degrees(cur, self.d)[i]
            sl = calc.coeffs_along(cur, i, deg + p)
            if any(sl[v] for v in range(deg + 1, deg + p + 1)):
                return math.inf
            return sum((v * sum(sl[v].values(), ZERO) for v in range(deg + 1)), ZERO)
        num_val = sum(cur.values(), ZERO)
        num_der = sum((e[i] * c for e, c in cur.items()), ZERO)
        if r == 0:
            return num_der
        # X F'(X) at 1 for F = N / (1 - r X^p):  N'(1)/(1-r) + N(1) r p / (1-r)^2
        return num_der / (1 - r) + num_val * r * p / (1 - r) ** 2

    def defect_mass(self, poly: Num, sign: int = 1) -> Union[Fraction, float]:
        """``sum_n max(sign * coeff_n(P/D), 0)`` over all states (exact; ``inf`` if a growing tail is positive)."""
        if not poly:
            return ZERO
        deg = degrees(poly, self.d)
        box = [deg[i] + (self.den.p[i] if self.den.r[i] != 0 else 0) for i in range(self.d)]
        arr = self.expand(poly, box)
        total = ZERO
        for idx in np.ndindex(*arr.shape):
            val = sign * arr[idx]
            if val <= 0:
                continue
            weight = ONE
            for i in range(self.d):
                r = self.den.r[i]
                if r != 0 and idx[i] > deg[i]:
                    if r >= 1:
                        return math.inf
                    weight /= 1 - r
            total += val * weight
        return total

    # ---------------------------------------------------------------- printing
    def to_expr(self, poly: Num) -> str:
        """``P / D`` as an expression string in the program variables (parsable by Prodigy's factories)."""
        def mono(e: Sequence[int], c: Fraction) -> str:
            parts = []
            for v, k in zip(self.vars, e):
                if k == 1:
                    parts.append(v)
                elif k > 1:
                    parts.append(f"{v}^{k}")
            coeff = f"({c})" if c.denominator != 1 or c < 0 else f"{c}"
            return "*".join([coeff] + parts) if parts else coeff
        num = " + ".join(mono(e, c) for e, c in sorted(poly.items())) if poly else "0"
        dens = [f"(1 - ({r})*{v}^{p})" if p > 1 else f"(1 - ({r})*{v})"
                for v, r, p in zip(self.vars, self.den.r, self.den.p) if r != 0]
        if not dens:
            return f"({num})"
        return f"({num}) / ({'*'.join(dens)})"


# ------------------------------------------------------------------------------------------- closed forms
def to_product_form(expr: Union[str, sympy.Expr], variables: Sequence[str]) -> Optional[Tuple[Num, Denominator]]:
    """Write a rational generating function as ``P / prod_i (1 - r_i X_i^{p_i})`` (at most one factor per variable,
    ``r_i >= 0``), or return None if this is not possible."""
    syms = [sympy.Symbol(v) for v in variables]
    e = sympify_symbolic_string(expr) if isinstance(expr, str) else expr
    e = sympy.together(sympy.expand(e))
    num, den = sympy.fraction(e)
    if any(s not in syms for s in e.free_symbols):
        return None
    const, factors = sympy.factor_list(sympy.expand(den), *syms)
    const = sympy.Rational(const)
    rates = [Fraction(0)] * len(syms)
    periods = [1] * len(syms)
    scale = const
    for fac, mult in factors:
        if mult != 1:
            return None
        free = fac.free_symbols
        if len(free) != 1:
            return None
        x = free.pop()
        i = syms.index(x)
        poly = sympy.Poly(fac, x)
        terms = poly.terms()
        if len(terms) != 2:
            return None
        (deg_a, a), (deg_b, b) = sorted(((t[0][0], t[1]) for t in terms))
        if deg_a != 0 or a == 0:
            return None
        r = -sympy.Rational(b) / sympy.Rational(a)
        if r < 0 or rates[i] != 0:
            return None
        rates[i], periods[i] = Fraction(int(r.p), int(r.q)), int(deg_b)
        scale *= a
    poly_num = sympy.Poly(sympy.expand(num / scale), *syms)
    numer: Num = {}
    for monom, c in poly_num.terms():
        c = sympy.Rational(c)
        numer[tuple(int(m) for m in monom)] = Fraction(int(c.p), int(c.q))
    return numer, Denominator(rates, periods)


def rational_mass(expr_str: str, variables: Sequence[str]) -> Optional[Union[Fraction, float]]:
    """Total mass of a rational generating function with NONNEGATIVE coefficients, or ``None`` if the closed form is
    not rational.  Let ``f(t) = F(t, ..., t)`` in lowest terms.  By Pringsheim's theorem, the radius of convergence of
    ``f`` is its smallest pole on the positive real axis; hence the mass ``sum of coefficients = lim_{t -> 1-} f(t)`` is
    infinite iff ``f`` has a pole in ``(0, 1]``, and equals ``f(1)`` otherwise.  Evaluating the closed form at ``1``
    directly is unsound: ``(1 + X)/(1 - 2X) = 1 + 3X + 6X^2 + ...`` evaluates to ``-2``."""
    t = sympy.Symbol("_t", positive=True)
    expr = sympify_symbolic_string(expr_str).subs({sympy.Symbol(v): t for v in variables}, simultaneous=True)
    f = sympy.cancel(sympy.together(expr))
    numer, den = sympy.fraction(f)
    if not (numer.is_polynomial(t) and den.is_polynomial(t)):
        return None
    den_poly = sympy.Poly(den, t)
    if den_poly.eval(0) == 0:
        return None  # not a power series at the origin
    if den_poly.degree() > 0 and den_poly.count_roots(0, 1) > 0:  # exact (Sturm sequences); no root at 0
        return math.inf
    value = f.subs(t, 1)
    return Fraction(str(sympy.nsimplify(value))) if value.is_rational else float(value)
