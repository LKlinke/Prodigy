# pylint: disable=invalid-name
# Names follow the mathematics (H, Phi, A, b, ...); scipy and FLINT are imported where they are needed.
r"""
Exact generating-function calculus over a fixed product denominator.

A (generalised) generating function is represented as ``P / D`` where ``D`` is a fixed product denominator::

    D = (1 - r_1 X_1^{p_1}) * ... * (1 - r_k X_k^{p_k})       (r_i >= 0 rational, p_i >= 1; r_i = 0: no factor)

and ``P`` is a polynomial with rational coefficients, stored as a dict mapping exponent tuples to
:class:`fractions.Fraction`.  The coefficients of ``P / D`` are *eventually geometric*: along variable ``i`` they
satisfy ``c_{n + p_i e_i} = r_i c_n`` as soon as ``n_i`` exceeds the degree of ``P`` in ``X_i``.

All loop-free pGCL statements of the supported fragment map numerators over ``D`` to numerators over the same
``D``: guards and observations comparing a single variable with a constant (or testing it modulo ``m`` with
``m | p_i``), increments and decrements (monus) by constants, constant assignments, sampling from ``unif``,
``bernoulli`` and ``binomial`` (and from ``geometric(q)`` if ``r_i = 1 - q`` and ``p_i = 1``), probabilistic choice,
conditionals, ``observe`` (which removes the violating mass) and ``abort`` (divergence).

The representation, the polynomial arithmetic and the series operations (coefficients, the exact finite positivity
check, masses and moments) are in :mod:`prodigy.distribution.product_form`; this module adds the statements.
"""
from __future__ import annotations

import math
from fractions import Fraction

from typing import Callable, Dict, Optional, Tuple, Union

from prodigy.analysis.lowerbounds.ir import IR, UnsupportedProgram, eval_number
from prodigy.distribution.product_form import (ONE, ZERO, Denominator, Exp, Num, ProductFormSeries, Value, degrees,
                                               monomial, padd, pmul, pscale, pshift)

__all__ = ["Denominator", "Exp", "GFCalculus", "Num", "Value", "degrees", "directed_decimal", "monomial", "padd",
           "pmul", "pscale", "pshift"]

# ------------------------------------------------------------------------------------------- linear guards
def _affine(e: IR) -> Optional[Tuple[Fraction, Optional[str], Fraction]]:
    """
    Write ``e`` as ``a * var + b`` with constants ``a, b`` (``var`` may be None), or return None.

    Subtraction is truncated at 0 (monus). It is the rational difference only if the difference is never negative,
    i.e. ``a >= 0`` and ``b >= 0`` for the difference; otherwise this returns None (a decrement ``x - k`` is
    handled by :func:`_decrement`).
    """
    t = e[0]
    if t == "num":
        return ZERO, None, Fraction(e[1])
    if t == "var":
        return ONE, e[1], ZERO
    if t in ("add", "sub"):
        l, r = _affine(e[1]), _affine(e[2])
        if l is None or r is None:
            return None
        if l[1] is not None and r[1] is not None and l[1] != r[1]:
            return None
        sign = 1 if t == "add" else -1
        var = l[1] if l[1] is not None else r[1]
        a, b = l[0] + sign * r[0], l[2] + sign * r[2]
        if t == "sub" and (a < 0 or b < 0):
            return None                                # monus may truncate: not affine
        return a, var, b
    if t == "mul":
        l, r = _affine(e[1]), _affine(e[2])
        if l is None or r is None:
            return None
        if l[1] is None:
            return l[2] * r[0], r[1], l[2] * r[2]
        if r[1] is None:
            return r[2] * l[0], l[1], r[2] * l[2]
        return None
    if t == "div":
        l, r = _affine(e[1]), _affine(e[2])
        if l is None or r is None or r[1] is not None or r[2] == 0:
            return None
        return l[0] / r[2], l[1], l[2] / r[2]
    return None


def _sum_of_vars(e: IR) -> Optional[Tuple[Dict[str, Fraction], Fraction]]:
    """Write ``e`` as ``sum_j a_j * var_j + b`` with constants, or return None (also for a subtraction that may be
    truncated at 0, see :func:`_affine`)."""
    t = e[0]
    if t == "num":
        return {}, Fraction(e[1])
    if t == "var":
        return {e[1]: ONE}, ZERO
    if t in ("add", "sub"):
        l, r = _sum_of_vars(e[1]), _sum_of_vars(e[2])
        if l is None or r is None:
            return None
        sign = 1 if t == "add" else -1
        coeffs = dict(l[0])
        for v, a in r[0].items():
            coeffs[v] = coeffs.get(v, ZERO) + sign * a
        constant = l[1] + sign * r[1]
        if t == "sub" and (constant < 0 or any(a < 0 for a in coeffs.values())):
            return None                                # monus may truncate
        return {v: a for v, a in coeffs.items() if a != 0}, constant
    if t == "mul":
        l, r = _sum_of_vars(e[1]), _sum_of_vars(e[2])
        if l is None or r is None:
            return None
        if not l[0]:
            return {v: l[1] * a for v, a in r[0].items()}, l[1] * r[1]
        if not r[0]:
            return {v: r[1] * a for v, a in l[0].items()}, r[1] * l[1]
    return None


def _decrement(e: IR) -> Optional[Tuple[str, int]]:
    """Write ``e`` as the decrement ``var - k`` (monus) by a natural constant ``k``, or return None. A chain
    ``(x - k1) - k2`` is ``x - (k1 + k2)``, as monus satisfies ``(a - b) - c = a - (b + c)``."""
    if e[0] != "sub":
        return None
    amount = _affine(e[2])
    if amount is None or amount[1] is not None or amount[2].denominator != 1 or amount[2] < 0:
        return None
    k = int(amount[2])
    if e[1][0] == "var":
        return e[1][1], k
    inner = _decrement(e[1])
    return None if inner is None else (inner[0], inner[1] + k)


def _int_pred(op: str, c: Fraction) -> Tuple[Callable[[int], bool], int]:
    """Predicate on naturals for ``v op c`` and the threshold beyond which it is constant."""
    preds = {"<": lambda v: v < c, "<=": lambda v: v <= c, ">": lambda v: v > c, ">=": lambda v: v >= c,
             "=": lambda v: v == c}
    return preds[op], max(0, math.floor(c) + 1)


# ------------------------------------------------------------------------------------------- the calculus
class GFCalculus(ProductFormSeries):
    """Exact transformer semantics of loop-free statements on numerators over a fixed denominator (the series
    operations are those of :class:`~prodigy.distribution.product_form.ProductFormSeries`)."""

    def restrict_pred(self, poly: Num, i: int, pred: Callable[[int], bool], threshold: int, modulus: int) -> Num:
        """Restriction of ``P / D`` to the states whose ``i``-th component satisfies ``pred``, where ``pred(v)``
        depends only on ``v mod modulus`` for ``v >= threshold``; requires ``modulus | p_i`` if ``r_i != 0``."""
        r, p = self.den.r[i], self.den.p[i]
        if r == 0:
            return {e: c for e, c in poly.items() if pred(e[i])}
        if p % modulus != 0:
            raise UnsupportedProgram(
                f"the period {p} of variable {self.vars[i]} is not a multiple of the modulus {modulus} of a guard")
        n = degrees(poly, self.d)[i]
        bound = max(threshold + p, n + 1)
        cs = self.coeffs_along(poly, i, bound - 1)
        g = [cs[v] if pred(v) else {} for v in range(bound)]
        out: Num = {}
        for v in range(bound):
            q = g[v]
            if v - p >= 0:
                q = padd(q, pscale(g[v - p], -r))
            for e, c in q.items():
                e2 = list(e)
                e2[i] = v
                out[tuple(e2)] = out.get(tuple(e2), 0) + c
        return {e: c for e, c in out.items() if c != 0}

    def _marginal_out(self, poly: Num, i: int) -> Num:
        """Numerator over ``D`` of ``F(X_i = 1)`` (the marginal without variable ``i``)."""
        elim = self.eliminate(poly, i)
        if elim is None:
            raise UnsupportedProgram(
                f"cannot assign to {self.vars[i]}: the measure is infinite in this variable (rate {self.den.r[i]})")
        return self._over_full(elim, i)

    # ---------------------------------------------------------------- guards
    def restrict(self, poly: Num, guard: IR) -> Num:
        """Restriction ``<P/D>_guard`` of the measure to the states satisfying ``guard``."""
        t = guard[0]
        if not poly:
            return {}
        if t == "true":
            return poly
        if t == "false":
            return {}
        if t == "and":
            return self.restrict(self.restrict(poly, guard[1]), guard[2])
        if t == "or":
            left = self.restrict(poly, guard[1])
            right = self.restrict(poly, guard[2])
            both = self.restrict(left, guard[2])
            return padd(left, right, pscale(both, -1))
        if t == "not":
            return padd(poly, pscale(self.restrict(poly, guard[1]), -1))
        if t == "cmp":
            op, lhs, rhs = guard[1], guard[2], guard[3]
            # modulo guards  x % m = c
            for a, b in ((lhs, rhs), (rhs, lhs)):
                if a[0] == "mod" and a[1][0] == "var" and a[2][0] == "num" and b[0] == "num" and op == "=":
                    i, m, res = self.idx[a[1][1]], int(a[2][1]), Fraction(b[1])
                    if res.denominator != 1:
                        return {}
                    residue = int(res)
                    return self.restrict_pred(poly, i, lambda v: v % m == residue, 0, m)
            la, ra = _affine(lhs), _affine(rhs)
            if la is not None and ra is not None:
                a, var, b = la[0] - ra[0], la[1] if la[1] is not None else ra[1], ra[2] - la[2]
                if la[1] is not None and ra[1] is not None and la[1] != ra[1]:
                    raise UnsupportedProgram(f"unsupported guard {guard} (relates two variables)")
                # a * var  op  b
                if var is None or a == 0:
                    holds = {"<": 0 < b, "<=": 0 <= b, ">": 0 > b, ">=": 0 >= b, "=": 0 == b}[op]
                    return poly if holds else {}
                c = b / a
                if a < 0:
                    op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "="}[op]
                pred, thr = _int_pred(op, c)
                return self.restrict_pred(poly, self.idx[var], pred, thr, 1)
        raise UnsupportedProgram(f"unsupported guard {guard}")

    # ---------------------------------------------------------------- statements
    def run(self, stmt: IR, poly: Num) -> Num:
        """Forward semantics of a loop-free statement on the measure ``P / D`` (unnormalised; ``observe`` removes
        the violating mass, ``abort`` diverges)."""
        t = stmt[0]
        if t == "seq":
            for s in stmt[1]:
                poly = self.run(s, poly)
            return poly
        if not poly:
            return {}
        if t == "skip":
            return poly
        if t == "abort":
            return {}
        if t == "observe":
            return self.restrict(poly, stmt[1])
        if t == "choice":
            prob = eval_number(stmt[1])
            if not 0 <= prob <= 1:
                raise UnsupportedProgram(f"probability {prob} not in [0, 1]")
            return padd(pscale(self.run(stmt[2], poly), prob), pscale(self.run(stmt[3], poly), 1 - prob))
        if t == "if":
            pt = self.restrict(poly, stmt[1])
            pf = padd(poly, pscale(pt, -1))
            return padd(self.run(stmt[2], pt), self.run(stmt[3], pf))
        if t == "assign":
            return self.assign(stmt[1], stmt[2], poly)
        raise UnsupportedProgram(f"unsupported statement {t}")

    def assign(self, var: str, e: IR, poly: Num) -> Num:
        if var not in self.idx:
            raise UnsupportedProgram(f"unknown variable {var}")
        i = self.idx[var]
        dec = _decrement(e)
        if dec is not None and dec[0] == var:          # x := x - k  (monus)
            return self.decrement(poly, i, dec[1]) if dec[1] > 0 else poly
        aff = _affine(e)
        if aff is not None:
            a, v, b = aff
            if v is None:                              # x := const
                if b.denominator != 1 or b < 0:
                    raise UnsupportedProgram(f"assignment {var} := {b} is not natural-valued")
                return pshift(self._marginal_out(poly, i), i, int(b))
            if v == var and a == 1:                    # x := x + k  (k >= 0: subtractions never truncate here)
                if b.denominator != 1:
                    raise UnsupportedProgram(f"non-integral increment in {var} := {e}")
                return pshift(poly, i, int(b))
            if v != var and a.denominator == 1 and a > 0 and b.denominator == 1 and b >= 0:
                # x := a*y + b  for a variable y without geometric factor: substitute X_y -> X_y X_x^a
                j = self.idx[v]
                if self.den.r[j] != 0:
                    raise UnsupportedProgram(f"{var} := {e} needs a finite-support template in {v}")
                base = self._marginal_out(poly, i)
                out: Num = {}
                for ex, c in base.items():
                    e2 = list(ex)
                    e2[i] += int(a) * ex[j] + int(b)
                    out[tuple(e2)] = out.get(tuple(e2), 0) + c
                return {ex: c for ex, c in out.items() if c != 0}
            raise UnsupportedProgram(f"unsupported assignment {var} := {e}")
        sum_form = _sum_of_vars(e)
        if sum_form is not None and var in sum_form[0] and sum_form[0][var] == 1:
            # x := x + sum_j a_j y_j + b  for variables y_j without geometric factor
            coeffs, b = sum_form
            if b.denominator != 1 or b < 0:
                raise UnsupportedProgram(f"unsupported assignment {var} := {e}")
            shifts = {}
            for v, a in coeffs.items():
                if v == var:
                    continue
                j = self.idx[v]
                if self.den.r[j] != 0 or a.denominator != 1 or a < 0:
                    raise UnsupportedProgram(f"{var} := {e} needs finite-support templates in the added variables")
                shifts[j] = int(a)
            out = {}
            for ex, c in poly.items():
                e2 = list(ex)
                e2[i] += sum(a * ex[j] for j, a in shifts.items()) + int(b)
                out[tuple(e2)] = out.get(tuple(e2), 0) + c
            return {ex: c for ex, c in out.items() if c != 0}
        if e[0] == "call":
            name, args = e[1], e[2]
            if name == "geometric":
                q = eval_number(args[0])
                if self.den.r[i] != 1 - q or self.den.p[i] != 1:
                    raise UnsupportedProgram(
                        f"sampling {var} := geometric({q}) needs the rate {1 - q} and period 1 for {var} "
                        f"(the denominator has rate {self.den.r[i]} and period {self.den.p[i]})")
                # F(X_i = 1) * q / (1 - (1-q) X_i): the factor (1 - (1-q) X_i) is already part of D
                elim = self.eliminate(poly, i)
                if elim is None:
                    raise UnsupportedProgram(f"the measure is infinite in {var}")
                return pscale(elim, q)
            base = self._marginal_out(poly, i)
            if name in ("unif", "unif_d"):
                lo, hi = int(eval_number(args[0])), int(eval_number(args[1]))
                w = Fraction(1, hi - lo + 1)
                dist = {}
                for k in range(lo, hi + 1):
                    ee = [0] * self.d
                    ee[i] = k
                    dist[tuple(ee)] = w
                return pmul(base, dist)
            if name == "bernoulli":
                q = eval_number(args[0])
                ee = [0] * self.d
                ee[i] = 1
                return pmul(base, padd({tuple([0] * self.d): 1 - q}, {tuple(ee): q}))
            if name == "binomial":
                n, q = int(eval_number(args[0])), eval_number(args[1])
                dist = {}
                for k in range(n + 1):
                    ee = [0] * self.d
                    ee[i] = k
                    dist[tuple(ee)] = Fraction(math.comb(n, k)) * q ** k * (1 - q) ** (n - k)
                return pmul(base, padd(dist))
        raise UnsupportedProgram(f"unsupported assignment {var} := {e}")

    def decrement(self, poly: Num, i: int, k: int) -> Num:
        """``x_i := x_i - k`` with monus semantics (values below ``k`` become ``0``)."""
        high = self.restrict_pred(poly, i, lambda v: v >= k, k, 1)
        low = padd(poly, pscale(high, -1))
        high = pshift(high, i, -k)
        slices = self.coeffs_along(low, i, k - 1)
        low = self._over_full(padd(*slices) if slices else {}, i)
        return padd(high, low)


def directed_decimal(q: Union[int, float, Fraction], upward: bool, digits: int = 10) -> str:
    """Decimal string with ``digits`` significant digits rounded towards ``+inf`` (``upward``) or ``-inf``."""
    if q == math.inf:
        return "inf"
    q = Fraction(q)
    if q == 0:
        return "0"
    a = abs(q)
    e = 0
    while Fraction(10) ** e > a:
        e -= 1
    while Fraction(10) ** (e + 1) <= a:
        e += 1
    scale = Fraction(10) ** (digits - 1 - e)
    v = q * scale
    n = math.floor(v)
    if upward and n != v:
        n += 1
    s = str(abs(n))
    k = digits - 1 - e
    if k > 0:
        s = s.rjust(k + 1, "0")
        s = s[:-k] + "." + s[-k:]
        s = s.rstrip("0").rstrip(".")
    else:
        s = s + "0" * (-k)
    return ("-" if n < 0 else "") + s
