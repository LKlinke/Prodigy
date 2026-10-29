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

Coefficient-wise nonnegativity of ``P / D`` is decided exactly by finitely many coefficients (Lemma "finite
positivity check" of the paper): ``P / D >= 0`` iff all coefficients in the box ``prod_i [0, deg_i P + p_i - 1]`` are
nonnegative.
"""
from __future__ import annotations

import math
from fractions import Fraction

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from prodigy.analysis.lowerbounds.ir import IR, UnsupportedProgram, eval_number

Exp = Tuple[int, ...]
Num = Dict[Exp, Fraction]
Value = Union[Fraction, float]

ONE = Fraction(1)
ZERO = Fraction(0)


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

    def __eq__(self, other):
        return isinstance(other, Denominator) and self.r == other.r and self.p == other.p

    def __repr__(self):
        return f"Denominator(rates={[str(r) for r in self.r]}, periods={self.p})"


# ------------------------------------------------------------------------------------------- polynomials
def padd(*polys: Num) -> Num:
    out: Num = {}
    for poly in polys:
        for e, c in poly.items():
            out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c != 0}


def pscale(poly: Num, a) -> Num:
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


def monomial(d: int, e: Iterable[int], c=ONE) -> Num:  # pylint: disable=unused-argument
    return {tuple(e): Fraction(c)} if c != 0 else {}


# ------------------------------------------------------------------------------------------- linear guards
def _affine(e: IR) -> Optional[Tuple[Fraction, Optional[str], Fraction]]:
    """Write ``e`` as ``a * var + b`` with constants ``a, b`` (``var`` may be None), or return None."""
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
        return l[0] + sign * r[0], var, l[2] + sign * r[2]
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
    """Write ``e`` as ``sum_j a_j * var_j + b`` with constants, or return None."""
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
        return {v: a for v, a in coeffs.items() if a != 0}, l[1] + sign * r[1]
    if t == "mul":
        l, r = _sum_of_vars(e[1]), _sum_of_vars(e[2])
        if l is None or r is None:
            return None
        if not l[0]:
            return {v: l[1] * a for v, a in r[0].items()}, l[1] * r[1]
        if not r[0]:
            return {v: r[1] * a for v, a in l[0].items()}, r[1] * l[1]
    return None


def _int_pred(op: str, c: Fraction):
    """Predicate on naturals for ``v op c`` and the threshold beyond which it is constant."""
    preds = {"<": lambda v: v < c, "<=": lambda v: v <= c, ">": lambda v: v > c, ">=": lambda v: v >= c,
             "=": lambda v: v == c}
    return preds[op], max(0, math.floor(c) + 1)


# ------------------------------------------------------------------------------------------- the calculus
class GFCalculus:
    """Exact transformer semantics of loop-free statements on numerators over a fixed denominator."""

    def __init__(self, variables: Sequence[str], den: Denominator):
        self.vars = list(variables)
        self.idx = {v: i for i, v in enumerate(self.vars)}
        self.den = den
        self.d = len(self.vars)
        if den.d != self.d:
            raise ValueError("denominator does not match the number of variables")

    def with_den(self, den: Denominator) -> "GFCalculus":
        return GFCalculus(self.vars, den)

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

    def restrict_pred(self, poly: Num, i: int, pred, threshold: int, modulus: int) -> Num:
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
                    return self.restrict_pred(poly, i, lambda v, m=m, res=int(res): v % m == res, 0, m)
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
        aff = _affine(e)
        if aff is not None:
            a, v, b = aff
            if v is None:                              # x := const
                if b.denominator != 1 or b < 0:
                    raise UnsupportedProgram(f"assignment {var} := {b} is not natural-valued")
                return pshift(self._marginal_out(poly, i), i, int(b))
            if v == var and a == 1:                    # x := x + k  /  x := x - k  (monus)
                if b.denominator != 1:
                    raise UnsupportedProgram(f"non-integral increment in {var} := {e}")
                k = int(b)
                if k >= 0:
                    return pshift(poly, i, k)
                return self.decrement(poly, i, -k)
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
                sl_v: List[Any] = [slice(None)] * self.d
                sl_v[i] = v
                sl_w: List[Any] = [slice(None)] * self.d
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
        def mono(e, c):
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


def directed_decimal(q, upward: bool, digits: int = 10) -> str:
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
