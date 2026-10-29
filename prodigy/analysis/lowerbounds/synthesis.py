# pylint: disable=invalid-name,import-outside-toplevel,use-dict-literal
# Names follow the mathematics (H, Phi, A, b, ...); scipy and FLINT are imported where they are needed.
r"""
Synthesis of EVT (occupation) sub- and superinvariants with eventually geometric templates by linear
programming, with exact rational certificates.

Template.  Fix a denominator ``D = prod_i (1 - r_i X_i^{p_i})`` and tail starts ``B_i``.  A template is given by its
values ``h_n`` on the fundamental box ``F = prod_i [0, B_i + p_i - 1]`` (``[0, B_i]`` if ``r_i = 0``); beyond ``F`` the
values continue geometrically, ``h_{n + p_i e_i} = r_i h_n`` for ``n_i >= B_i``.  Its generating function ``H = N / D``
is linear in ``h``.

Constraints.  ``H >= 0`` iff ``h >= 0``.  The residual ``Phi(H) - H = g + T(H) - H`` has a numerator over the same
``D`` that is affine in ``h``, and its coefficient-wise sign is decided exactly by a finite box of coefficients.
Hence lower certificates (``H <= Phi(H)``, finite mass; maximise the exit mass) and upper certificates
(``Phi(H) <= H``; minimise an objective) are linear programs.

Exactness.  The LP is solved in floating point (HiGHS).  The optimal vertex is then reconstructed in exact
rational arithmetic (the tight constraints form a square system that is solved with FLINT) and every constraint
is re-checked exactly.  Only exactly verified certificates are returned.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from prodigy.analysis.lowerbounds.egcalc import Denominator, GFCalculus, Num, degrees, padd, pmul, pscale
from prodigy.analysis.lowerbounds.ir import IR


@dataclass
class Certificate:
    """An exactly verified template certificate ``H = numerator / D``."""
    kind: str                                  # "lower" (sub-invariant of finite mass) or "upper" (super-invariant)
    calc: GFCalculus
    tail_start: List[int]
    numerator: Num
    exit_numerator: Num
    template_size: int
    margin: Optional[float]
    time_build: float
    time_lp: float
    exit_mass: Any = None
    total_mass: Any = None
    extra: dict = field(default_factory=dict)

    @property
    def den(self) -> Denominator:
        return self.calc.den

    def to_json(self) -> dict:
        return dict(kind=self.kind,
                    rule=("finite-mass EVT sub-invariant (lower induction)" if self.kind == "lower"
                          else "EVT super-invariant (Park induction)"),
                    variables=self.calc.vars, rates=[str(r) for r in self.den.r], periods=list(self.den.p),
                    tail_start=list(self.tail_start), template_size=self.template_size,
                    numerator=[[list(e), str(c)] for e, c in sorted(self.numerator.items())],
                    generating_function=self.calc.to_expr(self.numerator),
                    exit_mass=str(self.exit_mass), total_mass=str(self.total_mass))


def scaled_expansion(calc: GFCalculus, polys: Sequence[Num], box: Sequence[int]):
    """Coefficients of ``P_j / D`` on the box ``prod_i [0, box_i]`` for all ``P_j`` at once, as an exact integer
    array ``M[n, j] = S(n) * coeff_n(P_j / D)`` with positive row scales ``S(n) = L * prod_i b_i^floor(n_i / p_i)``
    (``r_i = a_i / b_i``, ``L`` a common denominator of the coefficients), and as a floating-point array of the
    unscaled coefficients.  Returns ``(M, F, S)`` with rows in C order of the box."""
    from math import lcm
    d, den = calc.d, calc.den
    common = 1
    for poly in polys:
        for c in poly.values():
            common = lcm(common, c.denominator)
    shape = [b + 1 for b in box] + [len(polys)]
    arr = np.zeros(shape, dtype=object)
    farr = np.zeros(shape, dtype=float)
    for j, poly in enumerate(polys):
        for e, c in poly.items():
            if all(e[k] <= box[k] for k in range(d)):
                arr[e + (j,)] += c.numerator * (common // c.denominator)
                farr[e + (j,)] += float(c)
    axis_scales = []
    for i in range(d):
        r, p, n_i = den.r[i], den.p[i], shape[i]
        if r == 0:
            axis_scales.append([1] * n_i)
            continue
        a, b = r.numerator, r.denominator
        scales = [b ** (v // p) for v in range(n_i)]
        axis_scales.append(scales)
        for v in range(n_i):
            if scales[v] != 1:
                idx: List[Any] = [slice(None)] * (d + 1)
                idx[i] = v
                arr[tuple(idx)] *= scales[v]
        rf = float(r)
        for v in range(p, n_i):
            iv: List[Any] = [slice(None)] * (d + 1)
            iv[i] = v
            iw: List[Any] = [slice(None)] * (d + 1)
            iw[i] = v - p
            arr[tuple(iv)] += a * arr[tuple(iw)]
            farr[tuple(iv)] += rf * farr[tuple(iw)]
    rows = int(np.prod(shape[:-1]))
    scale = []
    for cell in np.ndindex(*shape[:-1]):
        sc = common
        for i in range(d):
            sc *= axis_scales[i][cell[i]]
        scale.append(sc)
    return arr.reshape(rows, len(polys)), farr.reshape(rows, len(polys)), scale


class TemplateLP:
    """The linear program for one template shape (denominator, tail starts, optional support restriction).

    The constraint data are kept as exact integers: row ``n`` of ``A h + b`` is scaled by a positive integer
    ``S(n)``, which does not change its sign."""

    def __init__(self, calc: GFCalculus, init: Num, guard: IR, body: IR, tail_start: Sequence[int],
                 support: Optional[Callable[[Tuple[int, ...]], bool]] = None):
        t0 = time.perf_counter()
        self.calc, self.guard, self.body = calc, guard, body
        den, d = calc.den, calc.d
        self.d = d
        self.tail_start = list(tail_start)
        hi = [tail_start[i] + den.p[i] - 1 if den.r[i] != 0 else tail_start[i] for i in range(d)]
        self.basis = [tuple(int(x) for x in e) for e in np.ndindex(*[h + 1 for h in hi])]
        if support is not None:
            self.basis = [e for e in self.basis if support(e)]
        self.init = init
        self.basis_num: List[Num] = []
        residuals: List[Num] = []
        for n in self.basis:
            num = {n: Fraction(1)}
            for i in range(d):
                if den.r[i] != 0 and n[i] < tail_start[i]:
                    num = pmul(num, den.factor(i))
            self.basis_num.append(num)
            image = calc.run(body, calc.restrict(num, guard))
            residuals.append(padd(image, pscale(num, -1)))
        deg = [0] * d
        for poly in residuals + [init]:
            deg = [max(a, b) for a, b in zip(deg, degrees(poly, d))]
        self.box = [deg[i] + (den.p[i] - 1 if den.r[i] != 0 else 0) for i in range(d)]
        mat, fmat, scale = scaled_expansion(calc, residuals + [init], self.box)
        self.A, self.b = mat[:, :-1], mat[:, -1]
        self.Af, self.bf = fmat[:, :-1], fmat[:, -1]
        self.row_scale = scale
        self.time_build = time.perf_counter() - t0
        self.time_lp = 0.0
        self.margin_used: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.basis)

    # ------------------------------------------------------------------ objectives
    def numerator(self, h: Sequence[Fraction]) -> Num:
        out: Num = {}
        for hv, num in zip(h, self.basis_num):
            if hv != 0:
                for e, c in num.items():
                    out[e] = out.get(e, 0) + hv * c
        return {e: c for e, c in out.items() if c != 0}

    def exit_functional(self, query: Optional[IR] = None, moment: Optional[int] = None) -> List:
        """Coefficients of the linear functional ``h -> |[not guard] H|`` (restricted to ``query``) or of the
        moment of variable ``moment`` of the exit part."""
        out = []
        for num in self.basis_num:
            ex = self.calc.restrict(num, ("not", self.guard))
            if query is not None:
                ex = self.calc.restrict(ex, query)
            out.append(self.calc.mass(ex) if moment is None else self.calc.moment(ex, moment))
        return out

    def exit_box_functional(self, query: Optional[IR] = None) -> List:
        """Like :meth:`exit_functional`, but only counting the exit mass inside the fundamental box (finite even if
        the template has rates >= 1)."""
        hi = [t + p - 1 if r != 0 else t for t, r, p in zip(self.tail_start, self.calc.den.r, self.calc.den.p)]
        out = []
        for num in self.basis_num:
            ex = self.calc.restrict(num, ("not", self.guard))
            if query is not None:
                ex = self.calc.restrict(ex, query)
            arr = self.calc.expand(ex, hi) if ex else None
            out.append(sum(arr.flat, Fraction(0)) if arr is not None else Fraction(0))
        return out

    # ------------------------------------------------------------------ solving
    def column_scales(self) -> np.ndarray:
        """Expected magnitudes ``prod_i r_i^(n_i / p_i)`` of the template values (factor 1 for truncated variables).
        The values of a certificate typically decrease like this across the box, over many orders of magnitude.
        Scaling the LP variables by these magnitudes keeps the floating-point LP well-conditioned, lets the
        reconstruction tell small values from zeros, and makes the safety margins relative to the values."""
        den = self.calc.den
        logs = np.zeros(len(self.basis))
        for i in range(self.d):
            r = float(den.r[i])
            if r > 0:
                logs += np.array([e[i] for e in self.basis], dtype=float) * (math.log(r) / den.p[i])
        return np.exp(np.clip(logs, -700.0, 700.0))

    def solve(self, kind: str, objective: Optional[Sequence] = None,
              margins: Sequence[float] = (0.0, 1e-9, 1e-7)) -> Optional[List[Fraction]]:
        """Solve the LP for a lower (``kind = "lower"``, maximise) or upper (``kind = "upper"``, minimise)
        certificate and return exact template values that pass the exact check, or None."""
        from scipy.optimize import linprog
        t0 = time.perf_counter()
        if objective is None:
            objective = self.exit_functional()
        n = self.size
        if n == 0:
            return None
        c = np.array([float(x) if x != math.inf else 1e30 for x in objective])
        A = self.Af
        b = self.bf
        keep = np.asarray(np.any(self.A != 0, axis=1), dtype=bool)
        fixed = ~keep
        if kind == "lower":
            if any(v < 0 for v in self.b[fixed]):
                return None
            a_ub, b_ub, sense = -A[keep], b[keep], -1.0           # b + A h >= 0
        else:
            if any(v > 0 for v in self.b[fixed]):
                return None
            a_ub, b_ub, sense = A[keep], -b[keep], 1.0            # b + A h <= 0
        rows = np.where(keep)[0]
        # First with the unknowns scaled by their expected magnitudes (LP variables z = h / col), then unscaled: the
        # scaling helps when the values decay like the tails across the box, and can hurt when they decay much more
        # slowly (e.g. near criticality), because the reconstruction tolerances are relative to the largest value.
        # For each, first without a margin: the optimal vertex gives the best certificate.  A margin (in units of the
        # largest coefficient of each normalised row) only helps the exact reconstruction.
        infeasible = False
        for col in (self.column_scales(), np.ones(n)):
            a_c, c_c = a_ub * col[None, :], c * col
            norm = np.maximum(np.abs(a_c).max(axis=1), 1e-300)
            a_s, b_s = a_c / norm[:, None], b_ub / norm
            for mg in margins:
                res = linprog(sense * c_c, A_ub=a_s, b_ub=b_s - mg, bounds=[(0, None)] * n, method="highs",
                              options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
                if res.status == 2 and mg == 0.0:
                    infeasible = True                         # then so are the variants with margins or scaling
                    break
                if res.status != 0:
                    continue
                target = [Fraction(float(mg * nv)) * (1 if kind == "lower" else -1) * self.row_scale[r]
                          for nv, r in zip(norm, rows)]
                h = self._exact_vertex(res.x, a_s, b_s - mg, rows, kind, target)
                if h is not None:
                    self.time_lp = time.perf_counter() - t0
                    self.margin_used = mg
                    return h
            if infeasible or np.all(col == 1.0):
                break
        self.time_lp = time.perf_counter() - t0
        return None

    def _exact_vertex(self, x, a_s, b_s, rows, kind, target) -> Optional[List[Fraction]]:
        """Reconstruct the LP vertex exactly: the tight rows ``k`` satisfy ``b + A h = target_k`` exactly."""
        import flint
        import scipy.linalg
        n = len(x)
        slack = b_s - a_s @ x
        scale = max(1.0, float(np.abs(x).max()))
        for tol in (1e-9, 1e-8, 1e-7, 1e-10, 1e-11, 1e-6):
            tight = np.where(slack <= tol * scale)[0]
            free = [j for j in range(n) if x[j] > tol * scale]
            if not free:
                h = [Fraction(0)] * n
                if self.check(h, kind):
                    return h
                continue
            mat = a_s[tight][:, free]
            if mat.shape[0] < len(free):
                continue
            _, rq, piv = scipy.linalg.qr(mat.T, pivoting=True, mode="economic")
            diag = np.abs(np.diag(rq))
            rank = int((diag > 1e-11 * diag[0]).sum()) if len(diag) else 0
            if rank < len(free):
                continue
            sel = tight[piv[:len(free)]]
            a_ex = [[int(self.A[rows[s], j]) for j in free] for s in sel]
            b_ex = [target[s] - self.b[rows[s]] for s in sel]
            try:
                sol = flint.fmpq_mat(a_ex).solve(
                    flint.fmpq_mat([[flint.fmpq(v.numerator, v.denominator)] for v in b_ex]))
            except Exception:  # pylint: disable=broad-except
                continue
            h = [Fraction(0)] * n
            for k, j in enumerate(free):
                v = sol[k, 0]
                h[j] = Fraction(int(v.p), int(v.q))
            if self.check(h, kind):
                return h
        return None

    def check(self, h: Sequence[Fraction], kind: str) -> bool:
        """Exact verification: ``h >= 0`` and the residual has the right sign on the whole positivity box."""
        if any(v < 0 for v in h):
            return False
        from math import lcm
        den_h = 1
        for v in h:
            den_h = lcm(den_h, v.denominator)
        h_int = np.array([v.numerator * (den_h // v.denominator) for v in h], dtype=object)
        res = self.A.dot(h_int) + self.b * den_h
        if kind == "lower":
            return all(v >= 0 for v in res)
        return all(v <= 0 for v in res)

    def certificate(self, h: Sequence[Fraction], kind: str) -> Certificate:
        num = self.numerator(h)
        ex = self.calc.restrict(num, ("not", self.guard))
        cert = Certificate(kind=kind, calc=self.calc, tail_start=self.tail_start, numerator=num, exit_numerator=ex,
                           template_size=self.size, margin=self.margin_used, time_build=self.time_build,
                           time_lp=self.time_lp)
        cert.exit_mass = self.calc.mass(ex)
        cert.total_mass = self.calc.mass(num)
        return cert


def verify_certificate(calc: GFCalculus, init: Num, guard: IR, body: IR, numerator: Num, kind: str) -> Tuple[bool, str]:
    """Independent exact check of a certificate ``H = numerator / D`` (does not use the LP data):
    ``H >= 0``, and ``H <= g + T(H)`` with ``|H| < inf`` (lower) resp. ``g + T(H) <= H`` (upper)."""
    neg = calc.first_negative(numerator)
    if neg is not None:
        return False, f"H has the negative coefficient {neg[1]} at {neg[0]}"
    phi = padd(init, calc.run(body, calc.restrict(numerator, guard)))
    if kind == "lower":
        if calc.mass(numerator) == math.inf:
            return False, "H has infinite mass"
        bad = calc.first_negative(padd(phi, pscale(numerator, -1)))
        if bad is not None:
            return False, f"Phi(H) - H has the negative coefficient {bad[1]} at {bad[0]}"
        return True, "H <= Phi(H) and |H| < inf"
    bad = calc.first_negative(padd(numerator, pscale(phi, -1)))
    if bad is not None:
        return False, f"H - Phi(H) has the negative coefficient {bad[1]} at {bad[0]}"
    return True, "Phi(H) <= H"
