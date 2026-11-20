"""
The signs of the coefficients of a univariate rational function with a unique dominant simple pole (#431).

Let ``f = P / Q`` be reduced, with rational coefficients and ``Q(0) != 0``, and let ``Q`` have a root ``rho``
of multiplicity 1 whose modulus is strictly smaller than that of every other root. Then ``rho`` is real
(``Q`` has real coefficients, so ``conj(rho)`` is a root of the same modulus and hence ``rho`` itself). Write
``P / Q = E + R / Q`` with a polynomial ``E`` of degree ``e`` and ``deg R < deg Q``, and

    R / Q = A / (1 - x/rho) + T / Q1,    Q = (1 - x/rho) * Q1,    A = R(rho) / Q1(rho) != 0,

where ``Q1 = Q(0) * prod_beta (1 - x/beta)`` has the ``d = deg Q - 1`` other roots ``beta`` (with their
multiplicities), all of modulus at least ``sigma > |rho|``, and ``T = (R - A*Q1) / (1 - x/rho)`` is a
polynomial of degree ``< d``. The coefficients of ``prod_beta 1/(1 - x/beta)`` are bounded in modulus by those
of ``1/(1 - x/sigma)^d``, i.e. by ``binom(n+d-1, d-1) sigma^-n``; hence for ``n > e``

    a_n = A rho^-n + s_n,    |s_n| <= K binom(n+d-1, d-1) sigma^-n,    K = sum_k |t_k| sigma^k / |Q(0)|.

So ``a_n`` has the sign of ``A rho^-n`` as soon as ``binom(n+d-1, d-1) (|rho|/sigma)^n < |A| / K``; the
left-hand side decreases once ``(n+d) |rho|/sigma < n+1``, which gives an explicit index ``N`` beyond which
every coefficient has the sign of the dominant term. The coefficients up to ``N`` are computed exactly. This
decides the class:

* ``rho > 0`` and ``A > 0``: all coefficients beyond ``N`` are positive; non-negative iff the first ``N + 1``
  are (a negative one is the witness);
* ``A < 0`` or ``rho < 0``: the dominant term is negative for every ``n`` (or every other ``n``), so a
  coefficient among the first ``N + 3`` is negative and is returned as the witness.

Proof steps never use floating point: the roots are isolated by FLINT (``fmpz_poly.complex_roots``, Arb balls
with exact multiplicities), every quantity above is an Arb ball that contains the true value, and a
comparison counts only if it holds for the whole balls (outward rounding). If the moduli of two roots cannot
be separated at the highest precision (e.g. ``rho`` and ``-rho``, or a complex pair), the dominant pole is not
simple, or ``N`` exceeds :data:`MAX_EXACT_COEFFICIENTS`, the verdict is undecided.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional, Sequence, Tuple, cast

import flint
import sympy

PRECISIONS = (64, 128, 256, 512, 1024)
"""The working precisions (bits) of Arb, tried in this order until every comparison is decided."""

MAX_EXACT_COEFFICIENTS = 20000
"""The largest index ``N`` up to which the coefficients are computed exactly; beyond it, undecided."""

Witness = Tuple[int, Fraction]
"""A negative coefficient: its index and its exact value."""


@dataclass(frozen=True)
class PoleVerdict:
    """``holds``: True if all coefficients are proven non-negative, False with a ``witness`` (a negative
    coefficient), None with a ``reason`` if the rational function is not in the class or not decided."""
    holds: Optional[bool]
    reason: str = ""
    witness: Optional[Witness] = None


class _Undecided(Exception):
    """An Arb comparison or a bound is not decided at the current precision."""


class _UndecidedPermanently(_Undecided):
    """Outside the class or the limits at every precision (e.g. a multiple dominant pole)."""


def _fmpq_poly(expr: sympy.Expr, symbol: sympy.Symbol) -> Optional[flint.fmpq_poly]:
    try:
        poly = sympy.Poly(expr, symbol, domain=sympy.QQ)
    except (sympy.PolynomialError, sympy.CoercionFailed):
        return None
    coefficients = [flint.fmpq(int(c.p), int(c.q)) for c in reversed(poly.all_coeffs())]
    return flint.fmpq_poly(coefficients)


def dominant_pole_verdict(function: sympy.Expr, symbol: sympy.Symbol) -> PoleVerdict:
    """The verdict on the signs of the Taylor coefficients at 0 of ``function``, a rational function in
    ``symbol`` with rational coefficients (see the module documentation). Anything else is undecided."""
    numerator, denominator = sympy.fraction(sympy.cancel(sympy.together(function)))
    p = _fmpq_poly(numerator, symbol)
    q = _fmpq_poly(denominator, symbol)
    if p is None or q is None:
        return PoleVerdict(None, "it is not a rational function with rational coefficients in one variable")
    return rational_verdict(p, q)


def rational_verdict(p: flint.fmpq_poly, q: flint.fmpq_poly) -> PoleVerdict:
    """:func:`dominant_pole_verdict` for ``P / Q`` given as polynomials over ℚ."""
    common = p.gcd(q)
    if common.degree() > 0:
        p, q = p // common, q // common
    if q[0] == 0:
        return PoleVerdict(None, "it has a pole at 0")
    if q.degree() < 1:
        return PoleVerdict(None, "it is a polynomial")
    previous = flint.ctx.prec
    try:
        for precision in PRECISIONS:
            flint.ctx.prec = precision
            try:
                return _decide(p, q)
            except _UndecidedPermanently as undecided:
                return PoleVerdict(None, str(undecided))
            except _Undecided as undecided:
                reason = str(undecided)
    finally:
        flint.ctx.prec = previous
    return PoleVerdict(None, reason)


def _arb(value: flint.fmpq) -> flint.arb:
    return flint.arb(value.p) / flint.arb(value.q)


def _evaluate(coefficients: Sequence[flint.arb], point: flint.arb) -> flint.arb:
    result = flint.arb(0)
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


def _deflated(coefficients: Sequence[flint.arb], rho: flint.arb, length: int) -> List[flint.arb]:
    """The first ``length`` coefficients of ``F / (1 - x/rho)`` for a polynomial ``F`` divisible by it:
    ``s_0 = f_0``, ``s_i = f_i + s_(i-1)/rho`` (the exact recursion; balls contain the exact quotient)."""
    result: List[flint.arb] = []
    for index in range(length):
        result.append(coefficients[index] + (result[-1] / rho if result else 0))
    return result


def _dominant(q: flint.fmpq_poly) -> Tuple[flint.arb, flint.arb, int]:
    """``(rho, sigma, d)``: the dominant root (a ball around the real root), a lower bound of the moduli of
    the other roots, and their number with multiplicities."""
    # python-flint 0.9 returns (root, multiplicity) pairs; its type stubs say a list of roots
    roots = cast(List[Tuple[flint.acb, int]], q.numer().complex_roots())
    moduli = [abs(root) for root, _ in roots]
    # the candidate is chosen by the midpoints; only the comparisons of whole balls below prove it dominant
    index = min(range(len(roots)), key=lambda i: float(moduli[i].mid()))
    others = [moduli[i] for i in range(len(roots)) if i != index]
    if not all(moduli[index] < other for other in others):
        raise _Undecided("its poles of smallest modulus could not be separated (no unique dominant pole)")
    root, multiplicity = roots[index]
    if multiplicity != 1:
        raise _UndecidedPermanently(f"its dominant pole has multiplicity {multiplicity}, not 1")
    sigma = flint.arb(0)
    for position, other in enumerate(others):
        lower = other.lower()  # an exact point, so the comparison is exact
        if position == 0 or lower < sigma:
            sigma = lower
    return root.real, sigma, q.degree() - 1


def _sign(value: flint.arb, what: str) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    raise _Undecided(f"the sign of {what} could not be decided")


def _bound_index(rho: flint.arb, sigma: flint.arb, d: int, ratio: flint.arb) -> int:
    """The least ``N`` such that ``binom(n+d-1, d-1) (|rho|/sigma)^n < ratio`` for every ``n >= N``
    (``ratio = |A| / K`` as a lower bound)."""
    q = abs(rho).upper() / sigma
    if not q < 1:
        raise _Undecided("its dominant pole could not be separated from the others")
    for n in range(MAX_EXACT_COEFFICIENTS + 1):
        decreasing = flint.arb(n + d) * q < n + 1
        if decreasing and flint.arb(math.comb(n + d - 1, d - 1)) * q ** n < ratio:
            return n
    raise _UndecidedPermanently(f"the index from which on its dominant pole decides the signs exceeds "
                                f"{MAX_EXACT_COEFFICIENTS}")


def _decide(p: flint.fmpq_poly, q: flint.fmpq_poly) -> PoleVerdict:
    polynomial, remainder = divmod(p, q)
    rho, sigma, d = _dominant(q)
    q_arb = [_arb(c) for c in q.coeffs()]
    r_arb = [_arb(remainder[i]) for i in range(q.degree())]
    q1 = _deflated(q_arb, rho, d + 1)
    amplitude = _evaluate(r_arb, rho) / _evaluate(q1, rho)
    dominant_sign = _sign(amplitude, "the residue at its dominant pole")
    pole_sign = _sign(rho, "its dominant pole")
    if d == 0:
        index = 0
    else:
        t = _deflated([r - amplitude * s for r, s in zip(r_arb, q1)], rho, d)
        k = sum((abs(c) * sigma ** i for i, c in enumerate(t)), flint.arb(0)) / abs(_arb(q[0]))
        k_upper = k.upper()
        index = 0 if k_upper == 0 else _bound_index(rho, sigma, d, abs(amplitude).lower() / k_upper)
    index = max(index, polynomial.degree() + 1)
    count = index + (1 if dominant_sign > 0 and pole_sign > 0 else 3)
    if count > MAX_EXACT_COEFFICIENTS:
        raise _UndecidedPermanently(f"the index from which on its dominant pole decides the signs exceeds "
                                    f"{MAX_EXACT_COEFFICIENTS}")
    for n, coefficient in enumerate(_taylor(p, q, count)):
        if coefficient < 0:
            return PoleVerdict(False, witness=(n, Fraction(int(coefficient.p), int(coefficient.q))))
    if dominant_sign > 0 and pole_sign > 0:
        return PoleVerdict(True)
    # unreachable by the bound above: a negative coefficient lies among the first `count`
    return PoleVerdict(None, "no negative coefficient was found where the bound predicts one")


def _taylor(p: flint.fmpq_poly, q: flint.fmpq_poly, count: int) -> Iterator[flint.fmpq]:
    """The first ``count`` Taylor coefficients of ``P / Q`` at 0, exactly (``Q * S = P``)."""
    q0 = q[0]
    computed: List[flint.fmpq] = []
    degree = q.degree()
    for n in range(count):
        value = p[n] - sum((q[k] * computed[n - k] for k in range(1, min(n, degree) + 1)), flint.fmpq(0))
        computed.append(value / q0)
        yield computed[-1]
