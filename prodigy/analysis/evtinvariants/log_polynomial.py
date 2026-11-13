"""
Generating functions ``P + Q·log(1 - c·k)`` in one counter k: non-negativity, mass, convergence (#323).

A choice probability such as ``red/(red + 1)`` acts on the generating function as an integral operator
(:mod:`prodigy.analysis.instructionhandler.counter_operators`); applied to rational generating functions it produces
logarithms. The EVT invariant of the Pólya urn in ``yule_simon_urn.ppy`` is
``-log(1 - red) + stop·(-log(1 - red) + log(1 - red)/red + 1)``. This module handles the class

    f = P + Q·log(1 - c·k),

where P and Q are polynomials in the other program variables w whose coefficients are *Laurent* polynomials in k
(``log(1 - red)/red`` has the factor ``red^(-1)``) with rational coefficients, and c is a rational in (0, 1].

**Coefficients.** With ``log(1 - c k) = -Σ_(n≥1) c^n k^n / n``, the coefficient of ``w^α k^m`` is
``p_m - Σ_j q_j c^(m-j)/(m - j)`` (sum over the powers ``k^j`` of Q with ``m - j ≥ 1``). f is a power series iff these
vanish for ``m < 0``. For ``m > h`` (h the largest power in P or Q) the coefficient is ``c^m·S(m)`` with
``S(m) = -Σ_j q_j c^(-j)/(m - j)``, and ``S(m)·Π_j (m - j)`` is a polynomial N(m); the denominators are positive for
``m > h``. So finitely many coefficients are checked directly and the rest by the sign of N beyond Cauchy's root bound
(:func:`prodigy.analysis.evtinvariants.exp_polynomial.eventually_nonneg`).

**Convergence and mass.** ``log(1 - c k)`` has radius of convergence ``1/c ≥ 1``, so f converges on the open unit
polydisc (criterion (A) of :mod:`prodigy.analysis.evtinvariants.series_identity`). For the mass, f(t, …, t) is in
the same class; its coefficients for large m are ``c^m·S(m)``. For ``c < 1`` they are summable and the mass is f at 1
(analytic there). For ``c = 1``, ``S(m) = -(Σ_j q_j)/m + O(1/m²)``: the mass is finite iff ``Q(1) = Σ_j q_j = 0``, and
then ``Q(t) = (1 - t)·R(t)`` and ``(1 - t)·log(1 - t) → 0``, so by Abel's theorem the mass is ``P(1)``.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import sympy

from prodigy.analysis.evtinvariants.exp_polynomial import eventually_nonneg

Laurent = Dict[int, Fraction]
"""A Laurent polynomial in k: ``{j: q_j}`` stands for ``Σ q_j k^j``."""


class LogForm(NamedTuple):
    """``f = P + Q·log(1 - c·k)``; P and Q map the exponents of the other variables to Laurent polynomials in k."""
    counter: str
    rate: Fraction
    plain: Dict[Tuple[int, ...], Laurent]
    log: Dict[Tuple[int, ...], Laurent]


def log_form(expr: sympy.Expr, variables: Sequence[str]) -> Optional[LogForm]:
    """``expr`` as ``P + Q·log(1 - c·k)`` (module docstring), or None if it has another form, contains parameters or
    is no power series (a negative power of k does not cancel)."""
    expr = sympy.S(expr)
    names = set(variables)
    if any(s.name not in names for s in expr.free_symbols):
        return None
    atoms = list(expr.atoms(sympy.log))
    if len(atoms) != 1:
        return None
    argument = sympy.expand(atoms[0].args[0])
    symbols = list(argument.free_symbols)
    if len(symbols) != 1:
        return None
    k = symbols[0]
    poly = sympy.Poly(argument, k)
    if poly.degree() != 1 or poly.all_coeffs()[1] != 1 or not poly.all_coeffs()[0].is_Rational:
        return None
    rate = -Fraction(str(poly.all_coeffs()[0]))
    if not 0 < rate <= 1:
        return None
    ell = sympy.Dummy("l")
    substituted = expr.xreplace({atoms[0]: ell})
    numerator, denominator = sympy.fraction(sympy.together(substituted))
    constant, dependent = denominator.as_independent(k, as_Add=False)
    if not constant.is_Rational or constant == 0:
        return None
    shift = 0
    if dependent != 1:
        if dependent == k:
            shift = 1
        elif isinstance(dependent, sympy.Pow) and dependent.base == k and dependent.exp.is_Integer:
            shift = int(dependent.exp)
        else:
            return None
    others = sorted((s for s in expr.free_symbols if s != k), key=lambda s: s.name)
    try:
        numerator_poly = sympy.Poly(sympy.expand(numerator), ell, k, *others)
    except sympy.PolynomialError:
        return None
    plain: Dict[Tuple[int, ...], Laurent] = defaultdict(dict)
    logs: Dict[Tuple[int, ...], Laurent] = defaultdict(dict)
    for exponents, value in numerator_poly.terms():
        if exponents[0] > 1 or not value.is_Rational:
            return None
        target = logs if exponents[0] == 1 else plain
        power = exponents[1] - shift
        part = target[tuple(exponents[2:])]
        part[power] = part.get(power, Fraction(0)) + Fraction(str(value / constant))
    form = LogForm(k.name, rate, dict(plain), dict(logs))
    for monomial in set(form.plain) | set(form.log):
        low = min(list(form.plain.get(monomial, {})) + [j + 1 for j in form.log.get(monomial, {})] + [0])
        if any(_coefficient(form.plain.get(monomial, {}), form.log.get(monomial, {}), rate, m) != 0
               for m in range(low, 0)):
            return None
    return form


def _coefficient(p: Laurent, q: Laurent, c: Fraction, m: int) -> Fraction:
    """The coefficient of ``k^m`` in ``P(k) + Q(k)·log(1 - c k)``."""
    return p.get(m, Fraction(0)) - sum((q_j * c ** (m - j) / (m - j) for j, q_j in q.items() if m - j >= 1),
                                       Fraction(0))


def _series_nonneg(p: Laurent, q: Laurent, c: Fraction) -> Optional[bool]:
    p = {j: v for j, v in p.items() if v != 0}
    q = {j: v for j, v in q.items() if v != 0}
    head = max(list(p) + list(q) + [0])
    if any(_coefficient(p, q, c, m) < 0 for m in range(head + 1)):
        return False
    if not q:
        return True
    # N(m) = -Σ_j q_j c^(-j) Π_(i ≠ j) (m - i), positive multiple of the coefficients for m > head
    m_symbol = sympy.Symbol("m")
    numerator = -sum(sympy.Rational(q_j.numerator, q_j.denominator)
                     / sympy.Rational(c.numerator, c.denominator) ** j
                     * sympy.Mul(*(m_symbol - i for i in q if i != j)) for j, q_j in q.items())
    poly = sympy.Poly(sympy.expand(numerator), m_symbol)
    return eventually_nonneg([Fraction(str(x)) for x in poly.all_coeffs()], head)


def log_form_nonneg(expr: sympy.Expr, variables: Sequence[str]) -> Optional[bool]:
    """Decides non-negativity of all coefficients of ``P + Q·log(1 - c·k)`` (module docstring); None for other
    forms."""
    form = log_form(expr, variables)
    if form is None:
        return None
    verdict: Optional[bool] = True
    for monomial in set(form.plain) | set(form.log):
        result = _series_nonneg(form.plain.get(monomial, {}), form.log.get(monomial, {}), form.rate)
        if result is False:
            return False
        if result is None:
            verdict = None
    return verdict


def log_form_mass(expr: sympy.Expr, variables: Sequence[str]) -> Optional[Tuple[bool, Optional[sympy.Expr]]]:
    """``(finite, mass)`` of a function ``P + Q·log(1 - c·k)`` with non-negative coefficients (module docstring), or
    None for other forms."""
    form = log_form(expr, variables)
    if form is None:
        return None
    plain_total: Dict[int, Fraction] = defaultdict(Fraction)
    log_total: Dict[int, Fraction] = defaultdict(Fraction)
    for target, parts in ((plain_total, form.plain), (log_total, form.log)):
        for monomial, laurent in parts.items():
            for j, value in laurent.items():
                target[j + sum(monomial)] += value  # the diagonal t = k = w
    if form.rate < 1:
        value = sympy.S(expr).subs({s: 1 for s in sympy.S(expr).free_symbols})
        return True, value
    if sum(log_total.values()) != 0:
        return False, None
    total = sum(plain_total.values(), Fraction(0))
    return True, sympy.Rational(total.numerator, total.denominator)


def is_log_form(expr: sympy.Expr, variables: Sequence[str]) -> bool:
    """Whether ``expr`` is ``P + Q·log(1 - c·k)`` (module docstring): its series converges on the open unit
    polydisc."""
    return log_form(expr, variables) is not None


__all__: List[str] = ["LogForm", "log_form", "log_form_nonneg", "log_form_mass", "is_log_form"]
