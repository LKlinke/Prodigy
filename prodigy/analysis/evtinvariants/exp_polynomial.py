"""
Exponential polynomials in one counter: integration from 0, analyticity and non-negativity (#323).

A probabilistic choice whose probability is rational in a counter ``k`` acts on a generating function F as an
operator (:mod:`prodigy.analysis.instructionhandler.counter_operators`): ``1/(k + a)`` maps the coefficient
``f_n`` of ``k^n`` to ``f_n/(n + a)``, i.e. F to ``k^(-a) ∫_0^k t^(a-1) F(t) dt``. Generating functions with
such choices are no longer rational; the simplest closed forms are *exponential polynomials* in ``k``:
finite sums ``Σ r_(n,c)(w) · k^n · exp(c·k)`` with numbers ``c`` and coefficients ``r_(n,c)`` that do not depend
on ``k`` (``w`` are the other variables and the parameters). The Poisson generating function ``exp(λ(k - 1))``
is one. This module provides what the operator semantics and the EVT certificates need for them.

**Integration from 0 without integrating.** :func:`integral_from_zero` writes the integrand as an exponential
polynomial and integrates each term ``k^n exp(c k)`` by a closed formula (repeated integration by parts). The
formula is only a proposal: the result G is accepted only if the engine-independent checks prove

* ``∂G/∂k = J`` (an identity of closed forms, :func:`prodigy.distribution.equality.decide_equal`),
* ``G|_(k=0) = 0`` (an identity in the other variables), and
* G is analytic at the origin *by construction* (:func:`analytic_at_zero`).

Then G is the power series ``∫_0^k J``: both are analytic at the origin, have the same derivative in k there and
agree at ``k = 0``. The normalisation is the integration constant: a relation ``g + c = ∫ f`` (with an
antiderivative g of f) holds with the integral from 0 exactly for ``c = -g(0)``, since the integral from 0
vanishes at ``k = 0``; so ``G = g - g(0)`` is the only antiderivative that can be the integral from 0, and the check
``G(0) = 0`` is the check of that choice.

**Non-negativity** of the coefficients is decided for ``A·exp(c·k) + B`` with polynomials A and B with rational
coefficients in the program variables and a rational ``c > 0`` (:func:`exp_polynomial_nonneg`): the coefficient
of ``w^α k^m`` is ``c^m/m! · r_α(m)`` plus the coefficient of B, where ``r_α(m) = Σ_j a_(α,j) c^(-j) m(m-1)…(m-j+1)``
is a polynomial in m; finitely many coefficients are checked directly, the rest by the sign of ``r_α`` beyond its
largest real root (Cauchy's bound). Such a function is entire, so its series converges everywhere and its mass is
its value at 1 (:func:`entire_exp_polynomial`).
"""
from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import sympy

from prodigy.distribution.equality import decide_equal

MAX_DIRECT_COEFFICIENTS = 10_000
"""How many coefficients :func:`exp_polynomial_nonneg` checks one by one before it gives up (``None``)."""

Terms = Dict[Tuple[int, sympy.Expr], sympy.Expr]
"""An exponential polynomial in one variable k: ``{(n, c): r}`` stands for ``Σ r · k^n · exp(c·k)``."""


def _exponent_rate(factor: sympy.Expr, k: sympy.Symbol) -> Optional[sympy.Expr]:
    """``c`` if ``factor`` is ``exp(c·k)`` with a number c, else None."""
    if not isinstance(factor, sympy.exp):
        return None
    argument = sympy.expand(factor.args[0])
    rate = sympy.simplify(argument / k)
    if not rate.is_number or rate.has(k) or sympy.expand(argument - rate * k) != 0:
        return None
    return rate


def exp_polynomial_terms(expr: sympy.Expr, k: sympy.Symbol) -> Optional[Terms]:
    """The terms of ``expr`` as an exponential polynomial in ``k`` (see :data:`Terms`), or None if it is none.

    The coefficients may be any expressions free of k (rational functions of the other variables, parameters).
    Terms with the same ``(n, c)`` are combined, so the decomposition is unique: the functions ``k^n exp(c k)``
    are linearly independent over the functions of the other variables."""
    terms: Terms = defaultdict(lambda: sympy.Integer(0))
    expanded = sympy.expand(sympy.powsimp(sympy.expand(expr), combine="exp"))
    for term in sympy.Add.make_args(expanded):
        coefficient, dependent = term.as_independent(k, as_Add=False)
        degree, rate = 0, sympy.Integer(0)
        for factor in sympy.Mul.make_args(dependent):
            if factor == 1:
                continue
            if factor == k:
                degree += 1
                continue
            if isinstance(factor, sympy.Pow) and factor.base == k:
                if not (factor.exp.is_Integer and factor.exp >= 0):
                    return None
                degree += int(factor.exp)
                continue
            if isinstance(factor, sympy.Pow) and factor.exp.is_Integer and factor.exp > 0:
                inner = _exponent_rate(factor.base, k)
                if inner is None:
                    return None
                rate += inner * factor.exp
                continue
            inner = _exponent_rate(factor, k)
            if inner is None:
                return None
            rate += inner
        key = (degree, sympy.nsimplify(rate))
        terms[key] = terms[key] + coefficient
    return {key: value for key, value in terms.items() if sympy.simplify(value) != 0}


def _integral_of_term(n: int, c: sympy.Expr, k: sympy.Symbol) -> sympy.Expr:
    """``∫_0^k t^n exp(c t) dt`` by repeated integration by parts (a proposal, checked by the caller)."""
    if c == 0:
        return k ** (n + 1) / (n + 1)
    antiderivative = sympy.exp(c * k) * sum(
        (-1) ** j * sympy.Integer(math.factorial(n) // math.factorial(n - j)) * k ** (n - j) / c ** (j + 1)
        for j in range(n + 1))
    return antiderivative - (-1) ** n * sympy.Integer(math.factorial(n)) / c ** (n + 1)


def analytic_at_zero(expr: sympy.Expr, variables: Collection[sympy.Symbol]) -> bool:
    """Sufficient criterion: ``expr`` is analytic at the origin of ``variables`` *by construction*, for every
    value of the other symbols (parameters). Built from symbols and numbers with ``+``, ``·``, powers with natural
    exponents, powers with negative integer exponents of a base that is provably non-zero at the origin, ``exp``,
    and ``log`` of an argument that is a positive number at the origin. A removable singularity (``(exp(k) - 1)/k``)
    is not recognised."""
    origin = {v: 0 for v in variables}

    def at_origin_nonzero(base: sympy.Expr) -> bool:
        value = sympy.simplify(base.subs(origin))
        return value.is_number and value.is_zero is False and value.is_finite is True

    def check(node: sympy.Expr) -> bool:
        if node.is_Symbol or node.is_Number or node.is_NumberSymbol:
            return node.is_finite is not False
        if isinstance(node, (sympy.Add, sympy.Mul)):
            return all(check(arg) for arg in node.args)
        if isinstance(node, sympy.Pow):
            if not node.exp.is_Integer:
                return node.base.is_number and node.exp.is_number and node.is_finite is True
            return check(node.base) and (node.exp >= 0 or at_origin_nonzero(node.base))
        if isinstance(node, sympy.exp):
            return check(node.args[0])
        if isinstance(node, sympy.log):
            value = sympy.simplify(node.args[0].subs(origin))
            return check(node.args[0]) and value.is_number and value.is_positive is True
        return False

    return check(sympy.S(expr))


def integral_from_zero(integrand: sympy.Expr, k: sympy.Symbol,
                       variables: Collection[sympy.Symbol]) -> Optional[sympy.Expr]:
    """The power series ``∫_0^k J(t) dt`` of ``J = integrand`` as a closed form G, or None if the integrand is no
    exponential polynomial in k or the result is not certified (module docstring: ``∂G/∂k = J``, ``G(0) = 0``, G
    analytic at the origin of ``variables``, which must contain k)."""
    terms = exp_polynomial_terms(integrand, k)
    if terms is None:
        return None
    proposal = sympy.Add(*(coefficient * _integral_of_term(n, c, k) for (n, c), coefficient in terms.items()))
    # the integration constant: the integral from 0 is the antiderivative with value 0 at k = 0
    proposal = proposal - proposal.subs(k, 0)
    if decide_equal(sympy.diff(proposal, k), integrand, use_simplify=False) is not True:
        return None
    if decide_equal(proposal.subs(k, 0), sympy.Integer(0), use_simplify=False) is not True:
        return None
    if not analytic_at_zero(proposal, variables):
        return None
    return proposal


def _polynomial_parts(expr: sympy.Expr, variables: Sequence[str]) \
        -> Optional[Tuple[sympy.Symbol, sympy.Expr, Dict[Tuple[int, ...], Dict[int, Fraction]],
                          Dict[Tuple[int, ...], Dict[int, Fraction]]]]:
    """``expr`` as ``A·exp(c·k) + B`` for one variable k, a rational ``c > 0`` and polynomials A, B with rational
    coefficients in ``variables``: (k, c, A, B), where A and B map the exponents of the other variables to the
    coefficients of the powers of k. None if ``expr`` has another form or contains parameters."""
    expr = sympy.S(expr)
    names = set(variables)
    if any(s.name not in names for s in expr.free_symbols):
        return None
    by_name = {s.name: s for s in expr.free_symbols}  # the symbols may carry assumptions
    candidates = {s for s in expr.free_symbols
                  if any(isinstance(a, sympy.exp) and a.has(s) for a in expr.atoms(sympy.exp))}
    if len(candidates) > 1:
        return None
    if not candidates and not by_name:
        return None  # a constant
    k = next(iter(candidates)) if candidates else by_name[sorted(by_name)[0]]
    terms = exp_polynomial_terms(expr, k)
    if terms is None:
        return None
    rates = {c for (_, c) in terms if c != 0}
    if len(rates) > 1:
        return None
    rate = next(iter(rates)) if rates else sympy.Integer(1)
    if not (rate.is_Rational and rate > 0):
        return None
    others = sorted((by_name.get(v, sympy.Symbol(v)) for v in names if v != k.name), key=lambda s: s.name)
    exp_part: Dict[Tuple[int, ...], Dict[int, Fraction]] = defaultdict(dict)
    plain_part: Dict[Tuple[int, ...], Dict[int, Fraction]] = defaultdict(dict)
    for (n, c), coefficient in terms.items():
        try:
            poly = sympy.Poly(coefficient, *others) if others else None
        except sympy.PolynomialError:
            return None
        monomials = poly.terms() if poly is not None else [((), coefficient)]
        for monomial, value in monomials:
            if not sympy.S(value).is_Rational:
                return None
            target = exp_part if c != 0 else plain_part
            target[monomial][n] = target[monomial].get(n, Fraction(0)) + Fraction(str(value))
    return k, rate, exp_part, plain_part


def entire_exp_polynomial(expr: sympy.Expr, variables: Sequence[str]) -> bool:
    """Whether ``expr`` is ``A·exp(c·k) + B`` with polynomials A, B (see :func:`exp_polynomial_nonneg`): an entire
    function, whose series converges everywhere."""
    return _polynomial_parts(expr, variables) is not None


def _falling(m: int, j: int) -> int:
    result = 1
    for i in range(j):
        result *= m - i
    return result


def _coefficient(a: Dict[int, Fraction], b: Dict[int, Fraction], c: Fraction, m: int) -> Fraction:
    """The coefficient of ``k^m`` in ``A(k)·exp(c k) + B(k)``."""
    value = b.get(m, Fraction(0))
    for j, a_j in a.items():
        if j <= m:
            value += a_j * c ** (m - j) / math.factorial(m - j)
    return value


def _series_nonneg(a: Dict[int, Fraction], b: Dict[int, Fraction], c: Fraction) -> Optional[bool]:
    """Decides whether all coefficients of ``A(k)·exp(c k) + B(k)`` (c > 0) are non-negative."""
    a = {j: v for j, v in a.items() if v != 0}
    b = {j: v for j, v in b.items() if v != 0}
    head = max(list(a) + list(b) + [0])
    for m in range(head + 1):
        if _coefficient(a, b, c, m) < 0:
            return False
    if not a:
        return True
    # for m > head: coefficient = c^m/m! · r(m) with r(m) = Σ_j a_j c^(-j) m(m-1)…(m-j+1)
    m_symbol = sympy.Symbol("m")
    r = sympy.Poly(sum(sympy.Rational(v.numerator, v.denominator) / sympy.Rational(c.numerator, c.denominator) ** j
                       * sympy.ff(m_symbol, j) for j, v in a.items()), m_symbol)
    coefficients = [Fraction(str(x)) for x in r.all_coeffs()]
    if coefficients[0] < 0:
        return False  # r(m) < 0 for all large m: a negative coefficient
    bound = 1 + max((abs(x / coefficients[0]) for x in coefficients[1:]), default=Fraction(0))
    last = max(head, math.ceil(bound))
    if last - head > MAX_DIRECT_COEFFICIENTS:
        return None
    for m in range(head + 1, last + 1):
        if sum(x * m ** (len(coefficients) - 1 - i) for i, x in enumerate(coefficients)) < 0:
            return False
    return True  # beyond Cauchy's root bound, r has the sign of its leading coefficient


def exp_polynomial_nonneg(expr: sympy.Expr, variables: Sequence[str]) -> Optional[bool]:
    """Decides non-negativity of all coefficients of ``A·exp(c·k) + B`` (module docstring): True if all are
    non-negative, False if one is negative (both proven), None if ``expr`` has another form or the check is too
    long."""
    parts = _polynomial_parts(expr, variables)
    if parts is None:
        return None
    _, rate, exp_part, plain_part = parts
    c = Fraction(str(rate))
    verdict: Optional[bool] = True
    for monomial in set(exp_part) | set(plain_part):
        result = _series_nonneg(exp_part.get(monomial, {}), plain_part.get(monomial, {}), c)
        if result is False:
            return False
        if result is None:
            verdict = None
    return verdict


def exp_polynomial_value_at_one(expr: sympy.Expr, variables: Sequence[str]) -> sympy.Expr:
    """``expr`` with every variable set to 1 (its mass, if it is an entire function with non-negative coefficients)."""
    expr = sympy.S(expr)
    names = set(variables)
    return expr.subs({s: 1 for s in expr.free_symbols if s.name in names})


__all__: List[str] = ["exp_polynomial_terms", "integral_from_zero", "analytic_at_zero", "entire_exp_polynomial",
                      "exp_polynomial_nonneg", "exp_polynomial_value_at_one"]
