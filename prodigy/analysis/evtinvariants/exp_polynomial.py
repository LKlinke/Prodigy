"""
Exponential polynomials in one counter: integration from 0, analyticity and non-negativity (#323).

A probabilistic choice whose probability is rational in a counter ``k`` acts on a generating function F as an
operator (:mod:`prodigy.analysis.instructionhandler.counter_operators`): ``1/(k + a)`` maps the coefficient
``f_n`` of ``k^n`` to ``f_n/(n + a)``, i.e. F to ``k^(-a) ∫_0^k t^(a-1) F(t) dt``. Generating functions with
such choices are no longer rational; the simplest closed forms are *exponential polynomials* in ``k``:
finite sums ``Σ r_(n,c)(w) · k^n · exp(c·k)`` with numbers ``c`` and coefficients ``r_(n,c)`` that do not depend
on ``k`` (``w`` are the other variables and the parameters). The Poisson generating function ``exp(λ(k - 1))``
is one. This module provides what the operator semantics and the EVT certificates need for them.

**Integration from 0: the integration proposes, the derivative certifies.** :func:`integral_from_zero` writes the
integrand as an exponential polynomial and integrates each term ``k^n exp(c k)`` by a closed formula (repeated
integration by parts); for other integrands it asks SymPy's ``integrate`` (e.g. a logarithm for a rational
integrand). The antiderivative is only a proposal: the result G is accepted only if the engine-independent checks
prove

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

from prodigy.distribution.equality import decide_equal, sympify_symbolic_string

MAX_DIRECT_COEFFICIENTS = 10_000
"""How many coefficients :func:`exp_polynomial_nonneg` checks one by one before it gives up (``None``)."""

Terms = Dict[Tuple[int, sympy.Expr], sympy.Expr]
"""An exponential polynomial in one variable k: ``{(n, c): r}`` stands for ``Σ r · k^n · exp(c·k)``."""


def _affine_exponent(atom: sympy.Expr, k: sympy.Symbol) -> Optional[Tuple[sympy.Rational, sympy.Expr]]:
    """``(c, d)`` if ``atom`` is ``exp(c·k + d)`` with a rational c and d free of k, else None."""
    try:
        poly = sympy.Poly(sympy.expand(atom.args[0]), k)
    except sympy.PolynomialError:
        return None
    if poly.degree() > 1:
        return None
    coefficients = poly.all_coeffs()
    rate, offset = (coefficients[0], coefficients[1]) if len(coefficients) == 2 else (sympy.Integer(0), coefficients[0])
    if not sympify_symbolic_string(rate).is_Rational:
        return None
    return sympy.Rational(rate), offset


def exp_polynomial_terms(expr: sympy.Expr, k: sympy.Symbol) -> Optional[Terms]:
    """The terms of ``expr`` as an exponential polynomial in ``k`` (see :data:`Terms`), or None if it is none.

    The coefficients may be any expressions free of k (rational functions of the other variables, parameters).
    Terms with the same ``(n, c)`` are combined, so the decomposition is unique: the functions ``k^n exp(c k)``
    are linearly independent over the functions of the other variables. Computed by polynomial arithmetic: with
    ``z = exp(k/L)`` for a common denominator L of the rates, ``expr`` must be ``N(k, z)/(z^j·D)`` with a polynomial
    N and D free of k and z."""
    expr = sympify_symbolic_string(expr)
    exponents = {}
    for atom in expr.atoms(sympy.exp):
        if atom.has(k):
            affine = _affine_exponent(atom, k)
            if affine is None:
                return None
            exponents[atom] = affine
    common = math.lcm(1, *(int(rate.q) for rate, _ in exponents.values()))
    z = sympy.Dummy("z")
    substituted = expr.xreplace({atom: sympy.exp(offset) * z ** int(rate * common)
                                 for atom, (rate, offset) in exponents.items()})
    numerator, denominator = sympy.fraction(sympy.together(substituted))
    constant, dependent = denominator.as_independent(k, z, as_Add=False)
    shift = 0
    for factor in sympy.Mul.make_args(dependent):
        if factor == 1:
            continue
        if factor == z:
            shift += 1
        elif isinstance(factor, sympy.Pow) and factor.base == z and factor.exp.is_Integer and factor.exp > 0:
            shift += int(factor.exp)
        else:
            return None
    try:
        poly = sympy.Poly(sympy.expand(numerator), k, z)
    except sympy.PolynomialError:
        return None
    if any(c.has(k) or c.has(z) for c in poly.coeffs()):
        return None
    terms: Terms = defaultdict(lambda: sympy.Integer(0))
    for (n, m), coefficient in poly.terms():
        key = (int(n), sympy.Rational(int(m) - shift, common))
        terms[key] = terms[key] + coefficient / constant
    return {key: value for key, value in terms.items() if sympy.cancel(value) != 0}


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

    return check(sympify_symbolic_string(expr))


def _cas_antiderivative(integrand: sympy.Expr, k: sympy.Symbol,
                        variables: Collection[sympy.Symbol]) -> Optional[sympy.Expr]:
    """An antiderivative of ``integrand`` in k from SymPy's ``integrate`` (a proposal), or None if SymPy finds none.
    Only for integrands without parameters (symbols outside ``variables``): with the parameters of a template,
    ``integrate`` can take very long."""
    names = {v.name for v in variables}
    if any(s.name not in names for s in integrand.free_symbols):
        return None
    try:
        result = sympy.integrate(integrand, k)
    except Exception:  # pylint: disable=broad-except
        return None
    if result.has(sympy.Integral):
        return None
    # SymPy writes ∫ 1/(2 - k) as -log(k - 2), whose principal branch has its cut at the origin; log(-a) differs
    # from log(a) by a constant, so it is an antiderivative as well, and analytic where -a > 0
    origin = {v: 0 for v in variables}

    def flipped(argument: sympy.Expr) -> sympy.Expr:
        value = argument.subs(origin)
        return sympy.log(-argument) if value.is_number and value.is_negative is True else sympy.log(argument)

    return result.replace(sympy.log, flipped)


def integral_from_zero(integrand: sympy.Expr, k: sympy.Symbol,
                       variables: Collection[sympy.Symbol]) -> Optional[sympy.Expr]:
    """The power series ``∫_0^k J(t) dt`` of ``J = integrand`` as a closed form G, or None if no antiderivative is
    found or the result is not certified (module docstring: ``∂G/∂k = J``, ``G(0) = 0``, G analytic at the origin of
    ``variables``, which must contain k).

    The antiderivative g is proposed by the formula for exponential polynomials, otherwise by SymPy's ``integrate``
    (e.g. a logarithm for a rational integrand). Its constant is fixed as in the module docstring: G = g - g(0)."""
    terms = exp_polynomial_terms(integrand, k)
    if terms is not None:
        antiderivative: Optional[sympy.Expr] = sympy.Add(
            *(coefficient * _integral_of_term(n, c, k) for (n, c), coefficient in terms.items()))
    else:
        antiderivative = _cas_antiderivative(integrand, k, variables)
    if antiderivative is None:
        return None
    # the integration constant: the integral from 0 is the antiderivative with value 0 at k = 0
    at_zero = antiderivative.subs(k, 0)
    if at_zero.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return None
    proposal = antiderivative - at_zero
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
    expr = sympify_symbolic_string(expr)
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
            if not sympify_symbolic_string(value).is_Rational:
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
    return eventually_nonneg([Fraction(str(x)) for x in r.all_coeffs()], head)


def eventually_nonneg(coefficients: List[Fraction], head: int, strict: bool = False) -> Optional[bool]:
    """Decides ``r(m) ≥ 0`` (``r(m) > 0`` if ``strict``) for all integers ``m > head``, for the polynomial r with
    the ``coefficients`` (leading
    first): False if r is negative at some such m, None if that needs more than :data:`MAX_DIRECT_COEFFICIENTS`
    evaluations. Beyond Cauchy's root bound ``1 + max |r_i/r_lead|``, r has the sign of its leading coefficient; below
    it, r is evaluated."""
    while coefficients and coefficients[0] == 0:
        coefficients = coefficients[1:]
    if not coefficients:
        return not strict
    if coefficients[0] < 0:
        return False  # r(m) < 0 for all large m
    bound = 1 + max((abs(x / coefficients[0]) for x in coefficients[1:]), default=Fraction(0))
    last = max(head, math.ceil(bound))
    if last - head > MAX_DIRECT_COEFFICIENTS:
        return None
    for m in range(head + 1, last + 1):
        value = sum(x * m ** (len(coefficients) - 1 - i) for i, x in enumerate(coefficients))
        if value < 0 or (strict and value == 0):
            return False
    return True


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


def independent_factors(expr: sympy.Expr, variables: Sequence[str]) -> Optional[List[sympy.Expr]]:
    """``expr`` as a product of at least two factors over pairwise disjoint sets of ``variables`` (e.g.
    ``exp(k/2)·ok`` times ``1/(2 - j)``), or None. Constants go with the first factor. The coefficients of such a
    product are products of coefficients of the factors, so it has non-negative coefficients if all factors have;
    it converges where all factors converge, and its mass is the product of their masses."""
    expr = sympify_symbolic_string(expr)
    if not isinstance(expr, sympy.Mul):
        return None
    names = set(variables)
    groups: List[Tuple[set, List[sympy.Expr]]] = []
    constants: List[sympy.Expr] = []
    for factor in expr.args:
        used = {s.name for s in factor.free_symbols} & names
        if not used:
            constants.append(factor)
            continue
        merged = [g for g in groups if g[0] & used]
        rest = [g for g in groups if not g[0] & used]
        group_names = set(used).union(*(g[0] for g in merged))
        group_factors = [f for g in merged for f in g[1]] + [factor]
        groups = rest + [(group_names, group_factors)]
    if len(groups) < 2:
        return None
    factors = [sympy.Mul(*g[1]) for g in groups]
    # a non-negative series that is not 0 at the origin is positive there: a factor negative at the origin is
    # negated, and its sign goes to the constant (GiNaC writes 1/(2 - j) as -(-2 + j)^(-1))
    # The constant goes to a factor that is 0 at the origin (whose sign is not fixed that way), if there is one.
    origin = {s: 0 for s in expr.free_symbols if s.name in names}
    target = 0
    for i, factor in enumerate(factors):
        value = factor.subs(origin)
        if value.is_number and value.is_negative is True:
            factors[i] = -factor
            constants.append(sympy.Integer(-1))
        elif value.is_number and value.is_zero is True:
            target = i
    factors[target] = sympy.Mul(*constants) * factors[target]
    return factors


def exp_polynomial_value_at_one(expr: sympy.Expr, variables: Sequence[str]) -> sympy.Expr:
    """``expr`` with every variable set to 1 (its mass, if it is an entire function with non-negative coefficients)."""
    expr = sympify_symbolic_string(expr)
    names = set(variables)
    return expr.subs({s: 1 for s in expr.free_symbols if s.name in names})


__all__: List[str] = ["eventually_nonneg", "exp_polynomial_terms", "integral_from_zero", "analytic_at_zero",
                      "entire_exp_polynomial", "exp_polynomial_nonneg", "exp_polynomial_value_at_one",
                      "independent_factors"]
