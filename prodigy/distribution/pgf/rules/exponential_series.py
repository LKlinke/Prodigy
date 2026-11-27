"""
Taylor coefficients of exponentials of polynomials, by a linear recurrence (Prodigy#455).

The class: finite sums of terms ``R·exp(g)`` with ``R`` a rational function and ``g`` a polynomial in the
expanded variables, where no denominator of ``R`` vanishes at 0. Compound Poisson distributions belong to it:
``x = poisson(λ)`` followed by ``y = iid(D, x)`` has the generating function ``exp(λ·(x·P(y) − 1))`` with the
PGF ``P`` of ``D``, a polynomial in x for every ``D`` and in y for a finite ``D``; so do Poisson variables
(``exp(λ·(x − 1))``), their products with polynomials and rational functions (independent finite or geometric
variables), their marginals and their sums. Other symbols (variables that are not expanded, parameters) are
constants of the expansion: the coefficients of ``R`` and ``g`` are rational functions in them, and
subexpressions without an expanded variable that are not rational (``exp(-2)``, ``sqrt(p)``) are replaced by
fresh symbols as in :mod:`prodigy.distribution.pgf.rules.multisection`.

Justification. Write ``g = g_0 + ĝ`` with ``g_0 = g|_{x=0}`` and ``E = exp(ĝ) = Σ e_n x^n``. From ``E' = ĝ'·E``,
comparing the coefficients of x^(n-1):

    e_0 = 1,    n·e_n = Σ_{k=1}^{min(n, d)} k·g_k·e_{n-k}        (d = deg_x g)

With ``R = P/Q`` (polynomials in x, ``Q(0) ≠ 0``), the series ``S = R·E`` satisfies ``Q·S = P·E``, so

    s_n = (Σ_{j=0}^{min(n, deg P)} p_j·e_{n-j} − Σ_{k=1}^{min(n, deg Q)} q_k·s_{n-k}) / q_0

and the coefficient of x^n of ``R·exp(g)`` is ``s_n·exp(g_0)``. Each coefficient costs at most
``d + deg P + deg Q`` products of exact rational functions in the other symbols (the FLINT kernel of
:mod:`prodigy.distribution.kernel.rational`), instead of a symbolic series expansion whose cost grows much faster
with the order. The class is closed under taking a coefficient: ``s_n·exp(g_0)`` is again a term of the class in
the remaining variables (``s_n`` is rational, ``g_0`` a polynomial, and the denominators of ``s_n`` divide powers
of ``q_0`` and of the denominators of the ``g_k``, which do not vanish at 0 when ``Q(0) ≠ 0`` and the
denominator of ``g`` contains no expanded variable). So the coefficients of a joint distribution are computed by
the same recurrence in x and then in y, without leaving the kernel (:func:`exp_polynomial_columns`). Everything
is exact: the recurrences are identities of formal power series over the field of rational functions in the
other symbols, and the replaced subexpressions are constants of the expanded variables.

Not in the class (the caller falls back to the series expansion): an exponent that is not a polynomial
(``exp(1/(1-x))``), a non-integer power of an exponential, a term whose rational factor has a pole at 0 (even
if the sum has none, e.g. ``(exp(x) - 1)/x``), and expressions the kernel cannot represent. Constants without an
expanded variable that are not rational (``exp(-2)``, floats) are replaced by fresh symbols, as in
:mod:`prodigy.distribution.pgf.rules.rational_series`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import sympy

from prodigy.distribution.kernel.rational import NotRepresentableError, RationalGF
from prodigy.distribution.pgf.rules.multisection import _to_rational
from prodigy.distribution.pgf.rules.rational_series import _normalized

KernelTerm = Tuple[RationalGF, RationalGF]
"""A term ``R·exp(g)`` of the class in the kernel, as ``(R, g)``."""
Window = Tuple[int, int]
"""An inclusive range ``(low, high)`` of exponents."""


@dataclass(frozen=True)
class _Prepared:
    """A function of the class in the kernel: its terms and how to translate kernel values back to SymPy."""
    terms: List[KernelTerm]
    symbols: Dict[str, sympy.Symbol]
    """The SymPy symbol of each ring variable (fresh symbols for the replaced subexpressions)."""
    back: Dict[sympy.Symbol, sympy.Expr]
    """The replaced subexpression behind each fresh symbol."""
    expanded: Tuple[sympy.Symbol, ...]

    def value(self, parts: Sequence[KernelTerm]) -> sympy.Expr:
        """Σ R·exp(g) of terms without the expanded variables, as a SymPy expression in the original symbols
        (each R normalised as the coefficients of :mod:`prodigy.distribution.pgf.rules.rational_series`)."""
        others = sorted((s for s in self.symbols.values() if s not in self.expanded), key=sympy.default_sort_key)
        total: sympy.Expr = sympy.Integer(0)
        for rational, exponent in parts:
            value = _normalized(rational.to_sympy(self.symbols), others)
            constant = exponent.to_sympy(self.symbols)
            if self.back:
                value, constant = value.xreplace(self.back), constant.xreplace(self.back)
            if constant != 0:
                value = sympy.exp(constant) * value
                # distributed over a polynomial, like the terms of a series expansion
                if rational.is_polynomial():
                    value = sympy.expand_mul(value)
            total += value
        return total


def exp_polynomial_taylor_stream(function: sympy.Expr, variable: sympy.Symbol) -> Optional[Iterator[sympy.Expr]]:
    """
    The coefficients of x^0, x^1, ... (x = ``variable``) of the power series of ``function`` around 0 by the
    recurrences of the module docstring, one at a time, or None if ``function`` is not in the class or no
    exponent depends on x (a rational function: :mod:`prodigy.distribution.pgf.rules.rational_series`). The iterator
    does not end. Other symbols are constants.
    """
    prepared = _prepare(function, (variable,))
    if prepared is None or all(variable.name not in g.free_names() for _, g in prepared.terms):
        return None
    streams = [_coefficients(term, variable.name) for term in prepared.terms]
    return (prepared.value(parts) for parts in zip(*streams, strict=True))


def exp_polynomial_columns(function: sympy.Expr, first: sympy.Symbol, second: sympy.Symbol,
                           first_window: Window, second_window: Window) -> Optional[Iterator[Iterator[sympy.Expr]]]:
    """
    The coefficients of ``first^i second^j`` of ``function`` for ``i`` and ``j`` in the windows (inclusive
    ranges ``(low, high)``) as columns: the column of ``i`` yields the coefficients of ``j = low, ..., high``.
    None if ``function`` is not in the class in both variables; rational functions belong to it (``g = 0``).
    The coefficients below a window are computed in the kernel only, and a cell is translated to SymPy when
    its column is iterated.
    """
    prepared = _prepare(function, (first, second))
    if prepared is None:
        return None
    outer = zip(*(_coefficients(term, first.name) for term in prepared.terms), strict=True)
    return (_column(prepared, parts, second.name, second_window)
            for parts in itertools.islice(outer, first_window[0], first_window[1] + 1))


def _column(prepared: _Prepared, parts: Sequence[KernelTerm], name: str, window: Window) -> Iterator[sympy.Expr]:
    """The cells of a column (:func:`exp_polynomial_columns`), computed when the column is iterated."""
    inner = zip(*(_coefficients(term, name) for term in parts), strict=True)
    for cells in itertools.islice(inner, window[0], window[1] + 1):
        yield prepared.value(cells)


def _prepare(function: sympy.Expr, variables: Tuple[sympy.Symbol, ...]) -> Optional[_Prepared]:
    # pylint: disable=too-many-locals  # the intermediate values of one algorithm in the docstring
    """``function`` as terms of the class in the kernel, or None if it is not in the class (module docstring)."""
    replacements: Dict[sympy.Expr, sympy.Dummy] = {}
    rationals: List[Tuple[sympy.Expr, sympy.Expr]] = []
    summands = [term for summand in sympy.Add.make_args(sympy.sympify(function))
                for term in _exponential_terms(summand, variables)]
    for summand in summands:
        factors: List[sympy.Expr] = []
        exponent: sympy.Expr = sympy.Integer(0)
        for factor in sympy.Mul.make_args(summand):
            power = _exponent(factor, variables)
            if power is None:
                factors.append(factor)
            else:
                exponent += power
        rational = _to_rational(sympy.Mul(*factors), variables, replacements)
        polynomial = _to_rational(sympy.expand(exponent), variables, replacements)
        if rational is None or polynomial is None:
            return None
        rationals.append((rational, polynomial))
    ring = _ring(rationals, variables, replacements)
    if ring is None:
        return None
    names, symbols, rename, back = ring
    terms: List[KernelTerm] = []
    origin = {variable.name: 0 for variable in variables}
    try:
        for rational, polynomial in rationals:
            r = RationalGF.from_sympy(rational.xreplace(rename), names)
            g = RationalGF.from_sympy(polynomial.xreplace(rename), names)
            denominators = RationalGF(r.denominator, reduced=True), RationalGF(g.denominator, reduced=True)
            if any(name in denominators[1].free_names() for name in origin) \
                    or _at(denominators[0], origin).is_zero():
                return None  # g is not a polynomial, or R has a pole at 0
            terms.append((r, g))
    except NotRepresentableError:
        return None
    return _Prepared(terms, symbols, back, variables)


def _exponential_terms(summand: sympy.Expr, variables: Tuple[sympy.Symbol, ...]) -> List[sympy.Expr]:
    """
    ``summand`` as a sum of terms in which each exponential of an expanded variable is a factor, if it is not
    one already: exponentials that are common factors of a denominator (``1/(a·exp(y) + b·exp(y))``,
    ``1/(exp(y - 1) + exp(y + 1))``) are pulled out of it, and a numerator that is a sum of exponentials is
    multiplied out (``(exp(2*x) - 1)/(c·exp(x))`` gives ``exp(x)/c - exp(-x)/c``). Identities: the exponentials of
    sums are split into products (:func:`sympy.expand_power_exp`), the common factors are pulled out
    (:func:`sympy.factor_terms`), and the result is expanded into terms. The common case needs no rewriting
    (Prodigy#472).
    """
    if all(_exponent(factor, variables) is not None or not _has_exponential(factor, variables)
           for factor in sympy.Mul.make_args(summand)):
        return [summand]
    split = sympy.factor_terms(sympy.expand_power_exp(summand))
    return [sympy.factor_terms(term) for term in sympy.Add.make_args(sympy.expand(split))]


def _has_exponential(expression: sympy.Expr, variables: Tuple[sympy.Symbol, ...]) -> bool:
    return any(power.args[0].has(*variables) for power in expression.atoms(sympy.exp))


def _at(function: RationalGF, values: Dict[str, int]) -> RationalGF:
    for name, value in values.items():
        function = function.subs(name, value)
    return function


def _exponent(factor: sympy.Expr, variables: Tuple[sympy.Symbol, ...]) -> Optional[sympy.Expr]:
    """``a`` for a factor ``exp(a)`` (``n·a`` for ``exp(a)**n`` with an integer n) whose exponent contains an
    expanded variable, else None."""
    if isinstance(factor, sympy.exp) and factor.args[0].has(*variables):
        return factor.args[0]
    if (isinstance(factor, sympy.Pow) and isinstance(factor.base, sympy.exp) and factor.exp.is_Integer
            and factor.base.args[0].has(*variables)):
        return factor.exp * factor.base.args[0]
    return None


def _ring(rationals: Sequence[Tuple[sympy.Expr, sympy.Expr]], variables: Tuple[sympy.Symbol, ...],
          replacements: Dict[sympy.Expr, sympy.Dummy]) -> Optional[Tuple[
              List[str], Dict[str, sympy.Symbol], Dict[sympy.Symbol, sympy.Symbol], Dict[sympy.Symbol, sympy.Expr]]]:
    """
    The ring of the kernel (the expanded variables, the other symbols and a fresh name for each replacement): its
    sorted names, the SymPy symbol of each name, the fresh symbol of each replacement dummy and the subexpression
    behind each fresh symbol. None if two different symbols share a name.
    """
    free = set(variables).union(*(r.free_symbols | g.free_symbols for r, g in rationals))
    dummies = set(replacements.values())
    originals = [s for s in free if s not in dummies]
    taken = {s.name for s in originals}
    if len(taken) != len(originals):
        return None
    symbols: Dict[str, sympy.Symbol] = {s.name: s for s in originals}
    rename: Dict[sympy.Symbol, sympy.Symbol] = {}
    back: Dict[sympy.Symbol, sympy.Expr] = {}
    fresh = (f"_c{i}" for i in itertools.count())
    for original, dummy in replacements.items():
        if dummy in free:
            name = next(n for n in fresh if n not in taken)
            symbols[name] = rename[dummy] = sympy.Symbol(name)
            back[rename[dummy]] = original
    return sorted(symbols), symbols, rename, back


def _coefficients(term: KernelTerm, name: str) -> Iterator[KernelTerm]:
    """The coefficients of x^0, x^1, ... (x = ``name``) of a term ``R·exp(g)`` as terms ``(s_n, g_0)`` of the
    remaining variables (module docstring); the iterator does not end."""
    rational, exponent = term
    constant = exponent.subs(name, 0)
    g = _padded(exponent - constant, name)
    p = _padded(RationalGF(rational.numerator, reduced=True), name)
    q = _padded(RationalGF(rational.denominator, reduced=True), name)
    return ((s, constant) for s in _recurrence(g, p, q, RationalGF.constant(0, rational.names)))


def _padded(polynomial: RationalGF, name: str) -> List[RationalGF]:
    """The coefficients of a polynomial in ``name``, at least one."""
    return list(polynomial.coefficients(name)) or [RationalGF.constant(0, polynomial.names)]


def _recurrence(g: List[RationalGF], p: List[RationalGF], q: List[RationalGF],
                zero: RationalGF) -> Iterator[RationalGF]:
    # pylint: disable=too-many-locals  # the intermediate values of one algorithm in the docstring
    """s_n of the module docstring for n = 0, 1, ...: ``g``, ``p`` and ``q`` are the coefficients of ĝ, P and Q."""
    weighted = [(k, g[k] * k) for k in range(1, len(g)) if not g[k].is_zero()]
    numerator = [(j, pj) for j, pj in enumerate(p) if not pj.is_zero()]
    denominator = [(k, q[k]) for k in range(1, len(q)) if not q[k].is_zero()]
    inverse_q0 = q[0].inverse()
    e: List[RationalGF] = []
    s: List[RationalGF] = []
    for n in itertools.count():
        if n == 0:
            e.append(zero + 1)
        else:
            value = zero
            for k, weight in weighted:
                if k > n:
                    break
                value = value + weight * e[n - k]
            e.append(value / n)
        h = zero
        if not weighted:  # E = 1: the rational recursion of P/Q alone
            h = p[n] if n < len(p) else zero
        for j, pj in numerator if weighted else ():
            if j > n:
                break
            h = h + pj * e[n - j]
        for k, qk in denominator:
            if k > n:
                break
            h = h - qk * s[n - k]
        s.append(h * inverse_q0)
        yield s[n]
