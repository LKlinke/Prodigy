"""
A rational kernel on FLINT: rational functions over ℚ in the program variables and parameters (Prodigy#417).

:class:`RationalGF` is a generating function P/Q with P and Q multivariate polynomials over ℚ in FLINT's
``fmpq_mpoly`` (python-flint does not expose FLINT's ``fmpz_mpoly_q``, so the pair is stored and cancelled
by a GCD here). Program variables and parameters are both ring variables; which names are variables is the
caller's business. The operations are the CAS operations of the forward analysis on rational generating
functions: addition, multiplication, division, integer powers, substitution of a variable by a rational
function, cancellation and the Taylor coefficients in one variable (by the linear recursion of the
denominator, as in :mod:`prodigy.distribution.rational_series`), plus the conversion to and from SymPy.

The kernel is exact or refuses: an input that is not a rational function over ℚ (``exp``, ``log``,
``sqrt``, floats, ``pi``, ``I``, symbolic exponents) raises :class:`NotRepresentableError`, which tells the
caller to stay with SymPy. It has no case analysis (guards, support splits, zero mass); that stays in the
shared ``Distribution`` logic, which may use the kernel as a hook. Nothing in the analysis uses it yet.

Canonical form: every :class:`RationalGF` is reduced, i.e. gcd(P, Q) = 1 and the leading coefficient of Q
(in the lexicographic order of the sorted ring variables) is 1; the zero function is 0/1. Two reduced pairs
of the same function are equal up to this normalisation, so the representation is unique.

Measurement (``benchmarks/kernel.py``, table in ``docs/source/kernel.md``): on the loop operations of the
#348 benchmark set the kernel takes 0.2-1.2 ms, SymPy's ``cancel``-based arithmetic 70 ms-1.7 s (×340-1400),
SymPy's engine 30 ms-17 s and GiNaC's engine 13-240 ms (×44-205).
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import flint
import sympy

Scalar = Union[int, Fraction, flint.fmpq]
"""A rational constant the kernel accepts as an operand."""
Operand = Union["RationalGF", Scalar]
"""An operand of the arithmetic of :class:`RationalGF`."""
Polynomial = flint.fmpq_mpoly
"""A polynomial over ℚ in the ring variables (FLINT ``fmpq_mpoly``)."""
Context = flint.fmpq_mpoly_ctx
"""A polynomial ring over ℚ with named variables (FLINT ``fmpq_mpoly_ctx``)."""

ORDERING = "lex"
"""The monomial order of every ring; with sorted variable names, a ring's order restricts to a subring's."""


class NotRepresentableError(ValueError):
    """The expression is not a rational function over ℚ; the caller falls back to SymPy."""


class PoleAtZeroError(ValueError):
    """The function has a pole at 0 in the expanded variable, so it has no power series there."""


def ring(names: Iterable[str]) -> Context:
    """The polynomial ring over ℚ in the sorted, distinct `names` (FLINT caches the contexts)."""
    return flint.fmpq_mpoly_ctx.get(tuple(sorted(set(names))), ORDERING)


def _fmpq(value: Scalar) -> flint.fmpq:
    """`value` as a FLINT rational."""
    if isinstance(value, Fraction):
        return flint.fmpq(value.numerator, value.denominator)
    return flint.fmpq(value)


def _fraction(value: flint.fmpq) -> Fraction:
    """A FLINT rational as a :class:`~fractions.Fraction`."""
    return Fraction(int(value.p), int(value.q))


def _lift(poly: Polynomial, context: Context) -> Polynomial:
    """`poly` in `context`, which contains all variables of the ring of `poly` (mapped by name)."""
    return poly if poly.context() is context else poly.project_to_context(context)


def _split(poly: Polynomial, index: int) -> Dict[int, Polynomial]:
    """The coefficients of `poly` as a polynomial in the variable with index `index`: degree -> coefficient."""
    context = poly.context()
    parts: Dict[int, Dict[Tuple[int, ...], flint.fmpq]] = {}
    for exponents, coefficient in poly.to_dict().items():
        degree = int(exponents[index])
        rest = tuple(0 if i == index else int(e) for i, e in enumerate(exponents))
        parts.setdefault(degree, {})[rest] = coefficient
    return {degree: context.from_dict(terms) for degree, terms in parts.items()}


def _homogenized(poly: Polynomial, index: int, numerator: Polynomial, denominator: Polynomial) -> Tuple[
        Polynomial, int]:
    """
    (F, d) with poly(x = numerator/denominator) = F / denominator^d, x the variable with index `index` and d its
    degree in `poly`: F = sum_i c_i numerator^i denominator^(d-i) for poly = sum_i c_i x^i, by Horner's scheme.
    """
    if poly.is_zero():
        return poly, 0
    parts = _split(poly, index)
    degree = max(parts)
    zero = poly.context().constant(0)
    result = parts.get(degree, zero)
    for i in range(degree - 1, -1, -1):
        result = result * numerator + parts.get(i, zero) * denominator ** (degree - i)
    return result, degree


class RationalGF:
    """
    A rational function P/Q over ℚ in named ring variables, always reduced (see the module docstring).

    >>> f = RationalGF.from_sympy(sympy.sympify("(1 - p) / (1 - p*x)"))
    >>> [str(c) for c in f.coefficient_list("x", 3)]
    ['-p + 1', '-p^2 + p', '-p^3 + p^2']
    >>> f.subs("x", 1).constant_value()
    Fraction(1, 1)
    """
    __slots__ = ("numerator", "denominator")
    __hash__ = None  # type: ignore[assignment]  # mutable FLINT polynomials, compared by cross multiplication

    numerator: Polynomial
    denominator: Polynomial

    def __init__(self, numerator: Polynomial, denominator: Optional[Polynomial] = None, *,
                 reduced: bool = False) -> None:
        """P/Q from two polynomials of the same ring (Q = 1 by default); `reduced` skips the cancellation."""
        if denominator is None:
            denominator = numerator.context().constant(1)
        if numerator.context() is not denominator.context():
            context = ring(numerator.context().names() + denominator.context().names())
            numerator, denominator = _lift(numerator, context), _lift(denominator, context)
        if denominator.is_zero():
            raise ZeroDivisionError("The denominator of a rational function is 0.")
        if not reduced:
            numerator, denominator = _cancelled(numerator, denominator)
        self.numerator = numerator
        self.denominator = denominator

    # --- construction and conversion -----------------------------------------------------------------------

    @classmethod
    def constant(cls, value: Scalar, names: Iterable[str] = ()) -> RationalGF:
        """The constant `value` in the ring of `names`."""
        return cls(ring(names).constant(_fmpq(value)), reduced=True)

    @classmethod
    def variable(cls, name: str, names: Iterable[str] = ()) -> RationalGF:
        """The ring variable `name` in the ring of `names` and `name`."""
        context = ring(itertools.chain(names, [name]))
        return cls(context.gen(context.variable_to_index(name)), reduced=True)

    @classmethod
    def from_sympy(cls, expression: sympy.Expr, names: Iterable[str] = ()) -> RationalGF:
        """
        The rational function `expression` (a SymPy expression) in the ring of its free symbols and `names`.
        Raises :class:`NotRepresentableError` if `expression` is not a rational function over ℚ, e.g. if it
        contains ``exp``, ``log``, a root, a float, an irrational constant or a symbolic exponent.
        """
        expression = sympy.sympify(expression)
        symbols = expression.free_symbols
        symbol_names = [symbol.name for symbol in symbols]
        if len(set(symbol_names)) != len(symbol_names):
            raise NotRepresentableError(f"Two different symbols of {expression} have the same name.")
        context = ring(itertools.chain(names, symbol_names))
        return _from_sympy(expression, context, {})

    def to_sympy(self, symbols: Optional[Mapping[str, sympy.Symbol]] = None) -> sympy.Expr:
        """
        The function as a SymPy expression P/Q with expanded P and Q; `symbols` gives the SymPy symbol of a
        ring variable (by default a plain ``sympy.Symbol`` of its name).
        """
        names = self.names
        gens = [symbols[name] if symbols is not None and name in symbols else sympy.Symbol(name) for name in names]
        numerator = _poly_to_sympy(self.numerator, gens)
        if self.denominator.is_one():
            return numerator
        return numerator / _poly_to_sympy(self.denominator, gens)

    # --- inspection -------------------------------------------------------------------------------------------

    @property
    def context(self) -> Context:
        """The ring of numerator and denominator."""
        return self.numerator.context()

    @property
    def names(self) -> Tuple[str, ...]:
        """The names of the ring variables, sorted."""
        return tuple(self.context.names())

    def free_names(self) -> Tuple[str, ...]:
        """The names of the ring variables that occur in the function, sorted."""
        degrees = [max(n, d) for n, d in zip(self.numerator.degrees(), self.denominator.degrees())]
        return tuple(name for name, degree in zip(self.names, degrees) if degree > 0)

    def is_zero(self) -> bool:
        """Whether the function is 0."""
        return bool(self.numerator.is_zero())

    def is_polynomial(self) -> bool:
        """Whether the function is a polynomial (its reduced denominator is 1)."""
        return bool(self.denominator.is_one())

    def constant_value(self) -> Optional[Fraction]:
        """The value of a constant function, else None."""
        if not (self.numerator.is_constant() and self.denominator.is_constant()):
            return None
        if self.numerator.is_zero():
            return Fraction(0)
        return _fraction(self.numerator.leading_coefficient()) / _fraction(self.denominator.leading_coefficient())

    def in_ring(self, names: Iterable[str]) -> RationalGF:
        """The same function in the ring of its own variables and `names`."""
        context = ring(itertools.chain(self.names, names))
        if context is self.context:
            return self
        return RationalGF(_lift(self.numerator, context), _lift(self.denominator, context), reduced=True)

    # --- arithmetic -------------------------------------------------------------------------------------------

    def _coerced(self, other: Operand) -> Tuple[RationalGF, RationalGF]:
        """`self` and `other` in a common ring."""
        if not isinstance(other, RationalGF):
            return self, RationalGF(self.context.constant(_fmpq(other)), reduced=True)
        if other.context is self.context:
            return self, other
        names = self.names + other.names
        return self.in_ring(names), other.in_ring(names)

    def __add__(self, other: Operand) -> RationalGF:
        left, right = self._coerced(other)
        a, b, c, d = left.numerator, left.denominator, right.numerator, right.denominator
        if b == d:  # also the polynomial case b = d = 1, which needs no cancellation
            return RationalGF(a + c, b, reduced=b.is_one())
        g = b.gcd(d)
        b1, d1 = b / g, d / g
        # gcd(a d1 + c b1, b1 d1) = 1 for reduced inputs, so only the common factor g can cancel
        numerator = a * d1 + c * b1
        h = numerator.gcd(g)
        if not h.is_one():
            numerator, g = numerator / h, g / h
        return RationalGF(*_normalized(numerator, b1 * d1 * g), reduced=True)

    __radd__ = __add__

    def __neg__(self) -> RationalGF:
        return RationalGF(-self.numerator, self.denominator, reduced=True)

    def __sub__(self, other: Operand) -> RationalGF:
        return self + (-other if isinstance(other, RationalGF) else -_fmpq(other))

    def __rsub__(self, other: Operand) -> RationalGF:
        return -self + other

    def __mul__(self, other: Operand) -> RationalGF:
        left, right = self._coerced(other)
        a, b, c, d = left.numerator, left.denominator, right.numerator, right.denominator
        if b.is_one() and d.is_one():
            return RationalGF(a * c, b, reduced=True)
        # cross cancellation (Henrici): a/b and c/d are reduced, so gcd(a, d) and gcd(c, b) are all that cancels
        g1, g2 = a.gcd(d), c.gcd(b)
        return RationalGF(*_normalized((a / g1) * (c / g2), (b / g2) * (d / g1)), reduced=True)

    __rmul__ = __mul__

    def inverse(self) -> RationalGF:
        """1/f; raises ZeroDivisionError for f = 0."""
        if self.numerator.is_zero():
            raise ZeroDivisionError("The inverse of the zero function.")
        return RationalGF(*_normalized(self.denominator, self.numerator), reduced=True)

    def __truediv__(self, other: Operand) -> RationalGF:
        if isinstance(other, RationalGF):
            return self * other.inverse()
        value = _fmpq(other)
        if value == 0:
            raise ZeroDivisionError("Division of a rational function by 0.")
        return RationalGF(self.numerator / value, self.denominator, reduced=True)

    def __rtruediv__(self, other: Operand) -> RationalGF:
        return self.inverse() * other

    def __pow__(self, exponent: int) -> RationalGF:
        if not isinstance(exponent, int):
            raise NotRepresentableError(f"The exponent {exponent} is not an integer.")
        base = self if exponent >= 0 else self.inverse()
        n = abs(exponent)
        return RationalGF(*_normalized(base.numerator ** n, base.denominator ** n), reduced=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (RationalGF, int, Fraction, flint.fmpq)):
            return NotImplemented
        left, right = self._coerced(other)
        return bool(left.numerator * right.denominator == right.numerator * left.denominator)

    # --- substitution and coefficients ------------------------------------------------------------------------

    def subs(self, name: str, value: Operand) -> RationalGF:
        """
        The function with the ring variable `name` replaced by `value` (a constant or a rational function);
        raises ZeroDivisionError if the result has a denominator 0. A name that is not a ring variable leaves
        the function unchanged (it does not occur in it).
        """
        if name not in self.names:
            return self
        left, right = self._coerced(value)
        context = left.context
        index = context.variable_to_index(name)
        if right.is_polynomial():
            gens = list(context.gens())
            gens[index] = right.numerator
            return RationalGF(left.numerator.compose(*gens, ctx=context),
                              left.denominator.compose(*gens, ctx=context))
        p, q = right.numerator, right.denominator
        numerator, n_degree = _homogenized(left.numerator, index, p, q)
        denominator, d_degree = _homogenized(left.denominator, index, p, q)
        # f(p/q) = (N / q^n) / (D / q^d)
        if n_degree >= d_degree:
            denominator = denominator * q ** (n_degree - d_degree)
        else:
            numerator = numerator * q ** (d_degree - n_degree)
        return RationalGF(numerator, denominator)

    def coefficients(self, name: str) -> Iterator[RationalGF]:
        """
        The Taylor coefficients of x^0, x^1, ... of the function around x = 0 for x = `name`, as functions of
        the other ring variables (in the same ring). They satisfy the recursion of the denominator
        Q = sum_k q_k x^k: s_n = (p_n - sum_{k=1}^{n} q_k s_{n-k}) / q_0 with P = sum_k p_k x^k. If the
        function is a polynomial in x, the iterator ends after its last non-zero coefficient; otherwise it
        does not end. Raises :class:`PoleAtZeroError` if q_0 = 0.
        """
        context = self.context
        if name not in self.names:
            if not self.is_zero():
                yield self
            return
        index = context.variable_to_index(name)
        p, q = _split(self.numerator, index), _split(self.denominator, index)
        if 0 not in q:
            raise PoleAtZeroError(f"The function has a pole at {name} = 0.")
        zero = context.constant(0)
        q0 = q[0]
        length = max(p) + 1 if p else 0
        if max(q) == 0:
            for n in range(length):
                yield RationalGF(p.get(n, zero), q0)
            return
        if q0.is_constant():
            yield from _constant_recursion(p, q, _fmpq(q0.leading_coefficient()), context)
            return
        # q_0 depends on other variables: the s_n are rational in them
        previous: List[RationalGF] = []
        inverse_q0 = RationalGF(context.constant(1), q0)
        others = [(k, RationalGF(qk, reduced=True)) for k, qk in sorted(q.items())[1:]]
        for n in itertools.count():
            value = RationalGF(p.get(n, zero), reduced=True)
            for k, qk in others:
                if k > n:
                    break
                value = value - previous[n - k] * qk
            coefficient = value * inverse_q0
            previous.append(coefficient)
            yield coefficient

    def coefficient_list(self, name: str, count: int) -> List[RationalGF]:
        """The first `count` Taylor coefficients in `name` (:meth:`coefficients`), padded with zeros."""
        result = list(itertools.islice(self.coefficients(name), count))
        zero = RationalGF(self.context.constant(0), reduced=True)
        return result + [zero] * (count - len(result))

    # --- text ---------------------------------------------------------------------------------------------------

    def __str__(self) -> str:
        if self.denominator.is_one():
            return str(self.numerator)
        return f"({self.numerator})/({self.denominator})"

    def __repr__(self) -> str:
        return f"RationalGF({self}, ring={self.names})"


def _constant_recursion(p: Dict[int, Polynomial], q: Dict[int, Polynomial], q0: flint.fmpq,
                        context: Context) -> Iterator[RationalGF]:
    """The recursion of :meth:`RationalGF.coefficients` for a constant q_0: every s_n is a polynomial."""
    previous: List[Polynomial] = []
    others = sorted(q.items())[1:]
    zero = context.constant(0)
    for n in itertools.count():
        value = p.get(n, zero)
        for k, qk in others:
            if k > n:
                break
            value = value - qk * previous[n - k]
        coefficient = value / q0
        previous.append(coefficient)
        yield RationalGF(coefficient, reduced=True)


def _normalized(numerator: Polynomial, denominator: Polynomial) -> Tuple[Polynomial, Polynomial]:
    """(P, Q) scaled so that the leading coefficient of Q is 1; the zero function becomes 0/1."""
    if numerator.is_zero():
        return numerator, denominator.context().constant(1)
    lead = denominator.leading_coefficient()
    if lead != 1:
        numerator, denominator = numerator / lead, denominator / lead
    return numerator, denominator


def _cancelled(numerator: Polynomial, denominator: Polynomial) -> Tuple[Polynomial, Polynomial]:
    """(P, Q) divided by gcd(P, Q) and normalised (:func:`_normalized`)."""
    if not denominator.is_constant():
        g = numerator.gcd(denominator)
        if not g.is_constant():
            numerator, denominator = numerator / g, denominator / g
    return _normalized(numerator, denominator)


def _poly_to_sympy(poly: Polynomial, gens: List[sympy.Symbol]) -> sympy.Expr:
    """A polynomial as an expanded SymPy expression in `gens` (one per ring variable)."""
    terms = []
    for exponents, coefficient in poly.to_dict().items():
        factors = [sympy.Pow(gen, int(e)) for gen, e in zip(gens, exponents) if e]
        terms.append(sympy.Mul(sympy.Rational(int(coefficient.p), int(coefficient.q)), *factors))
    return sympy.Add(*terms)


def _from_sympy(expression: sympy.Expr, context: Context, cache: Dict[sympy.Expr, RationalGF]) -> RationalGF:
    """:meth:`RationalGF.from_sympy` by recursion on the expression tree, sharing equal subtrees."""
    if expression in cache:
        return cache[expression]
    result: RationalGF
    if expression.is_Rational:
        result = RationalGF(context.constant(flint.fmpq(int(expression.p), int(expression.q))), reduced=True)
    elif expression.is_Symbol:
        result = RationalGF(context.gen(context.variable_to_index(expression.name)), reduced=True)
    elif expression.is_Add:
        result = _sum([_from_sympy(arg, context, cache) for arg in expression.args], context)
    elif expression.is_Mul:
        result = RationalGF(context.constant(1), reduced=True)
        for arg in expression.args:
            result = result * _from_sympy(arg, context, cache)
    elif expression.is_Pow and expression.exp.is_Integer:
        result = _from_sympy(expression.base, context, cache) ** int(expression.exp)
    else:
        raise NotRepresentableError(f"{expression} is not a rational function over the rationals.")
    cache[expression] = result
    return result


def _sum(summands: List[RationalGF], context: Context) -> RationalGF:
    """The sum of `summands`: the polynomial ones are added first, without any cancellation."""
    polynomial = context.constant(0)
    result: Optional[RationalGF] = None
    for summand in summands:
        if summand.is_polynomial():
            polynomial = polynomial + summand.numerator
        else:
            result = summand if result is None else result + summand
    total = RationalGF(polynomial, reduced=True)
    return total if result is None else result + total
