"""
Exact filters `x < y + c` and `x = y + c` for two variables that are independent given the others.

Rational generating functions are not closed under the filter `x < y` in general: for the rational
F = (1/2) / (1 - x/4 - y/4) the part with x < y is algebraic but not rational (Prodigy#243). The
filter is rational, however, if F is a product

    F(x, y, r) = A(x, r) * B(y, r)

of a function rational in x and a function rational in y, where r stands for the other variables
and the parameters (x and y are independent given r). The coefficients a_i, b_j of A and B are then
rational functions in r, and the filtered parts are, with i the exponent of x and j that of y,

    sum_{j - i >= s} a_i b_j x^i y^j    and    sum_{j - i = s} a_i b_j x^i y^j    (s >= 0).

Write B = Q + R/D with polynomials Q, R, D in y (deg R < deg D = d, D(0) != 0). The coefficients
c_j of R/D satisfy the recurrence of D for every j >= d, so with the companion matrix C of D and
the vector v = (c_0, ..., c_{d-1}) we have c_j = e_1^T C^j v for all j >= 0, and

    sum_{j >= k} c_j y^j = e_1^T (yC)^k (I - yC)^(-1) v.

Summing over i with the weights a_i x^i, where A = N/E with polynomials N, E in x (E(0) != 0):

    sum_{j - i >= s} a_i c_j x^i y^j = e_1^T (yC)^s N(xyC) (E(xyC) (I - yC))^(-1) v,
    sum_{j - i = s}  a_i c_j x^i y^j = e_1^T (yC)^s N(xyC) E(xyC)^(-1) v.

Both are rational, with no root of D or E: the matrices have entries in the field of rational
functions in x, y and r, and E(xyC) and I - yC are invertible there (their determinants are
E(0)^d and 1 at x = y = 0). The finitely many terms of the polynomial part Q need the first
coefficients of A only. Relations with s < 0 follow by exchanging x and y, and the remaining
comparisons by complement.

All of this is exact identities of formal power series over the field of rational functions in r,
so it holds for signed measures and measures with infinite mass as well. Whether F is such a
product is decided exactly: F(x, y) F(x0, y0) = F(x, y0) F(x0, y) as rational functions for a point
(x0, y0) with F(x0, y0) != 0. Outside this class (x and y dependent given r, or F not rational in x
or y), :class:`ComparisonNotImplemented` names the reason; there is no approximation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import sympy
from sympy.polys.matrices import DomainMatrix

from prodigy.distribution.multisection import _to_rational


class ComparisonNotImplemented(NotImplementedError):
    """A comparison of two variables that cannot be filtered exactly; `reason` says why."""

    def __init__(self, condition: str, reason: str):
        super().__init__(f"The guard {condition} is not implemented on infinite generating functions: {reason}")
        self.condition = condition
        self.reason = reason


NOT_RATIONAL = "the generating function is not rational in the compared variables"
DEPENDENT = ("the compared variables both have infinite support and are dependent (the generating function is "
             "not a product of a function of one and a function of the other)")
NOT_A_DIFFERENCE = ("on infinite generating functions only comparisons of two variables of the form "
                    "x < y + c, x <= y + c, x = y + c, x > y + c and x >= y + c are implemented")

# base points (x0, y0) for the product test
_BASE_POINTS = [(0, 0), (0, sympy.Rational(1, 2)), (sympy.Rational(1, 2), 0),
                (sympy.Rational(1, 2), sympy.Rational(1, 3)), (sympy.Rational(1, 3), sympy.Rational(2, 7))]


class _Factors:
    """F = A(x) * B(y) over the field `domain` of rational functions in x, y and the other symbols."""

    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    def __init__(self, function: sympy.Expr, x: sympy.Symbol, y: sympy.Symbol, condition: str):
        replacements: Dict[sympy.Expr, sympy.Dummy] = {}
        rational = _to_rational(sympy.sympify(function), x, replacements)
        if rational is None or any(original.has(y) for original in replacements):
            raise ComparisonNotImplemented(condition, NOT_RATIONAL)
        self.back = {dummy: original for original, dummy in replacements.items()}
        others = sorted(rational.free_symbols - {x, y}, key=sympy.default_sort_key)
        self.x, self.y = x, y
        self.coefficients = sympy.QQ.frac_field(*others) if others else sympy.QQ
        self.domain = sympy.QQ.frac_field(x, y, *others)
        try:
            whole = self.domain.from_sympy(rational)
        except sympy.CoercionFailed as error:  # pragma: no cover - only symbols and rationals are left
            raise ComparisonNotImplemented(condition, NOT_RATIONAL) from error
        if self._value_at(whole, 0, 0) is None:
            raise ComparisonNotImplemented(condition, "the generating function has a pole at 0")
        self.whole = whole
        for x0, y0 in _BASE_POINTS:
            base = self._value_at(whole, x0, y0)
            if base is None or not base:
                continue
            left = self._value_at(whole, None, y0)
            right = self._value_at(whole, x0, None)
            assert left is not None and right is not None
            if whole * base != left * right:
                raise ComparisonNotImplemented(condition, DEPENDENT)
            # A(x) = F(x, y0), B(y) = F(x0, y) / F(x0, y0)
            self.left = self._polynomials(left, x, condition)
            self.right = self._polynomials(self.domain.quo(right, base), y, condition)
            return
        raise ComparisonNotImplemented(condition, DEPENDENT)  # pragma: no cover - F vanishes at all base points

    def _value_at(self, element, x0, y0):
        """F(x0, y0) (None keeps the variable), or None if a denominator vanishes there."""
        numerator, denominator = self.domain.numer(element), self.domain.denom(element)
        substitution = {}
        if x0 is not None:
            substitution[self.x] = x0
        if y0 is not None:
            substitution[self.y] = y0
        ring = self.domain.get_ring()
        value_numerator = ring.from_sympy(ring.to_sympy(numerator).subs(substitution))
        value_denominator = ring.from_sympy(ring.to_sympy(denominator).subs(substitution))
        if not value_denominator:
            return None
        return self.domain.quo(self.domain.convert_from(value_numerator, ring),
                               self.domain.convert_from(value_denominator, ring))

    def _polynomials(self, element, variable: sympy.Symbol, condition: str) -> Tuple[List, List]:
        """Coefficient lists (lowest degree first, in `self.coefficients`) of numerator and denominator in
        `variable`; the denominator is not 0 at 0."""
        ring = self.domain.get_ring()
        pair = []
        for part in (self.domain.numer(element), self.domain.denom(element)):
            expression = ring.to_sympy(part)
            if expression.has(self.x if variable == self.y else self.y):  # pragma: no cover - product checked
                raise ComparisonNotImplemented(condition, DEPENDENT)
            poly = sympy.Poly(expression, variable, domain=self.coefficients)
            pair.append([self.coefficients.convert(c) for c in reversed(poly.all_coeffs())])
        numerator, denominator = pair
        if not denominator[0]:
            raise ComparisonNotImplemented(condition, "the generating function has a pole at 0")
        return numerator, denominator


def _taylor(numerator: List, denominator: List, count: int, field) -> List:
    """The first `count` power-series coefficients of numerator / denominator."""
    coefficients: List = []
    for n in range(count):
        value = numerator[n] if n < len(numerator) else field.zero
        for k in range(1, min(n, len(denominator) - 1) + 1):
            value -= denominator[k] * coefficients[n - k]
        coefficients.append(field.quo(value, denominator[0]))
    return coefficients


def _divide(numerator: List, denominator: List, field) -> Tuple[List, List]:
    """Polynomial division numerator = quotient * denominator + remainder (coefficient lists)."""
    remainder = list(numerator)
    degree = len(denominator) - 1
    while degree > 0 and not denominator[degree]:  # pragma: no cover - the lists have no leading zeros
        degree -= 1
    quotient = [field.zero] * max(len(remainder) - degree, 0)
    for k in range(len(remainder) - 1, degree - 1, -1):
        factor = field.quo(remainder[k], denominator[degree])
        if factor:
            quotient[k - degree] = factor
            for m in range(degree + 1):
                remainder[k - degree + m] -= factor * denominator[m]
    return quotient, remainder[:degree]


def _polynomial_of_matrix(coefficients: List, matrix: DomainMatrix) -> DomainMatrix:
    """sum_k coefficients[k] * matrix^k (Horner)."""
    size = matrix.shape[0]
    identity = DomainMatrix.eye(size, matrix.domain).to_dense()
    result = DomainMatrix.zeros((size, size), matrix.domain).to_dense()
    for coefficient in reversed(coefficients):
        result = result.matmul(matrix) + identity * coefficient
    return result


def _filtered(factors: _Factors, swap: bool, shift: int, diagonal: bool):
    """sum over j - i >= shift (or = shift) of a_i b_j x^i y^j, with x and y exchanged if `swap`."""
    # pylint: disable=too-many-locals
    domain, field = factors.domain, factors.coefficients
    (a_num, a_den), (b_num, b_den) = (factors.right, factors.left) if swap else (factors.left, factors.right)
    x, y = (domain.from_sympy(factors.y), domain.from_sympy(factors.x)) if swap \
        else (domain.from_sympy(factors.x), domain.from_sympy(factors.y))

    def lift(element):
        return domain.convert_from(element, field)

    polynomial, remainder = _divide(b_num, b_den, field)
    result = domain.zero
    # the polynomial part of B: finitely many terms, each needs a prefix (or one coefficient) of A
    if polynomial:
        a_head = _taylor(a_num, a_den, len(polynomial), field)
        for j, coefficient in enumerate(polynomial):
            if not coefficient or j < shift:
                continue
            indices = [j - shift] if diagonal else range(j - shift + 1)
            for i in indices:
                result += lift(coefficient * a_head[i]) * x ** i * y ** j
    # the proper part R/D via the companion matrix of D
    degree = len(b_den) - 1
    if degree == 0 or not any(remainder):
        return result
    start = _taylor(remainder, b_den, degree, field)
    rows = [[domain.zero] * degree for _ in range(degree)]
    for row in range(degree - 1):
        rows[row][row + 1] = domain.one
    for column in range(degree):
        rows[degree - 1][column] = lift(-field.quo(b_den[degree - column], b_den[0]))
    companion = DomainMatrix(rows, (degree, degree), domain)
    y_companion = companion * y
    argument = y_companion * x
    system = _polynomial_of_matrix([lift(c) for c in a_den], argument)
    if not diagonal:
        system = system.matmul(DomainMatrix.eye(degree, domain).to_dense() - y_companion)
    vector = DomainMatrix([[lift(c)] for c in start], (degree, 1), domain)
    solution = system.lu_solve(vector)
    weighted = _polynomial_of_matrix([lift(c) for c in a_num], argument).matmul(solution)
    for _ in range(shift):
        weighted = y_companion.matmul(weighted)
    return result + weighted[0, 0].element


def filter_difference(function: sympy.Expr, x: str, y: str, relation: str, shift: int,
                      condition: str) -> sympy.Expr:
    """
    The part of `function` whose exponents i of the variable x and j of y satisfy `j - i >= shift` (relation
    `"ge"`) or `j - i = shift` (relation `"eq"`), see the module documentation. Other symbols are
    coefficients.

    :raises ComparisonNotImplemented: if `function` is not a product of a rational function in x and
        a rational function in y (given the other symbols); `condition` names the guard.
    """
    function = sympy.sympify(function)
    # the symbols of `function` with these names (an engine may give its symbols assumptions)
    by_name = {str(symbol): symbol for symbol in function.free_symbols}
    factors = _Factors(function, by_name.get(x, sympy.Symbol(x)), by_name.get(y, sympy.Symbol(y)), condition)
    if relation == "eq":
        element = _filtered(factors, swap=False, shift=shift, diagonal=True) if shift >= 0 \
            else _filtered(factors, swap=True, shift=-shift, diagonal=True)
    elif relation == "ge":
        # j - i >= s for s <= 0 is the complement of i - j >= 1 - s
        element = _filtered(factors, swap=False, shift=shift, diagonal=False) if shift > 0 \
            else factors.whole - _filtered(factors, swap=True, shift=1 - shift, diagonal=False)
    else:
        raise ValueError(f"Unknown relation {relation}")
    numerator = factors.domain.get_ring().to_sympy(factors.domain.numer(element))
    denominator = factors.domain.get_ring().to_sympy(factors.domain.denom(element))
    result = numerator / denominator
    return result.xreplace(factors.back) if factors.back else result
