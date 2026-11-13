"""``_term_generator`` yields the (coefficient, monomial) pairs of a finite SymPy polynomial, smallest first (#348)."""
import sympy

from prodigy.distribution.generating_function import _term_generator

X, Y = sympy.symbols("x y")


def _terms(expr, *gens):
    return list(_term_generator(sympy.Poly(expr, *gens)))


def test_univariate_in_increasing_order():
    assert _terms(3 * X ** 2 + 1 + 2 * X, X) == [(1, 1), (2, X), (3, X ** 2)]


def test_multivariate_order_of_states():
    assert _terms(X ** 2 + X * Y + Y ** 2 + 3, X, Y) == [(3, 1), (1, Y ** 2), (1, X * Y), (1, X ** 2)]


def test_other_generator_order():
    assert _terms(X + Y ** 2, Y, X) == [(1, X), (1, Y ** 2)]


def test_constant_polynomial():
    assert _terms(sympy.Integer(5), X) == [(5, 1)]


def test_zero_polynomial():
    assert _terms(sympy.Integer(0), X) == []


def test_symbolic_coefficients_and_gaps():
    p = sympy.Symbol("p")
    assert _terms(p * X ** 5 + (1 - p), X) == [(1 - p, 1), (p, X ** 5)]


def test_many_terms_reassemble():
    expr = sum((i + 1) * X ** i * Y ** (i % 3) for i in range(60))
    terms = _terms(expr, X, Y)
    assert len(terms) == 60
    assert sympy.expand(sum(c * m for c, m in terms) - expr) == 0
