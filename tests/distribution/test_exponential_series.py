"""
Taylor coefficients of exponentials of polynomials by the recurrence of ``F' = g'·F`` (Prodigy#455):
the closed form of the compound Poisson–Binomial distribution of the issue, generic members of the class
(random polynomial exponents, rational factors, sums, parameters, more than one variable) against SymPy's
series expansion, the joint columns against the nested expansion, and the functions outside the class.
"""
import itertools
import math
import random

import pytest
import sympy

from prodigy.distribution.coefficient_stream import series_prefix, taylor_stream, taylor_window
from prodigy.distribution.exponential_series import exp_polynomial_columns, exp_polynomial_taylor_stream

x, y, z, p, lam = sympy.symbols("x y z p lam")
HALF = sympy.Rational(1, 2)

# x = poisson(2); y = iid(binomial(10, 1/2), x): the forms of the SymPy and the GiNaC engine
ISSUE_SYMPY = sympy.exp(-2) * sympy.Mul(*(sympy.exp(2 * sympy.binomial(10, k) * x * y ** k / 1024)
                                          for k in range(11)))
ISSUE_GINAC = sympy.exp(2 * (x * ((1 + y) / 2) ** 10 - 1))


def _issue_closed_form(i, j):
    """P(X = i, Y = j) = exp(-2)·2^i/i!·binom(10 i, j)·2^(-10 i) (the issue)."""
    return sympy.exp(-2) * sympy.Integer(2) ** i / sympy.factorial(i) * sympy.binomial(10 * i, j) \
        / sympy.Integer(2) ** (10 * i)


@pytest.mark.parametrize("form", [ISSUE_SYMPY, ISSUE_GINAC], ids=["sympy-form", "ginac-form"])
def test_issue_grid_is_the_closed_form(form):
    columns = exp_polynomial_columns(form, x, y, (0, 7), (0, 20))
    assert columns is not None
    for i, column in enumerate(columns):
        for j, cell in enumerate(column):
            assert sympy.simplify(cell - _issue_closed_form(i, j)) == 0, (i, j)


def test_issue_grid_far_from_the_origin():
    columns = exp_polynomial_columns(ISSUE_SYMPY, x, y, (60, 61), (300, 302))
    cells = [list(column) for column in columns]
    assert [[sympy.simplify(c - _issue_closed_form(i, j)) for j, c in zip(range(300, 303), row)]
            for i, row in zip((60, 61), cells)] == [[0] * 3] * 2


@pytest.mark.parametrize("form", [ISSUE_SYMPY, ISSUE_GINAC], ids=["sympy-form", "ginac-form"])
def test_issue_marginals(form):
    # x is Poisson(2); y is compound: its coefficients are Σ_i P(X = i, Y = j)
    marginal_x = list(taylor_stream(form.subs(y, 1), x, 12))
    assert marginal_x == [sympy.exp(-2) * 2 ** i / sympy.factorial(i) for i in range(12)]
    marginal_y = list(taylor_stream(form.subs(x, 1), y, 6))
    expected = [sympy.exp(-2) * sum(sympy.Integer(2) ** i / sympy.factorial(i) * sympy.binomial(10 * i, j)
                                    / sympy.Integer(2) ** (10 * i) for i in range(60)) for j in range(6)]
    for value, approximation in zip(marginal_y, expected):
        # the sum over i converges quickly; the closed form exp(-1023/512)·(polynomial) is exact
        assert value.has(sympy.exp(sympy.Rational(-1023, 512)))
        assert abs(float(value) - float(approximation)) < 1e-12


def _same(a, b):
    """a = b, exactly, at rational values of the other symbols (SymPy's ``simplify`` does not always see that
    ``exp(y)**(4/5)``, which the series expansion writes, is ``exp(4*y/5)``)."""
    values = {p: sympy.Rational(2, 7), y: sympy.Rational(3, 5), z: sympy.Rational(1, 3), lam: sympy.Rational(5, 3)}
    return sympy.simplify((a - b).subs(values)) == 0


def _random_polynomial(rng, symbol, degree, others=()):
    """A polynomial in ``symbol`` with rational coefficients (some with ``others``) and a positive leading term."""
    terms = []
    for k in range(degree + 1):
        coefficient = sympy.Rational(rng.randint(-5, 5), rng.randint(1, 6))
        if others and rng.random() < 0.4:
            coefficient *= rng.choice(others)
        terms.append(coefficient * symbol ** k)
    terms.append(sympy.Rational(rng.randint(1, 4), rng.randint(1, 4)) * symbol ** (degree + 1))
    return sympy.Add(*terms)


@pytest.mark.parametrize("seed", range(6))
def test_generic_members_match_the_series_expansion(seed):
    rng = random.Random(seed)
    g = _random_polynomial(rng, x, rng.randint(1, 3), (p, y))
    factor = _random_polynomial(rng, x, rng.randint(0, 2)) / (1 - sympy.Rational(rng.randint(1, 3), 4) * x * p)
    function = factor * sympy.exp(g)
    if rng.random() < 0.5:
        function += HALF * sympy.exp(lam * (x - 1))
    stream = exp_polynomial_taylor_stream(function, x)
    assert stream is not None
    computed = list(itertools.islice(stream, 6))
    expanded = series_prefix(function, x, 6)
    assert all(_same(a, b) for a, b in zip(computed, expanded))


@pytest.mark.parametrize("seed", range(12))
def test_exponentials_in_a_denominator(seed):
    # members of the class written as a normalisation writes them (Prodigy#472): R·exp(g) with exp(-g) a common
    # factor of a denominator, also with the constants inside the exponentials (exp(x - 1/2) + exp(x + 3/2)),
    # and a sum of exponentials over c·exp(g) (the odd part of a Poisson draw, sinh(x)/sinh(1))
    rng = random.Random(200 + seed)
    g = _random_polynomial(rng, x, rng.randint(1, 2), (p,))
    split = [sympy.exp(-part) for part in sympy.Add.make_args(g)]
    constants = [sympy.Rational(rng.randint(1, 5), rng.randint(1, 4)), sympy.exp(-1), sympy.sqrt(3) * sympy.I]
    chosen = constants[:rng.randint(2, 3)]
    numerator = _random_polynomial(rng, x, rng.randint(0, 2))
    form = seed % 3
    if form == 0:
        function = numerator / sympy.Add(*(c * sympy.Mul(*split) for c in chosen))
        usual = numerator * sympy.exp(g) / sympy.Add(*chosen)
    elif form == 1:
        function = numerator / sympy.Add(*(sympy.exp(-g + k) for k in range(len(chosen))))
        usual = numerator * sympy.exp(g) / sympy.Add(*(sympy.exp(k) for k in range(len(chosen))))
    else:
        weights = [sympy.Rational(rng.randint(1, 5), rng.randint(1, 4)) for _ in range(3)]
        function = sympy.Add(*(w * sympy.exp(k * g + k) for k, w in enumerate(weights))) \
            / (chosen[0] * sympy.exp(g - 1) + chosen[1] * sympy.exp(g + 1))
        usual = sympy.Add(*(w * sympy.exp((k - 1) * g + k) for k, w in enumerate(weights))) \
            / (chosen[0] * sympy.exp(-1) + chosen[1] * sympy.exp(1))
    assert not any(isinstance(f, sympy.exp) for f in sympy.Mul.make_args(function))
    stream = exp_polynomial_taylor_stream(function, x)
    assert stream is not None
    computed = list(itertools.islice(stream, 5))
    # the expansion of the same function in the usual form (SymPy expands the other one very slowly)
    expanded = series_prefix(usual, x, 5)
    assert all(_same(a, b) for a, b in zip(computed, expanded))


@pytest.mark.parametrize("seed", range(4))
def test_generic_joint_columns_match_the_nested_expansion(seed):
    rng = random.Random(100 + seed)
    # exp(λ(P(x, y, z) − 1)) with a polynomial PGF P, times a geometric factor in y and an independent z
    pgf = sympy.Add(*(sympy.Rational(rng.randint(1, 4)) * x ** rng.randint(0, 2) * y ** rng.randint(0, 2)
                      * z ** rng.randint(0, 1) for _ in range(3)))
    pgf = pgf / pgf.subs({x: 1, y: 1, z: 1})
    function = sympy.exp(rng.randint(1, 3) * (pgf - 1)) * HALF / (1 - y / 2)
    columns = exp_polynomial_columns(function, x, y, (1, 3), (0, 3))
    nested = [list(taylor_stream(c, y, 4)) for c in list(taylor_stream(function, x, 4))[1:]]
    assert all(_same(a, b) for row, other in zip(columns, nested) for a, b in zip(row, other))


@pytest.mark.parametrize("function", [
    sympy.exp(2 * x - 2), sympy.exp(lam * (x - 1)), (1 - p) / (1 - p * x) * sympy.exp(x - 1),
    HALF * sympy.exp(x - 1) + HALF * x ** 2, sympy.sqrt(p) * sympy.exp(p * (x - 1)) / (1 - p * x),
])
def test_coefficients_are_identical_to_the_series_expansion(function):
    """On the functions whose series expansion was fast before, the coefficients are the same expressions."""
    assert list(itertools.islice(exp_polynomial_taylor_stream(function, x), 8)) == series_prefix(function, x, 8)


@pytest.mark.parametrize("function", [
    (1 - p) / (1 - p * x) * (1 - p) / (1 - p * y), HALF / (1 - HALF * x * y), (x + y) ** 3 / 8,
    HALF * (1 - p) / (1 - p * x * y) + HALF * x ** 2, sympy.exp(x + y - 2),
    (1 - p) * sympy.exp(lam * (x - 1)) / (1 - p * y), sympy.sqrt(p) * x * y / (2 - x - y),
])
def test_joint_columns_are_identical_to_the_nested_expansion(function):
    """Rational functions belong to the class (g = 0); the grid's cells are the same expressions as before."""
    window = list(map(list, taylor_window(function, x, y, (0, 4), (1, 4))))
    nested = [list(taylor_stream(c, y, 5))[1:] for c in taylor_stream(function, x, 5)]
    assert window == nested


@pytest.mark.parametrize("function", [
    sympy.exp(1 / (1 - x)), sympy.sqrt(sympy.exp(x)), (sympy.exp(x) - 1) / x, sympy.exp(x) / x,
    sympy.log(1 + x) * sympy.exp(x), x / (1 - x), sympy.exp(p),
])
def test_functions_outside_the_class(function):
    assert exp_polynomial_taylor_stream(function, x) is None


def test_symbols_with_the_same_name_are_outside_the_class():
    other = sympy.Symbol("p", positive=True)
    assert exp_polynomial_taylor_stream(sympy.exp(p * x) * other, x) is None


def test_taylor_stream_uses_the_recurrence_far_out():
    """The 400th coefficient of a Poisson distribution, which a series expansion takes long for."""
    coefficients = list(taylor_stream(sympy.exp(3 * (x - 1)), x, 401))
    assert coefficients[400] == sympy.exp(-3) * sympy.Integer(3) ** 400 / sympy.Integer(math.factorial(400))
