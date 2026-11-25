"""
Tests of internals of the SymPy engine. The contract of the ``Distribution`` interface is tested on all engines in
``tests/distribution/test_conformance.py`` (#361).
"""
import random

import pytest
import sympy

from prodigy.distribution.generating_function import (GeneratingFunction, _cancel, _is_finite_function, _limit_at_one,
                                                      _parse_to_sympy)


def test_split_addend():
    probability = sympy.S(random.random())
    number_of_vars = random.randint(1, 10)
    values = [random.randint(1, 5000) for _ in range(number_of_vars)]
    monomial = sympy.S(1)
    for i in range(number_of_vars):
        monomial *= sympy.S("x" + str(i))**values[i]
    addend = probability * monomial
    assert GeneratingFunction._split_addend(addend) == (probability, monomial)


@pytest.mark.parametrize("function, variables, finite", [
    ("x**2/2 + x/2", "x", True),
    ("(x**3 - 1)/(x - 1)", "x", True),  # the denominator cancels
    ("x/p + y", "x y", True),  # a parameter in the denominator
    ("(1/2)/(1 - x/2)", "x", False),
    ("x/(1 - y/2) - x*y/2/(1 - y/2)", "x y", True),  # the closed forms cancel to a polynomial
    ("(1 - (1 - c**2)**(1/2))/c", "c", False),
    ("exp(x - 1)", "x", False),
])
def test_finiteness_heuristic(function, variables, finite):
    # Issue #267: the heuristic skips ratsimp's division when the cancelled denominator has a
    # variable; its result is the one of the plain ratsimp check.
    expr = _parse_to_sympy(function)
    symbols = frozenset(sympy.Symbol(name, real=True) for name in variables.split())
    assert _is_finite_function(expr, symbols) is finite
    assert bool(expr.ratsimp().is_polynomial(*symbols)) is finite
    assert GeneratingFunction(function, *variables.split())._is_finite is finite


def _exponential_sum(rng, z):
    """A random nonzero combination of the constants exp(k/2) (and E), with coefficients that depend on z."""
    return sum(rng.randint(1, 9) * sympy.exp(sympy.Rational(rng.randint(-5, 7), 2)) * z ** rng.randint(0, 1)
               for _ in range(rng.randint(2, 4)))


@pytest.mark.parametrize("seed", range(12))
def test_cancel_of_sums_over_exponential_constants(seed):
    # #486: normalised Poisson draws are sums of fractions P_i(z)/(K_i*(z - 1)) with sums K_i of exponential
    # constants; the summands together have a polynomial numerator divisible by z - 1
    rng = random.Random(seed)
    z = sympy.Symbol("z", real=True)
    parts = [_exponential_sum(rng, z) for _ in range(rng.randint(2, 4))]
    constants = [_exponential_sum(rng, sympy.Integer(1)) for _ in parts]
    # the numerators sum to (z - 1)*z: the last one makes up the difference
    numerators = parts[:-1] + [(z - 1) * z * constants[-1] - sum(p * constants[-1] / c for p, c in zip(parts[:-1],
                                                                                            constants))]
    function = sum(n / (c * (z - 1)) for n, c in zip(numerators, constants))
    cancelled = _cancel(function)
    _, denominator = cancelled.as_numer_denom()
    assert z not in denominator.free_symbols
    point = sympy.Rational(rng.randint(2, 9), 11)
    assert abs(sympy.N(cancelled.subs(z, point) - point, 30)) < 1e-25  # the sum is z
    assert _is_finite_function(function, frozenset({z})) is True


@pytest.mark.parametrize("seed", range(12))
def test_marginal_of_a_function_singular_only_in_another_variable(seed):
    # #486: the denominators vanish at 1 only through z (K*(z - 1)*exp(x/2)); x is marginalised by substitution
    rng = random.Random(seed)
    x, z = sympy.symbols("x z", real=True)
    function = sum(rng.randint(1, 9) * z ** rng.randint(0, 3) * sympy.exp(x * rng.randint(0, 2) / 2)
                   / (_exponential_sum(rng, z) * sympy.exp(x / 2) - _exponential_sum(rng, sympy.Integer(1))
                      * sympy.exp(x / 2)) for _ in range(rng.randint(2, 4)))
    marginal = _limit_at_one(function, x)
    assert x not in marginal.free_symbols
    point = sympy.Rational(rng.randint(2, 9), 11)
    assert abs(sympy.N((marginal - function.subs(x, 1)).subs(z, point), 30)) < 1e-25
