"""
Tests of internals of the SymPy engine. The contract of the ``Distribution`` interface is tested on all engines in
``tests/distribution/test_conformance.py`` (#361).
"""
import random

import pytest
import sympy

from prodigy.distribution.generating_function import GeneratingFunction, _is_finite_function, _parse_to_sympy


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
