"""
Expected values of polynomial expressions in several variables (`?Ex[x + y]`, `?Ex[x*y]`). All engines
agree with the exact values; the GiNaC engine needs prodigy-ppl-ginac 0.0.4 for sums and products. Symbols that
are neither variables nor parameters are rejected.
"""
import pytest
import sympy

from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.symengine_distribution import SymenginePGF


def _ginac_factory():
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import ProdigyPGF
    return ProdigyPGF


FACTORIES = [
    pytest.param(lambda: SympyPGF, id="sympy"),
    pytest.param(_ginac_factory, id="ginac"),
    pytest.param(lambda: SymenginePGF, id="symengine"),
]

# x ~ geometric(1/2) on {0, 1, ...} (mean 1, E[x^2] = 3) and y ~ bernoulli(1/3), independent
GF = "1/2/(1 - x/2) * (2/3 + y/3)"
CASES = [
    ("x", "1"),
    ("y", "1/3"),
    ("x + y", "4/3"),
    ("2*x + 3*y + 1", "4"),
    ("x*y", "1/3"),
    ("x^2", "3"),
    ("x^2 + x*y", "10/3"),
]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("expression, expected", CASES)
def test_expected_value_of_polynomial(factory, expression, expected):
    dist = factory().from_expr(GF, "x", "y")
    assert sympy.simplify(sympy.S(str(dist.get_expected_value_of(expression))) - sympy.S(expected)) == 0


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("expression", ["x + z", "x*z", "x^2 + z"])
def test_unknown_symbols_are_rejected(factory, expression):
    # z is neither a variable nor a parameter; the GiNaC bindings would treat it as 0
    dist = factory().from_expr(GF, "x", "y")
    with pytest.raises(ValueError, match="unknown symbols"):
        dist.get_expected_value_of(expression)
