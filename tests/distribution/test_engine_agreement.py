"""
Engine-independent properties that the SymEngine tests checked (#464), on SymPy and GiNaC: residue classes with a
parameter agree between the engines, the odd part of a doubled Poisson variable is zero, and the Laurent terms after
a subtraction of variables stay unexpanded (#366). The rational sections without roots of unity are checked in
``test_modulo_sections.py``, the subtractions in ``test_negative_values.py``.
"""
import pytest
import sympy

from prodigy.distribution import known_convergent_series
from prodigy.distribution.generating_function import SympyPGF
from prodigy.pgcl.ppy_parser import parse_ppy_expr


def _ginac_factory():
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import ProdigyPGF
    return ProdigyPGF


FACTORIES = [pytest.param(lambda: SympyPGF, id="sympy"), pytest.param(_ginac_factory, id="ginac")]


def _expr(dist) -> sympy.Expr:
    return dist._series_expression()  # pylint: disable=protected-access


def _same(left: sympy.Expr, right: sympy.Expr) -> bool:
    names = {symbol.name: sympy.Symbol(symbol.name) for symbol in left.free_symbols | right.free_symbols}
    left, right = (e.xreplace({s: names[s.name] for s in e.free_symbols}) for e in (left, right))
    return sympy.simplify(left - right) == 0


@pytest.mark.parametrize("modulus, remainder", [(2, 0), (2, 1), (3, 2)])
def test_modulo_with_parameter_agrees_between_engines(modulus, remainder):
    pytest.importorskip("prodigy_ginac")
    sections = []
    for factory in FACTORIES:
        dist = factory.values[0]().from_expr("p/(1 - (1 - p)*x)", "x")
        sections.append(_expr(dist.filter(parse_ppy_expr(f"x % {modulus} == {remainder}"))))
    assert _same(sections[0], sections[1])


@pytest.mark.parametrize("factory", FACTORIES)
def test_odd_part_of_a_doubled_poisson_is_zero(factory):
    doubled = factory().from_expr("exp(3*(x - 1))", "x").update(parse_ppy_expr("x == x + x"))
    odd = doubled.filter(parse_ppy_expr("x % 2 == 1"))
    with known_convergent_series(True):
        assert sympy.simplify(sympy.sympify(str(odd.get_probability_mass()).replace("^", "**"))) == 0
    even = doubled.filter(parse_ppy_expr("x % 2 == 0"))
    assert _same(_expr(even), _expr(doubled))


@pytest.mark.parametrize("factory", FACTORIES)
def test_laurent_terms_after_a_subtraction_stay_unexpanded(factory):
    dist = factory().from_expr("(exp(2*x - 2) - exp(-2))/(2 - y)", "x", "y").update(parse_ppy_expr("x == x - 1"))
    expression = _expr(dist)
    assert _same(expression, sympy.sympify("(exp(2*x - 2) - exp(-2))/((2 - y)*x)"))
    # a pole at 0 of the factor 1/x, not expanded into a sum of terms
    assert expression.is_Mul or expression.is_Pow
    assert not expression.is_Add
