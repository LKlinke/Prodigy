"""
Contract of the iteration of a distribution, on all engines (#444): ``iter(dist)`` yields each state once, with
its full coefficient, also when the coefficients contain parameters. GiNaC listed a state once per term of the
expanded polynomial (``p*x + (1-p)`` gave ``-p`` and ``1`` for x = 0), and the case splits, filters and result
tables that take a yielded probability as Pr[state] were wrong.

Written as one parametrised file over the engines so that it can move into the conformance suite (#361).
"""
import pytest
import sympy

from prodigy.distribution.equality import decide_equal
from prodigy.distribution.generating_function import SympyPGF
from prodigy.pgcl.ppy_parser import parse_ppy_expr


def _ginac_factory():
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import ProdigyPGF
    return ProdigyPGF


FACTORIES = [
    pytest.param(lambda: SympyPGF, id="sympy"),
    pytest.param(_ginac_factory, id="ginac"),
]

# polynomials (the generating function, its variables) whose coefficients are sums with parameters
POLYNOMIALS = [
    ("p*x + (1-p)", ("x",)),
    ("p*x*y + (1-p)/2*x^2*y^4 + (1-p)/2", ("x", "y")),
    ("(p*x + 1 - p)*(q*y + 1 - q)", ("x", "y")),
    ("p*q*x + p*(1-q) + (1-p)*x", ("x",)),
    ("p*x + (1-p)*x^3", ("x",)),
    ("p*x*w + (1-p)*w^2", ("x", "w")),
    ("1/3*x + 1/3*x^2 + 1/3", ("x",)),
]


def _symbolic(text):
    expression = sympy.sympify(str(text).replace("^", "**"))
    return expression.xreplace({s: sympy.Symbol(s.name) for s in expression.free_symbols})


def _monomial(state):
    return sympy.Mul(*(sympy.Symbol(var) ** value for var, value in state.items()))


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("function, variables", POLYNOMIALS)
def test_one_item_per_state_with_its_full_coefficient(factory, function, variables):
    dist = factory().from_expr(function, *variables)
    items = list(dist)
    keys = [frozenset(state.items()) for _, state in items]
    assert len(keys) == len(set(keys)), items
    assert all(decide_equal(_symbolic(prob), sympy.S(0)) is not True for prob, _ in items), items
    rebuilt = sympy.Add(*(_symbolic(prob) * _monomial(state) for prob, state in items))
    assert decide_equal(rebuilt, _symbolic(function)) is True, items


@pytest.mark.parametrize("factory", FACTORIES)
def test_exhaustive_filter_keeps_sum_coefficients_together(factory):
    # the filter rebuilds the polynomial from (probability, state) pairs: a probability `1/2 - p/2` must stay a
    # factor of its monomial (GiNaC wrote `1/2-1/2*p * x^2`)
    dist = factory().from_expr("p*x + (1-p)/2*x^2 + (1-p)/2", "x")
    filtered = dist._exhaustive_search(parse_ppy_expr("not x == 1"))  # pylint: disable=protected-access
    expected = sympy.sympify("(1-p)/2*x**2 + (1-p)/2")
    assert decide_equal(_symbolic(filtered), expected) is True, str(filtered)
