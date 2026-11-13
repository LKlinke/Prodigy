"""Positivity heuristics and the sub-invariant checks read variables named like builtins or constants as
symbols (#354), not as ``pow``, ``pi`` or ``I``."""
import pytest
import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.coefficients import series_form
from prodigy.analysis.evtinvariants.heuristics.positivity.individual_sum_terms import SumPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.logical import NotHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.lowerbounds.subinvariant import rational_mass, to_product_form
from prodigy.distribution.equality import sympify_symbolic_string

NAMES = ["pow", "pi", "I", "S", "E"]


@pytest.mark.parametrize("name", NAMES)
def test_symbolic_string_keeps_names_as_symbols(name):
    expr = sympify_symbolic_string(f"{name}^2 + exp({name}) + 1/2")
    assert expr.free_symbols == {sympy.Symbol(name)}
    assert expr.has(sympy.exp)


def test_symbolic_string_keeps_functions():
    expr = sympify_symbolic_string("sqrt(x) + log(x) + 2**x")
    assert expr.free_symbols == {sympy.Symbol("x")}


@pytest.mark.parametrize("name", NAMES)
def test_series_form(name):
    expr, gens = series_form(f"1 / (1 - {name} * x)", ["x"])
    assert [g.name for g in gens] == ["x"]
    assert {s.name for s in expr.free_symbols} == {name, "x"}


class _Record(PositivityHeuristic):
    def __init__(self):
        super().__init__(None)
        self.seen = []

    def _is_positive(self, f):
        self.seen.append(sympy.S(f).free_symbols if not isinstance(f, str) else sympify_symbolic_string(f).free_symbols)
        return True


@pytest.mark.parametrize("name", NAMES)
def test_sum_positivity_splits_terms_with_symbols(name):
    record = _Record()
    assert SumPositivity(record).is_positive(f"{name} * x + {name}^2") is True
    assert all(syms <= {sympy.Symbol(name), sympy.Symbol("x")} for syms in record.seen)
    assert len(record.seen) == 2


@pytest.mark.parametrize("name", NAMES)
def test_not_heuristic_reads_symbols(name):
    record = _Record()
    # f = 0 is decided before the inner heuristic is asked, for any spelling of the symbol
    assert NotHeuristic(record).is_positive(f"{name} - {name}") is True
    assert not record.seen


@pytest.mark.parametrize("name", NAMES)
def test_product_form_with_named_variable(name):
    form = to_product_form(f"1 / (1 - {name} / 2)", [name])
    assert form is not None


@pytest.mark.parametrize("name", NAMES)
def test_rational_mass_with_named_variable(name):
    assert rational_mass(f"(1 - 1/2) / (1 - {name} / 2)", [name]) == 1
