"""
Contract of the metadata interface of a distribution, on all engines (#358): ``copy``, the getters and the
setters never share or change the variable and parameter sets of an existing distribution.

Written as one parametrised file over the engines so that it can move into the conformance suite (#361).
"""
import pytest

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


@pytest.fixture(name="dist", params=FACTORIES)
def _dist(request):
    return request.param().from_expr("p * x * y / 2 + (1 - p) * x*x", "x", "y")


def _snapshot(dist):
    return dist.get_variables(), dist.get_parameters()


def test_getters_return_copies(dist):
    before = _snapshot(dist)
    dist.get_variables().add("zzz")
    dist.get_parameters().add("zzz")
    dist.get_variables().clear()
    assert _snapshot(dist) == before


def test_copy_is_independent(dist):
    before = _snapshot(dist)
    clone = dist.copy()
    assert _snapshot(clone) == before
    clone.get_variables().add("zzz")
    clone.get_parameters().add("zzz")
    assert _snapshot(dist) == before
    assert _snapshot(clone) == before


def test_set_variables_does_not_change_receiver(dist):
    before = _snapshot(dist)
    result = dist.set_variables("x", "y")
    result.get_variables().add("zzz")
    result.get_parameters().add("zzz")
    assert _snapshot(dist) == before


def test_set_parameters_does_not_change_receiver(dist):
    before = _snapshot(dist)
    result = dist.set_parameters("p")
    assert "p" in result.get_parameters()
    assert _snapshot(dist) == before
