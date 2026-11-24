"""Fixtures of the distribution tests: ``factory`` runs a test once per engine (#361, see ``engines.py``)."""
import pytest

from tests.distribution.engines import FACTORIES


@pytest.fixture(name="factory", params=FACTORIES)
def _factory(request):
    """The factory of an engine (SymPy, GiNaC); GiNaC is skipped where its wheel is missing."""
    return request.param()
