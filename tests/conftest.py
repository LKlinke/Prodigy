import random

import pytest


@pytest.fixture(autouse=True)
def reproducible_random(request):
    """
    Several tests draw random distributions. Seeding per test (from the test's id) makes every
    run draw the same values, independent of test order. Unseeded, test_get_parameters in
    tests/distribution/sympy sometimes drew a polynomial that SymPy factored for many minutes.
    """
    state = random.getstate()
    random.seed(request.node.nodeid)
    yield
    random.setstate(state)
