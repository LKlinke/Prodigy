import random

import pytest

from prodigy.distribution.generating_function import GeneratingFunction, GeneratingFunctionSettings


@pytest.fixture(autouse=True)
def reproducible_random(request):
    """
    Several tests draw random distributions. Seeding per test (from the test's id) makes every
    run draw the same values, independent of test order, so a failure can be reproduced.
    """
    state = random.getstate()
    random.seed(request.node.nodeid)
    yield
    random.setstate(state)


@pytest.fixture(autouse=True)
def generating_function_settings_stay_scoped():
    """
    The engine settings of an analysis (ForwardAnalysisConfig.activate) must not outlive it. A
    leaked simplification once made test_get_parameters in tests/distribution/sympy factor a random
    polynomial in up to ten variables, which does not finish; fail at the leaking test instead.
    """
    yield
    assert GeneratingFunction.settings() == GeneratingFunctionSettings(), \
        "generating function settings leaked out of an analysis"
