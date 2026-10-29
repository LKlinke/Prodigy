import random

import pytest

from prodigy.distribution.generating_function import GeneratingFunction


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
def isolated_generating_function_settings():
    """
    Every ForwardAnalysisConfig writes its use_simplification and use_latex flags into class
    attributes of GeneratingFunction, which then hold for the whole process. Without restoring
    them, a test that analyses with simplification makes all later tests simplify as well:
    test_get_parameters in tests/distribution/sympy then factors a random polynomial in up to ten
    variables, which does not finish.
    """
    settings = GeneratingFunction.use_simplification, GeneratingFunction.use_latex_output
    yield
    GeneratingFunction.use_simplification, GeneratingFunction.use_latex_output = settings
