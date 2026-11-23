"""Engine values are turned into text without ``str`` (LaTeX while ``use_latex`` is active, #445, #465)."""
import sympy

from prodigy.analysis.analyzer import _provably_positive
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.distribution.equality import engine_text, sympify_engine_string


def test_engine_text_of_a_distribution_is_not_latex():
    config = ForwardAnalysisConfig(engine=ForwardAnalysisConfig.Engine.SYMPY, use_latex=True)
    with config.activate():
        dist = config.factory.from_expr("(1/2)*x**2/(1 - y/2)", "x", "y")
        assert "\\" not in engine_text(dist)
        assert sympify_engine_string(engine_text(dist), ["x", "y"]) == sympy.sympify("x**2/(2 - y)")
        assert _provably_positive(config.factory.from_expr("1/2", "x"), [])
        assert not _provably_positive(config.factory.from_expr("0", "x"), [])
