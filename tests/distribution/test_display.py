"""
The shown form of a result (#404): the option does not change the analysis, only the shown result is simplified,
by a bounded search over exact rewrites that never changes the generating function.
"""
import time

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.values import describe_distribution
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution import display
from prodigy.distribution.display import display_form
from prodigy.distribution.equality import decide_equal_strings
from prodigy.distribution.generating_function import GeneratingFunction
from prodigy.language.compiler import parse_program

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

Engine = ForwardAnalysisConfig.Engine
ENGINES = [pytest.param(Engine.SYMPY, id="SymPy"), pytest.param(Engine.GINAC, id="GiNaC",
                        marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed"))]


def _sympy(dist):
    return dist._sympy_expression()  # pylint: disable=protected-access


def test_a_removable_factor_is_cancelled():
    dist = GeneratingFunction("(x**2 - 1)/((x - 1)*(2 - y))", "x", "y")
    shown = display_form(dist)
    assert shown.simplified and shown.form == "cancel"
    assert shown.shown_ops < shown.computed_ops
    assert sympy.cancel(_sympy(shown.distribution) - _sympy(dist)) == 0
    assert shown.to_json()["simplified"] is True


def test_a_power_is_not_expanded():
    # the former normal form expanded (x/2 + 1/2)**10 into eleven terms
    dist = GeneratingFunction("(x/2 + 1/2)**10", "x")
    shown = display_form(dist)
    assert len(sympy.Add.make_args(_sympy(shown.distribution))) == 1
    assert shown.shown_ops <= shown.computed_ops


def test_a_short_form_is_kept():
    dist = GeneratingFunction("x/(2 - x)", "x")
    shown = display_form(dist)
    assert not shown.simplified
    assert shown.distribution is dist
    assert "already short" in shown.note


def test_large_forms_are_kept_without_trying():
    x = sympy.Symbol("x")
    big = sum(sympy.Rational(1, k + 2) * x ** k / (3 - x) ** (k % 3 + 1) for k in range(display.GCD_MAX_OPS))
    shown = display_form(GeneratingFunction(big, "x"))
    assert not shown.simplified and "too large" in shown.note
    assert shown.seconds < 1


def test_high_degrees_are_kept_without_trying():
    # sparse polynomials of high degree make GCDs and factorisations explode (#224)
    shown = display_form(GeneratingFunction("x**1024/(1 - x) + x**512/2 + 1/2", "x"))
    assert not shown.simplified and "degree" in shown.note


def test_the_time_limit_keeps_the_best_form_so_far(monkeypatch):
    def slow(expr):
        time.sleep(5)
        return expr

    rewrites = (("together", sympy.together, display.GCD_MAX_OPS), ("slow", slow, display.GCD_MAX_OPS))
    monkeypatch.setattr(display, "_REWRITES", rewrites)
    dist = GeneratingFunction("x/2 + x**2/2 + x**3/2", "x")
    start = time.perf_counter()
    shown = display_form(dist, seconds=0.3)
    assert time.perf_counter() - start < 2
    assert shown.form == "together"
    assert sympy.cancel(_sympy(shown.distribution) - _sympy(dist)) == 0


def test_the_time_limit_keeps_the_computed_form(monkeypatch):
    def slow(expr):
        time.sleep(5)
        return expr

    monkeypatch.setattr(display, "_REWRITES", (("slow", slow, display.GCD_MAX_OPS),))
    dist = GeneratingFunction("(x**2 - 1)/((x - 1)*(2 - y))", "x", "y")
    shown = display_form(dist, seconds=0.3)
    assert shown.distribution is dist
    assert "time limit" in shown.note


def test_a_wrong_rewrite_is_never_shown(monkeypatch):
    monkeypatch.setattr(display, "_REWRITES", (("wrong", lambda expr: sympy.Integer(1), display.GCD_MAX_OPS),))
    dist = GeneratingFunction("(x**2 - 1)/((x - 1)*(2 - y))", "x", "y")
    assert display_form(dist).distribution is dist


def test_a_failing_rewrite_gives_no_candidate(monkeypatch):
    def failing(expr):
        raise sympy.PolynomialError("no")

    monkeypatch.setattr(display, "_REWRITES", (("failing", failing, display.GCD_MAX_OPS),
                                               ("cancel", sympy.cancel, display.GCD_MAX_OPS)))
    shown = display_form(GeneratingFunction("(x**2 - 1)/((x - 1)*(2 - y))", "x", "y"))
    assert shown.form == "cancel"


@pytest.mark.parametrize("engine", ENGINES)
def test_the_shown_form_is_the_same_function_on_every_engine(engine):
    config = ForwardAnalysisConfig(engine=engine)
    with config.activate():
        dist = config.factory.from_expr("(x^2 - 1)/((x - 1)*(2 - y)) + p*x^2/2", "x", "y")
        shown = display_form(dist)
        assert decide_equal_strings(str(shown.distribution), str(dist), ["x", "y", "p"]) is True
        assert set(shown.distribution.get_variables()) == {"x", "y"}
        assert set(shown.distribution.get_parameters()) == {"p"}


def test_the_summary_says_which_form_is_shown():
    dist = GeneratingFunction("(x**2 - 1)/((x - 1)*(2 - x))", "x")
    plain = describe_distribution(dist, limit=3)
    assert "display" not in plain and "computed_form" not in plain
    shown = describe_distribution(dist, limit=3, simplify=True)
    assert shown["display"]["form"] == "cancel"
    assert shown["computed_form"] == plain["closed_form"]
    assert shown["computed_latex"] == plain["latex"]
    assert shown["closed_form"] != plain["closed_form"]
    # everything else comes from the computed distribution
    assert shown["marginals"] == plain["marginals"] and shown["mass"] == plain["mass"]


SOURCE = """x: nat;
y: nat;
{
    x = x + 1;
} [1/2] {
    y = y + 1;
}
{
    x = 2 * x;
} [1/3] {
    skip;
}"""


@pytest.mark.parametrize("engine", ENGINES)
def test_the_option_changes_only_the_shown_form(engine):
    results = []
    for simplification in (False, True):
        session = AnalysisSession(parse_program(SOURCE), source=SOURCE, capture_output=True,
                                  config=ForwardAnalysisConfig(engine=engine, use_simplification=simplification))
        results.append(session.run().to_json())
    plain, simplified = (result["distribution"] for result in results)
    assert "display" not in plain and "display" in simplified
    # (GiNaC's order of terms may differ from run to run)
    computed = simplified.get("computed_form", simplified["closed_form"])
    assert decide_equal_strings(computed, plain["closed_form"], ["x", "y"]) is True
    assert decide_equal_strings(simplified["closed_form"], plain["closed_form"], ["x", "y"]) is True
    assert simplified["marginals"] == plain["marginals"]
