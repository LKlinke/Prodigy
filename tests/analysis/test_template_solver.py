"""
The template solver of the EVT synthesis (#281): the fixed-point system of a template N / D is linear in the
coefficients of N, so these are solved by linear algebra and only the coefficients of D need a non-linear
solver. Undecided fixed points are reported as candidates; #298: a coefficient whose sign is not known gives
"unknown" in ``SingleRationalFunction`` (fixed with its rewrite in #301, tested here).
"""
from pathlib import Path

import time
from types import SimpleNamespace

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.heuristics.positivity.rational_function import SingleRationalFunction
from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
from prodigy.analysis.evtinvariants.heuristics.templates.rational_function import (DENOMINATOR_WEIGHT,
                                                                                   MAX_NUMERATOR_COEFFICIENTS,
                                                                                   RationalFunctionMaxDeg)
from prodigy.analysis.evtinvariants import invariant_synthesis
from prodigy.analysis.evtinvariants.template_solver import solve_template
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]

x, y, p = sympy.symbols("x y p")
n0, n1, d1, d2 = sympy.symbols("n_0 n_1 d_1 d_2")


def _run(source: str, decision: str = "evt_synthesis", engine=ForwardAnalysisConfig.Engine.SYMPY,
         strategy=SynthesisStrategies.DEFAULT) -> dict:
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={"0": decision}, config=ForwardAnalysisConfig(engine=engine,
                                                                                      strategy=strategy))
    return session.run().to_json()


def _same(expr: str, expected: str) -> bool:
    return sympy.simplify(sympy.S(expr.replace("^", "**")) - sympy.S(expected)) == 0


# -- the solver ------------------------------------------------------------------------------------

def test_linear_and_denominator_unknowns_are_solved():
    # (n_0 + n_1 x) / (1 + d_1 x) = 2 / (2 - x), i.e. n_0 = 1, n_1 = 0, d_1 = -1/2
    equation = (n0 + n1 * x) / (1 + d1 * x) - 2 / (2 - x)
    assert solve_template(equation, [n0, n1], [d1], [x]) == [{n0: 1, n1: 0, d1: sympy.Rational(-1, 2)}]


def test_inconsistent_system_has_no_solution():
    assert not solve_template(n0 / (1 + d1 * x) - (1 + x ** 2), [n0], [d1], [x])
    assert not solve_template(n0 + n1 * x - x ** 2, [n0, n1], [], [x])


def test_parameters_stay_symbolic():
    equation = (n0 + n1 * x) / (1 + d1 * x) - 1 / (1 - p * x)
    (solution,) = solve_template(equation, [n0, n1], [d1], [x], [p])
    assert solution == {n0: 1, n1: 0, d1: -p}


def test_complex_denominators_are_dropped():
    # 1 / (1 + x^2) = n_0 / (1 + d_1 x + d_2 x^2) also for d_1 = 0 only; the conditions have no complex
    # solution here, but a solution with a non-real value must never be proposed
    for solution in solve_template(n0 / (1 + d1 * x + d2 * x ** 2) - 1 / (1 + x ** 2), [n0], [d1, d2], [x]):
        assert all(value.is_real is not False for value in solution.values())


def test_full_numerators_cover_the_smaller_ones():
    templates = list(RationalFunctionMaxDeg(4, ["b", "x"], ForwardAnalysisConfig().factory).templates())
    assert len(templates) == 95
    # fewer unknowns first, those of the denominator weighted more
    costs = [(level + 1) ** 2 + DENOMINATOR_WEIGHT * ((bounds[0] + 1) * (bounds[1] + 1) - 1)
             for level, bounds in templates]
    assert costs == sorted(costs)
    # the same pairs as all degree bounds of D with a sum of at most 2 * level
    assert set(templates) == {(level, (e1, e2)) for level in range(5) for e1 in range(9) for e2 in range(9)
                              if e1 + e2 <= 2 * level}


def test_templates_are_enumerated_lazily():
    """With six variables there are millions of pairs; listing them all first took 17 s before the first template."""
    start = time.monotonic()
    templates = RationalFunctionMaxDeg(4, list("abcdef"), ForwardAnalysisConfig().factory).templates()
    first = [next(templates) for _ in range(20)]
    assert time.monotonic() - start < 1
    assert first[0] == (0, (0,) * 6)


def test_too_large_numerators_are_left_out():
    """With seven variables the full numerator of degree 4 has 5^7 coefficients; building that template ended
    the analysis with a RecursionError in SymPy."""
    heuristic = RationalFunctionMaxDeg(4, list("abcdefg"), ForwardAnalysisConfig().factory)
    assert heuristic.effective_max_deg == 1 and 2 ** 7 <= MAX_NUMERATOR_COEFFICIENTS < 3 ** 7
    assert {level for level, _ in heuristic.templates()} == {0, 1}


def test_message_names_the_degree_reached():
    strategy = SimpleNamespace(template_heuristics=RationalFunctionMaxDeg(4, list("abcdefg"),
                                                                          ForwardAnalysisConfig().factory))
    stats = {"tried": 3, "failed": 0, "skipped": 0}
    message = invariant_synthesis._bound_message(  # pylint: disable=protected-access
        "degree", strategy, None, None, stats, [])
    assert message == ("No rational EVT invariant up to degree 1 found (3 templates tried). Templates of a higher "
                       f"degree up to 4 have more than {MAX_NUMERATOR_COEFFICIENTS} coefficients in the numerator "
                       "and are left out.")


# -- loops in several variables that are found now -------------------------------------------------

MULTI_VARIABLE = [
    # a counter in two variables with a denominator of degree 2 in y
    ("c: nat; x: nat; y: nat;\nwhile (c == 0) { { c = 1; } [1/2] { { x = x + 1; } [1/2] { y = y + 2; } } }\n",
     "(2*c + 4) / (4 - x - y**2)"),
    # two alternating states
    ("c: nat; x: nat; y: nat;\nwhile (c < 2) {\n"
     "    if (c == 0) { { c = 1; } [1/2] { x = x + 1; } } else { { c = 2; } [1/3] { c = 0; y = y + 1; } }\n}\n",
     "(c**2 + 3*c + 6) / (6 - 3*x - 2*y)"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, invariant", MULTI_VARIABLE)
def test_multi_variable_loop_gets_its_exact_result(engine, source, invariant):
    # within the default bounds (the time limit makes a slow search end with "unknown")
    result = _run(source, engine=engine)
    assert result["status"] == "ok", result["error"]
    loop = result["loops"][0]
    assert loop["kind"] == "exact"
    assert _same(loop["certificates"][0]["invariant"], invariant)


@pytest.mark.parametrize("engine", ENGINES)
def test_thirds_geometric_is_found(engine):
    # unknown on main: the degree-3 denominator was not reached within the time limit
    source = (Path(__file__).parents[2] / "pgfexamples" / "invariant_synthesis" / "thirds_geometric.ppy").read_text()
    result = _run(source, engine=engine)
    assert result["status"] == "ok", result["error"]
    assert result["loops"][0]["kind"] == "exact"


def test_loop_never_entered_has_the_invariant_zero():
    source = ("x: nat; y: nat;\nwhile (x > 0) {\n    y = 1;\n    while (y == 1) { { y = 0; } [1/2] { x = x + 1; } }\n"
              "    x = x - 1;\n}\n")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={"*": "evt_synthesis"})
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert [loop["kind"] for loop in result["loops"]] == ["exact", "exact"]


# -- undecided fixed points and parameters ---------------------------------------------------------

PARAMETRIC = "param p: real;\nb: nat; x: nat;\nwhile (b == 0) { { b = 1; } [p] { x = x + 1; } }\n"


def test_undecided_fixed_point_is_reported_as_candidate():
    result = _run(PARAMETRIC, "evt_synthesis:max_degree=1")
    assert result["status"] == "error" and result["bounds"]["kind"] == "unknown"
    error = result["error"]
    assert error["type"] == "SynthesisBoundReached"
    (candidate,) = error["details"]["candidates"]
    assert _same(candidate, "(1 + b*p) / (1 - (1 - p)*x)")
    assert error["details"]["unproven_fixed_points"] == 1
    assert "reported as candidates, not as results" in error["message"]
    (message,) = result["loops"][0]["messages"]
    assert message.startswith("Candidate EVT invariant (not certified")


@pytest.mark.parametrize("strategy", [SynthesisStrategies.DEFAULT, SynthesisStrategies.SUM_THEN_RATIONAL])
def test_parametric_probability_is_no_conditioning(strategy):
    """The error term of a choice with probability p was compared with the constant 0 including the
    parameter sets, so every parametric program ended with "does not support conditioned distributions"."""
    result = _run(PARAMETRIC, "evt_synthesis:max_degree=1", strategy=strategy)
    assert result["error"]["type"] == "SynthesisBoundReached", result["error"]


@pytest.mark.parametrize("function", ["1/(1 - I*x)", "1/(1 + I - x)", "1/(p - x)"])
def test_single_rational_function_with_unknown_sign_is_undecided(function):
    assert SingleRationalFunction(variables=["x"]).is_positive(function) is None


def test_single_rational_function_still_decides_numbers():
    assert SingleRationalFunction(variables=["x"]).is_positive("1/(1 - x/2)") is True
