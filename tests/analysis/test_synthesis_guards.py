"""
The EVT invariant synthesis on a geometric loop with an equality guard (#282): its invariant
(2 + b*x) / (2 - x) has a numerator in two variables, which the positivity heuristic could not handle
("multivariate polynomials not supported"), and with the guard ``!(b == 1)`` single templates kept SymPy's
Gröbner basis busy for minutes.
"""
import time

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants import invariant_synthesis
from prodigy.analysis.evtinvariants.heuristics.positivity.rational_function import RationalFunctionDenomSign
from prodigy.analysis.evtinvariants.heuristics.templates.rational_function import RationalFunctionMaxDeg
from prodigy.analysis.evtinvariants.invariant_synthesis import DEFAULT_TEMPLATE_TIME_LIMIT
from prodigy.analysis.loop_strategies import get_loop_strategy
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

FACTORY = ForwardAnalysisConfig().factory


def _geometric(guard: str) -> str:
    return f"b: nat; x: nat;\nwhile ({guard}) {{ b = bernoulli(1/2); x = x + 1; }}\n"


def _run(source: str, decision: str, engine) -> dict:
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              decisions={"0": decision}, config=ForwardAnalysisConfig(engine=engine))
    return session.run().to_json()


def _same(expr: str, expected: str) -> bool:
    return sympy.simplify(sympy.S(expr.replace("^", "**")) - sympy.S(expected)) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("guard", ["b == 0", "!(b == 1)", "b < 1"])
def test_geometric_loop_gets_its_exact_result(engine, guard):
    start = time.monotonic()
    result = _run(_geometric(guard), "evt_synthesis", engine)
    assert time.monotonic() - start < 30
    assert result["status"] == "ok", result["error"]
    loop = result["loops"][0]
    assert loop["kind"] == "exact"
    certificate = loop["certificates"][0]
    assert certificate["nonnegative"] is True and certificate["finite_mass"] is True
    assert _same(certificate["invariant"], "(2 + b*x) / (2 - x)")
    # b = 1 after the loop, x is geometric on 1, 2, ...
    assert _same(result["distribution"]["closed_form"], "b*x / (2 - x)")


# -- the positivity heuristic in several variables --------------------------------------------------

@pytest.mark.parametrize("function, expected", [
    ("(2 + b*x) / (2 - x)", True),
    ("1 + x*(1 + b) / (2 - x)", True),
    ("-1 / (x - 2)", True),
    ("1 / (1 - x - y)", True),
    ("(1 + x*y) / (3 - x*y - y)", True),
    # not a power series: the constant term of the denominator is 0 (this was reported as non-negative)
    ("1 / (-x)", None),
    ("(1 - x) / (1 - y)", None),
    ("x - y", None),
    ("1 / (2 - x) - 1 / (3 - y)", None),
])
def test_positivity_in_several_variables(function, expected):
    heuristic = RationalFunctionDenomSign(FACTORY, variables=["b", "x", "y"])
    assert heuristic.is_positive(sympy.S(function)) is expected


@pytest.mark.parametrize("function", ["p / (1 - x)", "1 / (1 - p*x)", "1 / (p - x)"])
def test_positivity_does_not_assume_the_sign_of_a_parameter(function):
    assert RationalFunctionDenomSign(FACTORY, variables=["x"]).is_positive(sympy.S(function)) is None


# -- templates and the time limit per template ---------------------------------------------------

def test_template_denominators_have_the_constant_term_one():
    templates = list(RationalFunctionMaxDeg(2, ["b", "x"], FACTORY).generate())
    assert len(templates) > 10
    for template in templates:
        _, denominator = sympy.S(str(template)).as_numer_denom()
        assert "d_0" not in str(template)
        assert sympy.Poly(denominator, sympy.S("b"), sympy.S("x")).coeff_monomial(1) == 1


def test_template_time_limit_is_registered():
    spec = get_loop_strategy("evt_synthesis").param("template_time_limit")
    assert spec.default == DEFAULT_TEMPLATE_TIME_LIMIT and spec.minimum == 1 and spec.optional


def test_a_slow_template_is_skipped(monkeypatch):
    solutions = invariant_synthesis._solutions  # pylint: disable=protected-access
    calls = []

    def first_hangs(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            time.sleep(60)
        return solutions(*args, **kwargs)

    monkeypatch.setattr(invariant_synthesis, "_solutions", first_hangs)
    start = time.monotonic()
    result = _run(_geometric("b == 0"), "evt_synthesis:template_time_limit=1", ForwardAnalysisConfig.Engine.SYMPY)
    assert time.monotonic() - start < 30
    assert result["status"] == "ok", result["error"]
    assert result["loops"][0]["kind"] == "exact"
    assert len(calls) > 1


def test_skipped_templates_are_reported(monkeypatch):
    def hangs(*_args, **_kwargs):
        time.sleep(60)

    monkeypatch.setattr(invariant_synthesis, "_solutions", hangs)
    result = _run(_geometric("b == 0"), "evt_synthesis:max_degree=0,template_time_limit=1",
                  ForwardAnalysisConfig.Engine.SYMPY)
    error = result["error"]
    assert error["type"] == "SynthesisBoundReached" and error["details"]["reason"] == "degree"
    assert error["details"]["templates_skipped"] == 1 and error["details"]["template_time_limit"] == 1
    assert error["message"] == ("No rational EVT invariant up to degree 0 found (1 template tried, 1 of them "
                                "skipped after the time limit of 1 s per template).")
