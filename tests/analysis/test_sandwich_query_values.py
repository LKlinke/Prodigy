"""The sandwich computes the exact query values on its lower bound from the certificate (issue #87): they must be
exactly what evaluating the queries on the generating function of the lower bound gives, on every engine."""
from fractions import Fraction
from pathlib import Path

import pytest
import sympy

import prodigy.analysis.lowerbounds.sandwich as sandwich_module
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.lowerbounds.parameters import TemplateShape
from prodigy.analysis.lowerbounds.problem import load_problem, queries_after
from prodigy.analysis.lowerbounds.sandwich import evt_sandwich
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.ast import ExpectationInstr, ProbabilityQueryInstr, WhileInstr

LOOP = (Path(__file__).resolve().parents[2] / "pgfexamples" / "lower_bounds" / "sandwich"
        / "rw_counter_23.ppy").read_text().split("query Pr", 1)[0]
QUERIES = """query Pr[c >= 5];
query Pr[c == 3];
print;
query Ex[c];
query Ex[x];
query Pr[c <= 6 and x == 0];
query Pr[not c == 1];
query Pr[c];
"""
SOURCE = LOOP + QUERIES
# GiNaC's expectation only takes a variable, so affine expectations are compared on the other engines only
AFFINE = "query Ex[2*c + 1];\n"
# small fixed templates, so that the test does not depend on (and wait for) the parameter search
LOWER = [TemplateShape([Fraction(0), Fraction(17, 20)], [1, 2], [6, 12], reach_box=[6, 14])]
UPPER = [TemplateShape([Fraction(1, 2), Fraction(9, 10)], [2, 2], [8, 16])]


def _fixed_sandwich(problem, **kwargs):
    return evt_sandwich(problem, lower_shapes=LOWER, upper_shapes=UPPER, verbose=False,
                        **{k: v for k, v in kwargs.items() if k not in ("lower_shapes", "upper_shapes", "verbose")})


@pytest.fixture(name="result", scope="module")
def fixture_result():
    program = parse_pgcl(SOURCE + AFFINE, "ppy")
    problem = load_problem(program)
    assert problem.epilogue == ("seq", [])
    loop = next(ins for ins in program.instructions if isinstance(ins, WhileInstr))
    problem.queries = queries_after(program, loop)  # as the while handler does: without the marginal
    return _fixed_sandwich(problem)


@pytest.mark.parametrize("engine", list(ForwardAnalysisConfig.Engine))
def test_query_values_equal_the_values_on_the_lower_bound(engine, result):
    config = ForwardAnalysisConfig(engine=engine)
    lower = config.factory.from_expr(result.lower.calc.to_expr(result.lower.exit_numerator), "x", "c")
    program = parse_pgcl(SOURCE + AFFINE, "ppy")
    queries = [ins for ins in program.instructions if isinstance(ins, (ProbabilityQueryInstr, ExpectationInstr))]
    by_text = {(b.query.kind, b.query.text): b for b in result.bounds}
    checked = 0
    for ins in queries:
        kind = "?Pr" if isinstance(ins, ProbabilityQueryInstr) else "?Ex"
        bound = by_text.get((kind, str(ins.expr)))
        if bound is None:  # the marginal ?Pr[c]
            continue
        if kind == "?Ex" and str(ins.expr) != "c" and str(ins.expr) != "x" \
                and engine == ForwardAnalysisConfig.Engine.GINAC:
            continue
        if kind == "?Pr":
            old = lower.filter(ins.expr).get_probability_mass()
        else:
            old = lower.get_expected_value_of(ins.expr)
        assert isinstance(bound.value, Fraction)
        assert sympy.nsimplify(sympy.sympify(str(old))) == sympy.Rational(bound.value.numerator,
                                                                           bound.value.denominator)
        # the guarantee: the value lies within the reported bounds
        assert bound.lower >= bound.value and bound.value <= bound.upper
        checked += 1
    assert checked == (6 if engine == ForwardAnalysisConfig.Engine.GINAC else 7)


def _session_queries(monkeypatch, engine, *, shortcut: bool):
    monkeypatch.setattr(sandwich_module, "evt_sandwich", _fixed_sandwich)
    if not shortcut:
        monkeypatch.setattr(AnalysisSession, "loop_query_value", lambda self, kind, expression, dist: None)
    used = []
    original = AnalysisSession.loop_query_value

    def spy(self, kind, expression, dist):
        value = original(self, kind, expression, dist)
        used.append(value is not None)
        return value
    monkeypatch.setattr(AnalysisSession, "loop_query_value", spy)
    session = AnalysisSession(parse_pgcl(SOURCE, "ppy"), source=SOURCE, syntax="ppy", config=ForwardAnalysisConfig(engine=engine),
                              decisions={"0": "evt_sandwich"}, capture_output=True)
    out = session.run().to_json()
    assert out["status"] == "ok", out.get("error")
    monkeypatch.undo()
    return out, used


def _comparable(queries):
    """GiNaC prints the terms of a sum in no fixed order: compare printed generating functions as expressions."""
    out = []
    for query in queries:
        query = dict(query)
        if query["kind"] in ("print", "marginal"):
            query["value"] = sympy.expand(sympy.sympify(query["value"].replace("^", "**")))
            query.pop("data", None)
        out.append(query)
    return out


@pytest.mark.parametrize("engine", list(ForwardAnalysisConfig.Engine))
def test_session_answers_like_the_closed_form(monkeypatch, engine):
    new, used = _session_queries(monkeypatch, engine, shortcut=True)
    old, unused = _session_queries(monkeypatch, engine, shortcut=False)
    assert used == [True] * 6 and unused == [False] * 6
    assert _comparable(new["queries"]) == _comparable(old["queries"])
    assert new["bounds"] == old["bounds"]


def test_values_end_at_the_next_instruction(monkeypatch):
    # `skip` returns the very same distribution object; the values must end anyway
    source = LOOP + "skip;\n" + QUERIES
    monkeypatch.setattr(sandwich_module, "evt_sandwich", _fixed_sandwich)
    used = []
    original = AnalysisSession.loop_query_value

    def spy(self, kind, expression, dist):
        value = original(self, kind, expression, dist)
        used.append(value is not None)
        return value
    monkeypatch.setattr(AnalysisSession, "loop_query_value", spy)
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=ForwardAnalysisConfig(),
                              decisions={"0": "evt_sandwich"}, capture_output=True)
    assert session.run().to_json()["status"] == "ok"
    assert used == [False] * 6
