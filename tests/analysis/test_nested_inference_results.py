"""
Results of nested inference (#104): the ``distributions`` section of the result JSON, the step
trace, the GUI's diagnostics, and the examples in ``pgfexamples/inference/nested`` against their
hand-computed ``# expected:`` values.
"""
import re
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl
from prodigy.web.diagnostics import check_source

Engine = ForwardAnalysisConfig.Engine
EXAMPLES = Path(__file__).parents[2] / "pgfexamples" / "inference" / "nested"

SOURCE = """
def g(n: nat) -> nat {
    a: nat = bernoulli(1/2);
    observe(a + n >= 1);
    return a + n;
}
n: nat = unif(0, 1);
d: dist = infer(g, n);
c: dist = infer(g, 1);
x: nat = sample(d);
"""


def run(source, **kwargs):
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", **kwargs)
    return session.run().to_json()


def test_distributions_section():
    result = run(SOURCE)
    assert result["status"] == "ok"
    # the program state does not contain the dist variables
    assert result["distribution"]["variables"] == ["n", "x"]
    entries = {entry["variable"]: entry for entry in result["distributions"]}
    assert set(entries) == {"c", "d"}
    c = entries["c"]
    assert c["depends_on_state"] is False
    [value] = c["values"]
    # g(1): 1 or 2, each 1/2; Z = 1
    assert value["infer"] == "infer(g, n = 1)" and value["function"] == "g"
    assert value["arguments"] == {"n": "1"}
    assert sympy.simplify(sym(value["closed_form"]) - sym("c/2 + c^2/2")) == 0
    assert value["mean"] == "3/2" and value["normalizing_constant"] == "1"
    assert value["finite_support"] is True
    assert [row["prob"] for row in value["table"]["rows"]] == ["1/2", "1/2"]
    d = entries["d"]
    assert d["depends_on_state"] is True
    # g(0) = 1 with Z = 1/2, g(1) as above; each with probability 1/2
    assert sorted((v["infer"], v["probability"], v["normalizing_constant"]) for v in d["values"]) == \
        [("infer(g, n = 0)", "1/2", "1/2"), ("infer(g, n = 1)", "1/2", "1")]


def test_distributions_without_closed_form_and_without_dist():
    result = run("def h(n: nat) -> nat {\n    return n;\n}\nn: nat = geometric(1/2);\n"
                 "d: dist = infer(h, n);\n")
    [value] = result["distributions"][0]["values"]
    assert value["closed"] is False and value["closed_form"] is None
    assert "infinite support" in value["note"]
    assert run("x: nat = bernoulli(1/2);\n")["distributions"] == []


def test_pgcl_spelling():
    source = "dist d;\nnat x;\nfun f := {\n nat a;\n a := bernoulli(1/2);\n return a;\n}\nd := infer { f() }\n"
    session = AnalysisSession(parse_pgcl(source), source=source)
    result = session.run().to_json()
    assert result["distributions"][0]["values"][0]["infer"] == "infer { f() }"


def test_trace_shows_dist_values_apart():
    result = run(SOURCE, record_steps=True)
    steps = {step["summary"]: step for step in result["steps"]}
    after = steps["d = infer(g, n);"]["after"]
    assert "d" not in after["closed_form"] and set(after["marginals"]) == {"n", "x"} - {"x"} | {"x"}
    assert sorted(entry["infer"] for entry in after["dists"] if entry["variable"] == "d") == \
        ["infer(g, n = 0)", "infer(g, n = 1)"]
    before = steps["d = infer(g, n);"]["before"]
    assert [entry["infer"] for entry in before["dists"] if entry["variable"] == "d"] == [None]


def test_diagnostics_warn_about_dist_in_a_loop_state():
    source = "def f(n: nat) -> nat {\n    return n;\n}\nd: dist;\nc: nat = 2;\nwhile (c > 0) {\n" \
             "    d = infer(f, c);\n    c = c - 1;\n}\n"
    result = check_source(source, "ppy")
    warning = next(d for d in result["diagnostics"] if d["severity"] == "warning")
    assert "dist values in the state of a loop are not implemented" in warning["message"]
    assert warning["line"] == 6
    assert result["ok"]  # a valid program


def _expected(source):
    return [match.group(1).strip() for match in re.finditer(r"#\s*expected:\s*(.+)", source)]


@pytest.mark.parametrize("engine", [Engine.SYMPY, Engine.SYMENGINE, Engine.GINAC])
@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.ppy")), ids=lambda p: p.stem)
def test_examples(path, engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    source = path.read_text()
    result = run(source, config=ForwardAnalysisConfig(engine=engine))
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == "exact"
    values = [query["value"] for query in result["queries"]]
    assert [sym(v) for v in values] == [sym(e) for e in _expected(source)]
