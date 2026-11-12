"""Queries that cannot be computed fail only themselves (#306); filters on a sum where one summand
has finite support are computed by a case split (#306); bool literals can be assigned (#283)."""
import json

import pytest
import sympy
from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl

ENGINES = ["SYMPY", "GINAC", "SYMENGINE"]


def _engine(name):
    if name == "GINAC":
        pytest.importorskip("prodigy_ginac")
    return getattr(ForwardAnalysisConfig.Engine, name)


def _run(source, syntax, engine):
    config = ForwardAnalysisConfig(engine=_engine(engine))
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config,
                              decisions={}, capture_output=True)
    return session.run().to_json()


# P(x + y < 3) for x ~ binomial(4, 1/2), y ~ geometric(1/3): sum over x of P(x) * P(y < 3 - x)
SUM_WITH_FINITE_SUMMAND = {
    "ppy": "x: nat = binomial(4, 1/2);\ny: nat = geometric(1/3);\nquery Pr[x + y < 3];\nquery Ex[x];\n",
    "pgcl": "nat x;\nnat y;\nx := binomial(4, 1/2);\ny := geometric(1/3);\n?Pr[x + y < 3];\n?Ex[x];\n",
}

# both summands have infinite support, and x * y < 3 is no linear guard: not computable (yet)
NOT_COMPUTABLE = {
    "ppy": "x: nat = geometric(1/2);\ny: nat = geometric(1/3);\nquery Pr[x * y < 3];\nquery Ex[x];\n",
    "pgcl": "nat x;\nnat y;\nx := geometric(1/2);\ny := geometric(1/3);\n?Pr[x * y < 3];\n?Ex[x];\n",
}


@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
@pytest.mark.parametrize("engine", ENGINES)
def test_filter_on_sum_with_finite_summand_is_computed(syntax, engine):
    result = _run(SUM_WITH_FINITE_SUMMAND[syntax], syntax, engine)
    assert result["status"] == "ok", result["error"]
    probability, expectation = result["queries"]
    assert sym(probability["value"]) == sympy.Rational(133, 432)
    assert probability["bound"] == Bound.EXACT and probability["error"] is None
    assert sym(expectation["value"]) == 2


@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
@pytest.mark.parametrize("engine", ENGINES)
def test_query_that_cannot_be_computed_fails_only_itself(syntax, engine):
    result = _run(NOT_COMPUTABLE[syntax], syntax, engine)
    assert result["status"] == "ok", result["error"]
    assert result["distribution"] is not None
    assert result["bounds"]["kind"] == Bound.EXACT
    failed, expectation = result["queries"]
    assert failed["kind"] == "probability" and failed["line"] == (3 if syntax == "ppy" else 5)
    assert failed["error"]["type"] == "NotImplementedError"
    assert "not computable" in failed["error"]["message"]
    assert failed["value"] is None and failed["lower"] is None and failed["upper"] is None
    assert failed["bound"] == Bound.UNKNOWN
    assert failed["note"].startswith("Not computed:")
    assert expectation["error"] is None and sym(expectation["value"]) == 1
    assert any("could not be computed" in warning for warning in result["warnings"])


def test_case_split_keeps_correlations():
    # y depends on x: P(x + y < 2) = P(x = 0) P(y < 2 | x = 0) + P(x = 1) P(y < 1 | x = 1)
    source = ("x: nat = bernoulli(1/2);\ny: nat = 0;\nif (x == 0) {\n    y = geometric(1/2);\n} else {\n"
              "    y = geometric(1/3);\n}\nquery Pr[x + y < 2];\n")
    result = _run(source, "ppy", "SYMPY")
    assert result["status"] == "ok", result["error"]
    (query,) = result["queries"]
    assert sym(query["value"]) == sympy.Rational(1, 2) * sympy.Rational(3, 4) + sympy.Rational(1, 2) * sympy.Rational(1, 3)


def test_cli_reports_failed_query_and_keeps_the_rest(tmp_path):
    program = tmp_path / "program.ppy"
    program.write_text(NOT_COMPUTABLE["ppy"], encoding="utf-8")
    result = CliRunner().invoke(cli, ["analyze", str(program)])
    assert result.exit_code == 0, result.output
    assert "Pr[x * y < 3]:" in result.output and "not computed:" in result.output
    assert "E[x] = 1" in result.output
    as_json = CliRunner().invoke(cli, ["analyze", "--json", str(program)])
    assert as_json.exit_code == 0, as_json.output
    queries = json.loads(as_json.output)["queries"]
    assert [query["error"] is None for query in queries] == [False, True]


BOOL_LITERALS = {
    "ppy": "x: nat = binomial(2, 1/2);\nb: bool = false;\nc: bool = true;\nb = true;\nc = false;\n"
           "if (x == 0) {\n    c = true;\n}\nquery Pr[b];\nquery Pr[c];\n",
    "pgcl": "nat x;\nbool b;\nbool c;\nx := binomial(2, 1/2);\nc := true;\nb := true;\nc := false;\n"
            "if (x = 0) { c := true } else { skip }\n?Pr[b = 1];\n?Pr[c = 1];\n",
}


@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
@pytest.mark.parametrize("engine", ENGINES)
def test_bool_literal_assignment(syntax, engine):
    result = _run(BOOL_LITERALS[syntax], syntax, engine)
    assert result["status"] == "ok", result["error"]
    if syntax == "ppy":
        b_marginal, c_marginal = result["queries"]
        assert sym(b_marginal["value"]) == sym("b")
        assert sym(c_marginal["value"]) == sym("3/4 + c/4")
    else:
        b_true, c_true = result["queries"]
        assert sym(b_true["value"]) == 1
        assert sym(c_true["value"]) == sympy.Rational(1, 4)
