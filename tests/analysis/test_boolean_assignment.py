"""A comparison or boolean operation on the right-hand side of an assignment, e.g. ``b = x < 2``, is
the case split ``if (x < 2) { b = true } else { b = false }`` (#319)."""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

ENGINES = ["SYMPY", "GINAC", "SYMENGINE"]


def _run(source, syntax, engine):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine))
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config,
                              decisions={}, capture_output=True)
    return session.run().to_json()


# x ~ geometric(1/2) with P(x = k) = 1/2^(k+1). At the end b = [x = 3], c = [x >= 2], d = true.
PROGRAMS = {
    "ppy": "x: nat = geometric(1/2);\nb: bool;\nc: bool;\nd: bool;\n"
           "b = x < 2;\nc = not b;\nb = b and c or x == 3;\nd = not d;\n"
           "query Pr[b == 1];\nquery Pr[c == 1];\nquery Pr[b == 1 and c == 1];\nquery Pr[d == 1];\n",
    "pgcl": "nat x;\nbool b;\nbool c;\nbool d;\nx := geometric(1/2);\n"
            "b := x < 2;\nc := not b;\nb := b & c || x = 3;\nd := not d;\n"
            "?Pr[b = 1];\n?Pr[c = 1];\n?Pr[b = 1 & c = 1];\n?Pr[d = 1];\n",
}


@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
@pytest.mark.parametrize("engine", ENGINES)
def test_boolean_expression_on_the_right_hand_side(syntax, engine):
    result = _run(PROGRAMS[syntax], syntax, engine)
    assert result["status"] == "ok", result["error"]
    values = [sympy.simplify(sym(query["value"])) for query in result["queries"]]
    assert values == [sympy.Rational(1, 16), sympy.Rational(1, 4), sympy.Rational(1, 16), 1]


@pytest.mark.parametrize("engine", ENGINES)
def test_boolean_expression_with_a_draw(engine):
    """A draw inside the boolean expression is lowered first, then the case split runs."""
    source = "b: bool;\nb = bernoulli(1/3) == 1 or false;\nquery Pr[b == 1];\n"
    result = _run(source, "ppy", engine)
    assert result["status"] == "ok", result["error"]
    (query,) = result["queries"]
    assert sympy.simplify(sym(query["value"])) == sympy.Rational(1, 3)
