"""
Candidate measures whose series diverges at 1 although the closed form is finite there (#393): an EVT
invariant candidate like `1/((1 - 2a)(2 - b))` has the finite closed form `-1/(2 - b)` at a = 1, but its
series in a has radius 1/2. The product `a * b` in the loop body is a case split over the values of b, which
needs the marginal of b; the check must end without a result (status `error`, `NonFiniteMarginalError`), never
with a guarantee. A guard like `a < 2*b` uses the marginal only to choose its error and still ends with "not
implemented" (or the formal comparison, #243).
"""
import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

PROGRAM = """nat a; nat b; nat c; nat w;
while (w = 0) { c := a * b; {w := 1} [1/2] {b := b + 1} }
?Ex[c]
"""
GUARD_PROGRAM = """nat a; nat b; nat w;
while (w = 0) { if (a < 2*b) { w := 1 } else { {b := b + 1} [1/2] {w := 2} } }
?Ex[b]
"""

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.GINAC]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("invariant", ["1/((1 - 2*a)*(2 - b))", "1/((1 - 2*a)*(2 - b)) - 1/((2 - a)*(2 - b))"])
def test_candidate_with_a_divergent_marginal_gives_no_result(engine, invariant):
    result = _run(PROGRAM, engine, invariant)
    assert result["error"]["type"] == "NonFiniteMarginalError"
    assert "not proven to be a finite measure" in result["error"]["message"]


@pytest.mark.parametrize("engine", ENGINES)
def test_guard_on_a_divergent_candidate_gives_no_result(engine):
    _run(GUARD_PROGRAM, engine, "1/((1 - 2*a)*(2 - b))")


def _run(program, engine, invariant):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=engine)
    decisions = {"0": {"strategy": "evt_invariant", "params": {"invariant": invariant}}}
    session = AnalysisSession(parse_pgcl(program), source=program, config=config, decisions=decisions,
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "error"
    assert all(loop["kind"] == Bound.UNKNOWN for loop in result["loops"])
    assert not any(query.get("bound") in (Bound.EXACT, Bound.LOWER, Bound.UPPER)
                   for query in result.get("queries", []))
    return result
