"""
Strategies that compute with candidate measures (#242): an EVT invariant candidate may have infinite
mass, and then a guard like `a < b` in the loop body cannot be unfolded over the values of `b`. The
check must end without a result (loop `unknown`, status `error`), never with a proven invariant built
from a wrong filter.
"""
import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

PROGRAM = """nat a; nat b; nat w;
while (w = 0) { if (a < b) { w := 1 } else { {b := b + 1} [1/2] {w := 2} } }
?Ex[b]
"""

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("simplification", [True, False])
def test_infinite_mass_invariant_candidate_gives_no_result(engine, simplification):
    config = ForwardAnalysisConfig(engine=engine, use_simplification=simplification)
    decisions = {"0": {"strategy": "evt_invariant", "params": {"invariant": "1/((1-a)*(1-b))"}}}
    session = AnalysisSession(parse_pgcl(PROGRAM), source=PROGRAM, config=config, decisions=decisions,
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "error"
    assert result["error"]["type"] == "NonFiniteMarginalError"
    assert "not a finite measure" in result["error"]["message"]
    assert result["loops"] and all(loop["kind"] == Bound.UNKNOWN for loop in result["loops"])
    assert not any(query.get("bound") in (Bound.EXACT, Bound.LOWER, Bound.UPPER)
                   for query in result.get("queries", []))
