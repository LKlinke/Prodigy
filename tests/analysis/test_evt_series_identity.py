"""
EVT invariants are checked on closed forms, Park's lemma needs the equation for power series (#276).
A spurious closed-form fixed point whose series diverges inside the unit polydisc must not give an
upper bound when the loop body marginalises a variable; see
:mod:`prodigy.analysis.evtinvariants.series_identity`.
"""
import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.evtinvariants.series_identity import converges_on_unit_polydisc, formal_loop
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.SYMENGINE,
           ForwardAnalysisConfig.Engine.GINAC]

# A random walk on x that stops with probability 1/5 per step; it stops at x = 0 or, from x > 0, sets
# x := 5 (a marginal). By symmetry Pr[x = 5] = Pr[x = 0] = 1/2. The least fixed point is
# 5/(2 - x) + t*(1/2 + x^5/2). The candidate below is a fixed point of the closed forms only: its
# series has radius 1/2, so the marginal in `x := 5` is ∞ for the series but finite (5) for the closed
# form. Its terminating part is t, which would claim Pr[x = 5] <= 0.
RESET_WALK = """nat x; nat t;
while (t = 0) {
  { if (x = 0) { t := 1 } else { t := 1; x := 5 } } [1/5] { {x := x + 1} [1/2] { if (x > 0) { x := x - 1 } else { skip } } }
}
?Pr[x = 5]
"""
SPURIOUS = "(5/3)/(1-2*x) + (10/3)/(1-x/2) + t"

BIASED_WALK = """nat x;
x := 1
while (x > 0) { {x := x - 1} [1/3] {x := x + 1} }
?Pr[x = 0]
"""

# never terminates; the body marginalises x, but the invariant 1/(1 - c) converges on the open polydisc
DIVERGING = """nat x; nat c; nat t;
while (t = 0) { c := c + 1; x := 0 }
?Pr[c = 0]
"""


def _run(source, decision, engine):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_pgcl(source), source=source, config=config, decisions={"0": decision},
                              capture_output=True)
    return session.run().to_json()


@pytest.mark.parametrize("engine", ENGINES)
def test_spurious_closed_form_fixed_point_gives_no_upper_bound(engine):
    result = _run(RESET_WALK, {"strategy": "evt_invariant", "params": {"invariant": SPURIOUS}}, engine)
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == Bound.UNKNOWN
    certificate = loop["certificates"][0]
    assert certificate["series_identity"] is None
    query = result["queries"][0]
    assert query["bound"] == Bound.UNKNOWN and query["upper"] is None


@pytest.mark.parametrize("engine", ENGINES)
def test_formal_loop_keeps_the_upper_bound(engine):
    # the spurious invariant -1/2 + 3/2/(1 - 2x) has radius 1/2, but the walk only shifts x
    result = _run(BIASED_WALK, {"strategy": "evt_invariant", "params": {"invariant": "-1/2 + 3/2/(1 - 2*x)"}},
                  engine)
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    if loop["certificates"][0]["nonnegative"] is not True:
        pytest.skip("non-negativity is not shown on this engine")
    assert loop["kind"] == Bound.UPPER
    assert loop["certificates"][0]["series_identity"] == "formal loop body"
    assert result["queries"][0]["bound"] == Bound.UPPER


@pytest.mark.parametrize("engine", ENGINES)
def test_infinite_mass_converging_on_the_polydisc_keeps_the_upper_bound(engine):
    result = _run(DIVERGING, {"strategy": "evt_invariant", "params": {"invariant": "1/(1 - c)"}}, engine)
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == Bound.UPPER
    certificate = loop["certificates"][0]
    assert certificate["finite_mass"] is False
    assert certificate["series_identity"] == "converges on the open unit polydisc"


@pytest.mark.parametrize("expression, expected", [
    ("(x + 1/2)/(1 - c/2)", True),
    ("(1 + x)/(1 - x)", True),
    ("1/((1 - x)*(1 - c))", True),
    ("1/(1 - 2*x)", False),
    ("1/(1 - (x + c)/2)", True),
    ("1/(1 - x*c - x)", False),
    (SPURIOUS, False),
    ("1/(1 - p*x)", None),
    ("(1 - sqrt(1 - c^2))/c", None),
])
def test_converges_on_unit_polydisc(expression, expected):
    assert converges_on_unit_polydisc(expression, ["x", "c", "t"]) is expected


@pytest.mark.parametrize("source, expected", [
    (BIASED_WALK, True),
    ("nat x; nat y; while (x < 10 & not (y = 2)) { {x := x + y} [1/2] {y := 3 + y}; skip }", True),
    ("nat x; nat y; bool b; while (b) { if (x >= 3) { x := x - 2 } else { b := b } }", True),
    (RESET_WALK, False),
    (DIVERGING, False),
    ("nat x; nat y; while (x < y) { x := x + 1 }", False),
    ("nat x; while (x > 0) { x := geometric(1/2) }", False),
    ("nat x; while (x > 0) { x := 2 * x }", False),
    ("nat x; nat y; while (x > 0) { x := x - y }", False),
    ("nat x; while (x > 0) { {x := x - 1} [1/(x+1)] {skip} }", False),
    ("nat x; while (x > 0) { while (x > 5) { x := x - 1 } }", False),
])
def test_formal_loop(source, expected):
    program = parse_pgcl(source)
    loop = next(instr for instr in program.instructions if type(instr).__name__ == "WhileInstr")
    assert formal_loop(loop, program.variables.keys()) is expected
