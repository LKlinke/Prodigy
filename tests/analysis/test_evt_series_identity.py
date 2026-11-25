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
from prodigy.language import parse_program

ENGINES = [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.GINAC]

# A random walk on x that stops with probability 1/5 per step; it stops at x = 0 or, from x > 0, sets
# x := 5 (a marginal). By symmetry Pr[x = 5] = Pr[x = 0] = 1/2. The least fixed point is
# 5/(2 - x) + t*(1/2 + x^5/2). The candidate below is a fixed point of the closed forms only: its
# series has radius 1/2, so the marginal in `x := 5` is ∞ for the series but finite (5) for the closed
# form. Its terminating part is t, which would claim Pr[x = 5] <= 0.
RESET_WALK = """x: nat;
t: nat;
while (t == 0) {
    {
        if (x == 0) {
            t = 1;
        } else {
            t = 1;
            x = 5;
        }
    } [1/5] {
        {
            x = x + 1;
        } [1/2] {
            if (x > 0) {
                x = x - 1;
            } else {
                skip;
            }
        }
    }
}
query Pr[x == 5];
"""
SPURIOUS = "(5/3)/(1-2*x) + (10/3)/(1-x/2) + t"

BIASED_WALK = """x: nat;
x = 1;
while (x > 0) {
    {
        x = x - 1;
    } [1/3] {
        x = x + 1;
    }
}
query Pr[x == 0];
"""

# never terminates; the body marginalises x, but the invariant 1/(1 - c) converges on the open polydisc
DIVERGING = """x: nat;
c: nat;
t: nat;
while (t == 0) {
    c = c + 1;
    x = 0;
}
query Pr[c == 0];
"""


def _run(source, decision, engine):
    if engine == ForwardAnalysisConfig.Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_program(source), source=source, config=config, decisions={"0": decision},
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
def test_least_fixed_point_of_the_reset_walk_is_exact(engine):
    # its non-negativity needs the summands checked before they are combined (#297); GiNaC hands it over as
    # -1/2*(-2+x)^(-1)*(10+2*t*x^5-t*x^6+2*t-t*x), which needs the division by the denominator
    result = _run(RESET_WALK, {"strategy": "evt_invariant", "params": {"invariant": "5/(2 - x) + t*(1/2 + x^5/2)"}},
                  engine)
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == Bound.EXACT
    assert loop["certificates"][0]["nonnegative"] is True
    query = result["queries"][0]
    assert query["bound"] == Bound.EXACT and query["value"] == "1/2"


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


# A symmetric walk that ends at x = 0 almost surely, with y = 0 throughout. The body is formal, so every
# non-negative closed-form fixed point is one of the series: I = ((1 + x^2) I0(y) - 2x^2)/(1 - x)^2 for any
# I0(y) = I(0, y) (the terminating part), here I0 = 2/(1 - 2y). Park bounds its coefficients, but its series
# diverges at y = 1, where the closed form gives -2: main reported Pr[x = 0] <= -2 and Ex[y] <= 4 (#393).
FORMAL_WALK = """x: nat;
y: nat;
x = 1;
while (x > 0) {
    {
        x = x - 1;
    } [1/2] {
        x = x + 1;
    }
}
query Pr[x == 0];
query Ex[y];
"""
DIVERGENT_EXIT = "(2 + 4*x^2*y)/((1 - x)^2*(1 - 2*y))"


@pytest.mark.parametrize("engine", ENGINES)
def test_formal_loop_with_a_divergent_terminating_part_gives_no_upper_bound(engine):
    result = _run(FORMAL_WALK, {"strategy": "evt_invariant", "params": {"invariant": DIVERGENT_EXIT}}, engine)
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["certificates"][0]["nonnegative"] is True
    assert loop["certificates"][0]["series_identity"] == "formal loop body"
    assert loop["kind"] == Bound.UNKNOWN
    assert "not proven to converge at 1" in " ".join(loop["messages"])
    for query in result["queries"]:
        assert query["bound"] == Bound.UNKNOWN and query["upper"] is None


@pytest.mark.parametrize("engine", ENGINES)
def test_formal_loop_with_a_convergent_terminating_part_keeps_the_upper_bound(engine):
    # the same family with I0 = 1: the least fixed point (1 + x)/(1 - x), whose terminating part is the point
    # mass at 0; its series converges on the open polydisc (test_formal_loop_keeps_the_upper_bound has the
    # formal body as the only reason)
    result = _run(FORMAL_WALK, {"strategy": "evt_invariant", "params": {"invariant": "(1 + x^2 - 2*x^2)/(1 - x)^2"}},
                  engine)
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    if loop["certificates"][0]["nonnegative"] is not True:
        pytest.skip("non-negativity is not shown on this engine")
    assert loop["kind"] == Bound.UPPER
    probability, expectation = result["queries"]
    assert probability["bound"] == Bound.UPPER and probability["upper"] == "1"
    assert expectation["bound"] == Bound.UPPER and expectation["upper"] == "0"


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
    ("""x: nat;
y: nat;
while (x < 10 and not y == 2) {
    {
        x = x + y;
    } [1/2] {
        y = 3 + y;
    }
    skip;
}""", True),
    ("""x: nat;
y: nat;
b: bool;
while (b) {
    if (x >= 3) {
        x = x - 2;
    } else {
        b = b;
    }
}""", True),
    (RESET_WALK, False),
    (DIVERGING, False),
    ("""x: nat;
y: nat;
while (x < y) {
    x = x + 1;
}""", False),
    ("""x: nat;
while (x > 0) {
    x = geometric(1/2);
}""", False),
    ("""x: nat;
while (x > 0) {
    x = 2 * x;
}""", False),
    ("""x: nat;
y: nat;
while (x > 0) {
    x = x - y;
}""", False),
    ("""x: nat;
while (x > 0) {
    {
        x = x - 1;
    } [1 / (x + 1)] {
        skip;
    }
}""", False),
    ("""x: nat;
while (x > 0) {
    while (x > 5) {
        x = x - 1;
    }
}""", False),
])
def test_formal_loop(source, expected):
    program = parse_program(source)
    loop = next(instr for instr in program.instructions if type(instr).__name__ == "WhileInstr")
    assert formal_loop(loop, program.variables.keys()) is expected
