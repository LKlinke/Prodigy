"""
Semantics of two `.ppy` constructs (#106): declarations inside blocks (lowered to an initialising
assignment and `x := 0`, the marginalisation, at the end of the block) and floor division `//`
(exact via the residue classes of the dividend).

Expected results are computed by hand or by expanding the program's distribution independently
(`_brute_force`), and the generating functions are compared by their power series coefficients.
"""
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import (convert_xor, parse_expr,
                                        standard_transformations)

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import Program, parse_pgcl
from prodigy.pgcl.compiler import compile_pgcl

Engine = ForwardAnalysisConfig.Engine


def _ginac(engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")


ENGINES = [Engine.SYMPY, Engine.SYMENGINE, Engine.GINAC]


def run(source, engine, initial="1"):
    _ginac(engine)
    program = compile_pgcl(source, syntax="ppy")
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine)
    return compute_discrete_distribution(program, config.factory.from_expr(initial), config)


def coefficients(dist, variables, order):
    """
    The coefficients of the power series of `dist` (a rational function) up to `order` in every
    variable, by truncated division of its numerator by its denominator.
    """
    names = [sympy.Symbol(name) for name in variables]
    function = parse_expr(str(dist), local_dict=dict(zip(variables, names)),
                          transformations=standard_transformations + (convert_xor,))
    numerator, denominator = sympy.fraction(sympy.together(function))
    num = sympy.Poly(numerator, *names).as_dict()
    den = sympy.Poly(denominator, *names).as_dict()
    zero = tuple(0 for _ in variables)
    assert den.get(zero, 0) != 0
    series = {}
    for exponents in sorted(_exponent_tuples(len(variables), order), key=sum):
        value = sympy.Rational(num.get(exponents, 0))
        for key, coefficient in den.items():
            rest = tuple(e - k for e, k in zip(exponents, key))
            if key != zero and min(rest) >= 0:
                value -= coefficient * series.get(rest, 0)
        series[exponents] = value / den[zero]
    return {key: Fraction(str(value)) for key, value in series.items() if value != 0}


def _exponent_tuples(count, order):
    if count == 0:
        yield ()
        return
    for first in range(order + 1):
        for rest in _exponent_tuples(count - 1, order):
            yield (first, *rest)


def _brute_force(states, variables, order):
    """`states`: (probability, {variable: value}); keeps the states within `order`."""
    result = {}
    for probability, state in states:
        key = tuple(state[name] for name in variables)
        if max(key) <= order:
            result[key] = result.get(key, Fraction(0)) + probability
    return result


def geometric(k):
    """P(x = k) for x := geometric(1/2)."""
    return Fraction(1, 2 ** (k + 1))


# --- floor division -------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_finite(engine):
    dist, _ = run("x, y: nat; x = unif(0, 9); y = x // 3;", engine)
    expected = _brute_force([(Fraction(1, 10), {"x": k, "y": k // 3}) for k in range(10)], ["x", "y"], 9)
    assert coefficients(dist, ["x", "y"], 9) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_in_place(engine):
    dist, _ = run("x: nat; x = unif(0, 9); x = x // 4;", engine)
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("(2 + 2*x + x^2)/5", "x")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, value", [
    ("x, y: nat; x = geometric(1/2); y = x // 3;", lambda k: k // 3),
    ("x, y: nat; x = geometric(1/2); y = (x + 1) // 2;", lambda k: (k + 1) // 2),
    ("x, y: nat; x = geometric(1/2); y = 2 * x // 3 + 1;", lambda k: 2 * k // 3 + 1),
])
def test_floor_division_infinite_rational(engine, source, value):
    dist, _ = run(source, engine)
    order = 7
    expected = _brute_force([(geometric(k), {"x": k, "y": value(k)}) for k in range(3 * order)],
                            ["x", "y"], order)
    assert coefficients(dist, ["x", "y"], order) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_closed_form(engine):
    dist, _ = run("x, y: nat; x = geometric(1/2); y = x // 2;", engine)
    config = ForwardAnalysisConfig(engine=engine)
    # sum_m 2^-(2m+1) (1 + x/2) (x^2 y)^m
    assert dist == config.factory.from_expr("(2 + x)/(4 - x^2*y)", "x", "y")


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_in_guards(engine):
    dist, _ = run("x, y: nat; x = geometric(1/2); if (x // 2 <= 1) { y = 1; } else { y = 2; }", engine)
    order = 6
    expected = _brute_force([(geometric(k), {"x": k, "y": 1 if k // 2 <= 1 else 2})
                             for k in range(order + 1)], ["x", "y"], order)
    assert coefficients(dist, ["x", "y"], order) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_rounds_alike_in_guards_and_assignments(engine):
    dist, _ = run("x, y, z: nat; x = unif(0, 11); y = x // 4; if (y == x // 4 and not (x // 4 < y)) "
                  "{ z = 1; }", engine)
    assert dist.filter(parse_pgcl("nat z; ?Pr[z = 1]").instructions[0].expr).get_probability_mass() in (1, "1")


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_of_constants(engine):
    dist, _ = run("x: nat; x = 7 // 2; if (9 // 4 == 2) { x = x + 10; }", engine)
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("x^13", "x")


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_of_non_rational_function_is_rejected(engine):
    with pytest.raises(ValueError, match="Cannot compute the floor division by 2 exactly"):
        run("x: nat; x = poisson(2); x = x // 2;", engine)


def analyse(source, decisions, engine=Engine.SYMPY):
    _ginac(engine)
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                              decisions=decisions, capture_output=True)
    return session.run().to_json()


def query(result, expr):
    return next(q for q in result["queries"] if q["expr"] == expr)


@pytest.mark.parametrize("engine", ENGINES)
def test_floor_division_in_loop_guard_and_query(engine):
    source = "x: nat = unif(0, 9);\ny: nat;\nwhile (x // 2 > 0) { x = x - 1; y = y + 1; }\n" \
             "query Pr[x == 1];\nquery Ex[(y + 1) // 2];\n"
    result = analyse(source, {"0": "unroll:iterations=20"}, engine)
    assert result["status"] == "ok" and result["bounds"]["kind"] == Bound.EXACT
    assert query(result, "x == 1")["value"] == "9/10"
    # y = 0, 0, 1, ..., 8 for x = 0, ..., 9
    assert query(result, "(y + 1) // 2")["value"] == "2"


# --- block-scoped declarations ----------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_block_variable_is_marginalised(engine):
    dist, _ = run("x: nat; if (x == 0) { t: nat = unif(1, 3); x = 2 * t; }", engine)
    assert dist.get_variables() == {"x"}
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("(x^2 + x^4 + x^6)/3", "x")


@pytest.mark.parametrize("engine", ENGINES)
def test_block_variable_restarts_in_every_iteration(engine):
    dist, _ = run("s: nat; loop(3) { t: nat; t = t + 1; s = s + t; }", engine)
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("s^3", "s")


@pytest.mark.parametrize("engine", ENGINES)
def test_block_variables_in_sibling_and_nested_blocks(engine):
    source = """x, y: nat;
    x = bernoulli(1/2);
    if (x == 1) { t: nat = 3; y = t; } else { t: nat = 5; { y = t; } [1/2] { u: nat = t + 1; y = u; } }
    """
    dist, _ = run(source, engine)
    config = ForwardAnalysisConfig(engine=engine)
    assert dist.get_variables() == {"x", "y"}
    assert dist == config.factory.from_expr("x*y^3/2 + y^5/4 + y^6/4", "x", "y")


@pytest.mark.parametrize("engine", ENGINES)
def test_observe_and_abort_inside_a_block(engine):
    dist, error = run("x: nat = unif(0, 3); loop(1) { y: nat = bernoulli(1/2); observe(not (x + y == 2)); }",
                      engine)
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("(2 + x + x^2 + 2*x^3)/6", "x")  # conditioned
    assert sympy.sympify(str(error.get_probability_mass())) == sympy.Rational(1, 4)

    dist, _ = run("x: nat; if (x == 0) { t: nat = bernoulli(1/2); if (t == 1) { abort; } x = 5; }", engine)
    assert dist == config.factory.from_expr("x^5/2", "x")


@pytest.mark.parametrize("engine", ENGINES)
def test_block_variable_in_function(engine):
    source = """def f(a: nat) -> nat { r: nat; if (a > 0) { t: nat = bernoulli(1/2); r = a + t; } return r; }
    x: nat; x = f(2);"""
    dist, _ = run(source, engine)
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("(x^2 + x^3)/2", "x")


@pytest.mark.parametrize("engine", ENGINES)
def test_input_value_of_block_variable_is_dropped(engine):
    dist, _ = run("x: nat; if (x == 1) { t: nat = 2; x = x + t; }", engine, initial="x*t^3")
    config = ForwardAnalysisConfig(engine=engine)
    assert dist == config.factory.from_expr("x^3", "x")


GEOMETRIC_PPY = """x: nat = 1;
c: nat;
while (x == 1) {
    coin: nat = bernoulli(1/2);
    if (coin == 1) { x = 0; } else { c = c + 1; }
}
query Ex[c];
query Pr[c == 0];
"""


@pytest.mark.parametrize("engine", ENGINES)
def test_evt_invariant_does_not_mention_block_variable(engine):
    decisions = {"0": {"strategy": "evt_invariant", "params": {"invariant": "(x + 1/2)/(1 - c/2)"}}}
    result = analyse(GEOMETRIC_PPY, decisions, engine)
    assert result["status"] == "ok" and result["bounds"]["kind"] == Bound.EXACT
    assert result["distribution"]["variables"] == ["c", "x"]
    assert query(result, "c")["value"] == "1"
    assert query(result, "c == 0")["value"] == "1/2"


@pytest.mark.parametrize("engine", ENGINES)
def test_synthesised_invariant_does_not_need_block_variable(engine):
    result = analyse(GEOMETRIC_PPY, {"0": "evt_synthesis"}, engine)
    assert result["status"] == "ok" and result["bounds"]["kind"] == Bound.EXACT
    assert query(result, "c")["value"] == "1"
    certificate = result["loops"][0]["certificates"][0]
    assert "coin" not in str(certificate)


@pytest.mark.parametrize("engine", ENGINES)
def test_unrolling_with_block_variable(engine):
    result = analyse(GEOMETRIC_PPY, {"0": "unroll:iterations=5"}, engine)
    assert result["bounds"]["kind"] == Bound.LOWER
    pr = query(result, "c == 0")
    assert (Fraction(pr["lower"]), Fraction(pr["upper"])) == (Fraction(1, 2), Fraction(17, 32))


def test_sandwich_with_block_variable():
    result = analyse(GEOMETRIC_PPY, {"0": "evt_sandwich"})
    assert result["status"] == "ok"
    pr = query(result, "c == 0")
    assert Fraction(pr["lower"]) <= Fraction(1, 2) <= Fraction(pr["upper"])
