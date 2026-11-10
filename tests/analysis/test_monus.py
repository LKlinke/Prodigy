"""
Subtraction on natural numbers is truncated at 0 (monus, issue #220): `a - b = max(a - b, 0)` in
every intermediate result of assignments, guards and queries, on every engine, and in the exact
calculus of `prodigy.analysis.lowerbounds.egcalc`. The expected values are computed by hand.
"""
import random
from fractions import Fraction

import pytest
import sympy

from prodigy.analysis.analyzer import compute_discrete_distribution, compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.lowerbounds.egcalc import Denominator, GFCalculus
from prodigy.analysis.lowerbounds.ir import UnsupportedProgram, instr_to_ir
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import Program, parse_pgcl
from prodigy.pgcl.compiler import compile_pgcl

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

Engine = ForwardAnalysisConfig.Engine
GINAC = pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")
ENGINES = [pytest.param(Engine.SYMPY, id="SymPy"), pytest.param(Engine.SYMENGINE, id="SymEngine"),
           pytest.param(Engine.GINAC, id="GiNaC", marks=GINAC)]
SIMPLIFICATION = [pytest.param(True, id="simplified"), pytest.param(False, id="unsimplified")]


def _queries(source: str, engine, simplification: bool = False) -> list:
    config = ForwardAnalysisConfig(engine=engine, use_simplification=simplification)
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", capture_output=True,
                              config=config)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    return [sympy.nsimplify(query["value"]) if query["value"] is not None else None
            for query in result["queries"]]


def _distribution(source: str, engine, simplification: bool = False):
    program = compile_pgcl(source, syntax="ppy")
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine, use_simplification=simplification)
    result, _ = compute_discrete_distribution(program, config.factory.from_expr("1"), config)
    return result, config.factory


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
def test_expected_values_of_the_issue(engine, simplification):
    source = ("n: nat = 5;\nquery Ex[n - 7 + 7];\nquery Ex[n - 7];\nquery Ex[(n - 7) * (n - 7)];\n"
              "query Ex[n - 3];\nquery Pr[n - 7 == 0];\nquery Pr[n - 3 >= 2];\n")
    assert _queries(source, engine, simplification) == [7, 0, 0, 2, 1, 1]


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    pytest.param("n: nat = 5;\nx: nat = n - 7;", "n^5", id="truncated"),
    # every intermediate result is truncated: no simplification to n before
    pytest.param("n: nat = 5;\nx: nat = (n - 7) + 7;", "n^5*x^7", id="intermediate"),
    pytest.param("x: nat = 3 - 5;", "1", id="constants"),
    pytest.param("x: nat = 2;\nx = x - x;", "1", id="same-variable"),
    pytest.param("x: nat = 0;\nx = x - 1;", "1", id="already-zero"),
    pytest.param("x: nat = unif(0, 3);\nx = x - 2;", "3/4 + x/4", id="uniform"),
    # a - b + b = a holds only if a >= b
    pytest.param("a: nat = unif(1, 2);\nb: nat = 2;\nc: nat = a - b + b;", "a/2*b^2*c^2 + a^2/2*b^2*c^2",
                 id="add-back"),
    pytest.param("x: nat = 2;\ny: nat = 5 - x;", "x^2*y^3", id="variable-subtrahend"),
    pytest.param("x: nat = unif(0, 2);\ny: nat = 1 - x;", "1/3*y + 1/3*x + 1/3*x^2", id="variable-subtrahend-0"),
])
def test_assignments(engine, simplification, source, expected):
    result, factory = _distribution(source, engine, simplification)
    assert result == factory.from_expr(expected, *result.get_variables())


@pytest.mark.parametrize("engine", ENGINES)
def test_subtracting_a_sum_is_subtracting_twice(engine):
    prefix = "a: nat = unif(0, 6);\nb: nat = unif(0, 2);\nc: nat = unif(0, 3);\nx: nat = 0;\n"
    once, factory = _distribution(prefix + "x = a - (b + c);", engine)
    twice, _ = _distribution(prefix + "x = (a - b) - c;", engine)
    assert once == factory.from_expr(str(twice), *twice.get_variables())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    # n - 7 == 0 holds for every n <= 7
    pytest.param("n: nat = unif(5, 9);\nquery Pr[n - 7 == 0];", [Fraction(3, 5)], id="probability"),
    pytest.param("n: nat = unif(5, 9);\nx: nat = 0;\nif (n - 7 == 0) { x = 1; }\nquery Pr[x == 1];",
                 [Fraction(3, 5)], id="if"),
    pytest.param("n: nat = unif(0, 5);\nobserve(n - 3 == 0);\nquery Ex[n];", [Fraction(3, 2)], id="observe"),
    pytest.param("n: nat = unif(0, 5);\nquery Ex[n - 3];\nquery Ex[n - 3 + 3];",
                 [Fraction(1, 2), Fraction(7, 2)], id="expected-value"),
    # x = x - 1 without a guard: monus keeps x at 0
    pytest.param("x: nat = 3;\nloop (5) { x = x - 1; }\nquery Pr[x == 0];", [Fraction(1)], id="loop"),
])
def test_guards_and_queries(engine, source, expected):
    assert _queries(source, engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", [
    # g geometric(1/2): Pr[g = k] = 2^-(k+1)
    pytest.param("x: nat = 2;\ny: nat = geometric(1/2);\nx = x - y;\nquery Ex[x];\nquery Pr[x == 0];",
                 [Fraction(5, 4), Fraction(1, 4)], id="geometric-subtrahend"),
    pytest.param("g: nat = geometric(1/2);\nh: nat = g - 2;\nquery Ex[h];\nquery Ex[g - 2];\nquery Ex[2 - g];",
                 [Fraction(1, 4), Fraction(1, 4), Fraction(5, 4)], id="geometric-minuend"),
    # y - x is never negative here: the exact negativity check, no filter
    pytest.param("x: nat = geometric(1/2);\nb: nat = bernoulli(1/2);\ny: nat = x + b;\nz: nat = y - x;\n"
                 "query Ex[z];", [Fraction(1, 2)], id="never-negative"),
])
def test_infinite_supports(engine, source, expected):
    assert _queries(source, engine) == expected


@pytest.mark.parametrize("simplification", SIMPLIFICATION)
@pytest.mark.parametrize("engine", ENGINES)
def test_parameters(engine, simplification):
    source = "param p: real;\nx: nat = geometric(p);\ny: nat = x - 2;\nquery Ex[y];\nquery Ex[x - 2];\n"
    p = sympy.Symbol("p")
    for value in _queries(source, engine, simplification):
        assert sympy.simplify(sympy.sympify(str(value), locals={"p": p}) - (1 - p) ** 3 / p) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source", [
    pytest.param("x: nat = geometric(1/2);\ny: nat = geometric(1/3);\nz: nat = x - y;", id="assignment"),
    pytest.param("x: nat = geometric(1/2);\ny: nat = geometric(1/3);\nquery Ex[x - y];", id="query"),
])
def test_two_infinite_variables_are_not_implemented(engine, source):
    program = compile_pgcl(source, syntax="ppy")
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(program, source=source, syntax="ppy", capture_output=True, config=config)
    result = session.run().to_json()
    assert result["status"] == "error"
    assert result["error"]["type"] == "NotImplementedError"
    assert "x - y" in result["error"]["message"]


# --- the exact calculus of the lower-bound strategies agrees with the forward analysis

DECREMENTS = [
    "x = x - 3;",
    "x = x - 1 - 1;",
    "x = x - (1 + 2);",
    "{x = x - 2;} [1/3] {x = x + 1;}",
    "y = y + 2;\nx = x - 1;\ny = y - 3;",
    "if (x > 1) { x = x - 4; } else { y = y - 1; }",
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("body", DECREMENTS)
def test_calculus_agrees_with_forward_analysis_on_decrements(engine, body):
    program = compile_pgcl(f"x: nat;\ny: nat;\n{body}\n", syntax="ppy")
    assert isinstance(program, Program), program
    variables = ["x", "y"]
    config = ForwardAnalysisConfig(engine=engine)
    syms = {v: sympy.Symbol(v) for v in variables}
    rng = random.Random(220)
    for rates in ([Fraction(1, 2), Fraction(0)], [Fraction(1, 3), Fraction(2, 3)]):
        calc = GFCalculus(variables, Denominator(rates, [1, 1]))
        poly = {(rng.randint(0, 4), rng.randint(0, 4)): Fraction(rng.randint(1, 5), 7) for _ in range(4)}
        ours = calc.run(instr_to_ir(program.instructions), poly)
        dist = config.factory.from_expr(calc.to_expr(poly), *variables)
        zero = config.factory.from_expr("0", *variables)
        out, _ = compute_semantics(program.instructions, ProgramInfo(program), dist, zero, config)
        diff = sympy.sympify(str(out).replace("^", "**"), locals=syms) - \
            sympy.sympify(calc.to_expr(ours).replace("^", "**"), locals=syms)
        point = {s: sympy.Rational(rng.randint(1, 9), 23) for s in syms.values()}
        assert abs(complex(diff.subs(point).evalf(30))) < 1e-20


@pytest.mark.parametrize("body", ["x = x - 3 + 1;", "x = x - y;", "if (x - 1 > 0) { x = 0; }"])
def test_calculus_rejects_subtractions_that_may_truncate_inside(body):
    """`(x - 3) + 1` is not `x - 2` under monus; the calculus must not treat it as affine."""
    program = compile_pgcl(f"x: nat;\ny: nat;\n{body}\n", syntax="ppy")
    assert isinstance(program, Program), program
    calc = GFCalculus(["x", "y"], Denominator([Fraction(1, 2), Fraction(0)], [1, 1]))
    with pytest.raises(UnsupportedProgram):
        calc.run(instr_to_ir(program.instructions), {(0, 0): Fraction(1)})


@pytest.mark.parametrize("engine", ENGINES)
def test_poisson_decrement(engine):
    source = "x: nat = poisson(3);\nx = x - 2;\nquery Ex[x];\nquery Pr[x == 0];\nquery Ex[x - 1];\n"
    expected = [1 + 5 * sympy.exp(-3), sympy.Rational(17, 2) * sympy.exp(-3), sympy.Rational(27, 2) * sympy.exp(-3)]
    for value, exact in zip(_queries(source, engine), expected):
        assert sympy.simplify(sympy.S(str(value)) - exact) == 0
