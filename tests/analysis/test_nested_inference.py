"""
Nested inference (#104): ``dist`` values, ``infer``, ``sample``, ``Pr[d == k]`` and ``Ex[d]``.

The expected results are computed by hand (see the comments); generating functions are compared
symbolically. The design is in ``docs/source/language-design.md`` ("Nested inference").
"""
import json

import pytest
import sympy
from click.testing import CliRunner

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import Program, parse_pgcl
from prodigy.pgcl.compiler import compile_pgcl
from prodigy.web.diagnostics import check_source
from prodigy.web.requests import run_analysis

Engine = ForwardAnalysisConfig.Engine
ENGINES = [Engine.SYMPY, Engine.SYMENGINE, Engine.GINAC]


def analyse(source, engine=Engine.SYMPY, syntax="ppy", decisions=None):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    program = compile_pgcl(source, syntax=syntax)
    assert isinstance(program, Program), program
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config,
                              decisions=decisions, capture_output=True)
    return session.run().to_json(), session


def same(actual, expected):
    """Whether two generating functions (text) are equal."""
    difference = sympy.simplify(sym(actual) - sym(expected))
    return difference == 0


def closed_form(result):
    assert result["status"] == "ok", result["error"]
    return result["distribution"]["closed_form"]


def query_values(result):
    return [query["value"] for query in result["queries"]]


# f(n): a, b ~ bernoulli(1/2), observe(a + b >= 1), return a + b + n.
# Given a + b >= 1: a + b = 1 with probability 2/3, = 2 with 1/3, so
# infer(f, n) = (2 z^(n+1) + z^(n+2)) / 3, Pr[d == n + 1] = 2/3, Ex[d] = n + 4/3.
F = """
def f(n: nat) -> nat {
    a: nat = bernoulli(1/2);
    b: nat = bernoulli(1/2);
    observe(a + b >= 1);
    return a + b + n;
}
"""


@pytest.mark.parametrize("engine", ENGINES)
def test_constant_dist_two_draws_and_copy(engine):
    source = F + """
d: dist = infer(f, 0);
x: nat = sample(d);
e2: dist = d;
d = infer(f, 1);
y: nat = sample(e2);
z: nat = sample(d);
"""
    result, session = analyse(source, engine)
    # x and y draw independently from infer(f, 0), z from infer(f, 1)
    g0 = "(2*{v} + {v}^2)/3"
    g1 = "(2*{v}^2 + {v}^3)/3"
    expected = f"({g0.format(v='x')})*({g0.format(v='y')})*({g1.format(v='z')})"
    assert same(closed_form(result), expected)
    assert result["bounds"]["kind"] == "exact"
    # the observations inside f do not condition the program
    assert sym(result["distribution"]["error_probability"]) == 0
    values = {value.variable: value for value in session.dist_values}
    assert same(str(values["d"].closed.distribution), g1.format(v="dist_value__"))
    assert same(str(values["e2"].closed.distribution), g0.format(v="dist_value__"))
    assert values["d"].probability == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_pr_and_ex_in_guards_and_probabilities(engine):
    source = F + """
d: dist = infer(f, 0);
w: nat = 0;
if (Ex[d] > 1 and Pr[d == 1] == 2/3 and Pr[d < 2] < 1 and Pr[2 <= d] == 1/3) {
    w = 1;
}
v: nat = 0;
{ v = 1; } [Pr[d == 2]] { v = 2; }
u: nat = bernoulli(Ex[d] - 1);
query Pr[w == 1];
query Pr[v == 1];
query Ex[u];
"""
    result, _ = analyse(source, engine)
    assert [sym(value) for value in query_values(result)] == [1, sympy.Rational(1, 3), sympy.Rational(1, 3)]


@pytest.mark.parametrize("engine", ENGINES)
def test_state_dependent_arguments_with_finite_support(engine):
    # g(n): a ~ bernoulli(1/2), observe(a + n >= 1), return a + n.
    # n = 0: value 1; n = 1: 1 or 2; n = 2: 2 or 3 (each 1/2). Changing n afterwards does not
    # change d, and both draws depend on n only through d.
    source = """
def g(n: nat) -> nat {
    a: nat = bernoulli(1/2);
    observe(a + n >= 1);
    return a + n;
}
n: nat = unif(0, 2);
d: dist = infer(g, n);
n = 0;
x: nat = sample(d);
c: dist = d;
y: nat = sample(c);
"""
    result, session = analyse(source, engine)
    expected = "x*y/3 + (x/2 + x^2/2)*(y/2 + y^2/2)/3 + (x^2/2 + x^3/2)*(y^2/2 + y^3/2)/3"
    assert same(closed_form(result), expected)
    assert sorted(value.probability for value in session.dist_values if value.variable == "d") == \
        [sympy.Rational(1, 3)] * 3


@pytest.mark.parametrize("engine", ENGINES)
def test_infinite_support_without_observe_is_a_call(engine):
    # n ~ geometric(1/2), d = infer(h, n) with h(n) = n + bernoulli(1/2) (no observe, Z = 1):
    # x = n + bernoulli(1/2); Pr[n = k] = (1/2)^(k+1)
    source = """
def h(n: nat) -> nat {
    a: nat = bernoulli(1/2);
    return a + n;
}
n: nat = geometric(1/2);
d: dist = infer(h, n);
x: nat = sample(d);
"""
    result, _ = analyse(source, engine)
    # x and n stay dependent: x = n or x = n + 1
    assert same(closed_form(result), "(1 + x)/2 * (1/2)/(1 - n*x/2)")


@pytest.mark.parametrize("engine", ENGINES)
def test_dist_as_function_argument(engine):
    source = F + """
def twice(q: dist) -> nat {
    a: nat = sample(q);
    b: nat = sample(q);
    return a + b;
}
d: dist = infer(f, 0);
x: nat = twice(d);
"""
    result, _ = analyse(source, engine)
    assert same(closed_form(result), "((2*x + x^2)/3)^2")


@pytest.mark.parametrize("engine", ENGINES)
def test_dist_parameters_in_loops_and_infer_with_a_dist_argument(engine):
    # d = bernoulli(1/2); k = infer(h, d): a + b of two draws given a + b >= 1, so 1 with 2/3 and
    # 2 with 1/3 (mean 4/3). Each iteration adds g(d) (Ex[d] = 1/2 > 1/4, so a draw, mean 1/2)
    # and a draw from k: Ex[s] = 2 * (1/2 + 4/3) = 11/3; s >= 2 always.
    source = """
def f() -> nat {
    a: nat = bernoulli(1/2);
    return a;
}
def g(q: dist) -> nat {
    w: nat = 0;
    if (Ex[q] > 1/4) {
        w = sample(q);
    }
    return w;
}
def h(q: dist) -> nat {
    a: nat = sample(q);
    b: nat = sample(q);
    observe(a + b >= 1);
    return a + b;
}
d: dist = infer(f);
k: dist = infer(h, d);
c: nat = 2;
s: nat = 0;
while (c > 0) {
    y: nat = g(d);
    t: nat = sample(k);
    s = s + y + t;
    c = c - 1;
}
query Ex[s];
query Pr[s < 2];
"""
    result, _ = analyse(source, engine, decisions={"*": "unroll:iterations=5"})
    assert query_values(result) == ["11/3", "0"]
    assert result["bounds"]["kind"] == "exact"


@pytest.mark.parametrize("engine", ENGINES)
def test_loop_that_reads_a_dist(engine):
    # three iterations add three draws from geometric(1/2) (mean 1): Ex[x] = 3
    source = """
def f() -> nat {
    a: nat = geometric(1/2);
    return a;
}
d: dist = infer(f);
x: nat = 0;
c: nat = 3;
while (c > 0 and Ex[d] < 2) {
    y: nat = sample(d);
    x = x + y;
    c = c - 1;
}
query Ex[x];
"""
    result, _ = analyse(source, engine, decisions={"*": "unroll:iterations=10"})
    assert query_values(result) == ["3"]
    assert result["bounds"]["kind"] == "exact"


@pytest.mark.parametrize("engine", ENGINES)
def test_infinite_expectation_and_irrational_probability(engine):
    # Ex[d] = ∞ for 1 - sqrt(1 - z); Pr[p == 0] = exp(-1) ≈ 0.368 for poisson(1)
    source = """
def f() -> nat {
    y: nat;
    y = sample_pgf(1 - (1 - y) ** (1/2), y);
    return y;
}
def h() -> nat {
    y: nat = poisson(1);
    return y;
}
d: dist = infer(f);
p: dist = infer(h);
w: nat = 0;
if (Ex[d] > 100 and Pr[p == 0] > 1/3 and Pr[p == 0] < 3/8) {
    w = 1;
}
query Pr[w == 1];
"""
    result, _ = analyse(source, engine)
    assert query_values(result) == ["1"]


def test_irrational_probability_as_a_probability_is_not_implemented():
    source = """
def h() -> nat {
    y: nat = poisson(1);
    return y;
}
p: dist = infer(h);
w: nat = bernoulli(Pr[p == 0]);
"""
    result, _ = analyse(source)
    assert result["status"] == "error"
    assert "not a rational number" in result["error"]["message"]


@pytest.mark.parametrize("engine", ENGINES)
def test_normalising_constant_zero_is_an_analysis_error(engine):
    source = """
def f() -> nat {
    a: nat = bernoulli(1/2);
    observe(a == 2);
    return a;
}
d: dist = infer(f);
"""
    result, _ = analyse(source, engine)
    assert result["status"] == "error"
    assert result["error"]["type"] == "UndefinedDistributionError"
    assert "Z = 0" in result["error"]["message"]


def test_unassigned_dist_is_an_error():
    result, _ = analyse("d: dist;\nx: nat = sample(d);\n")
    assert result["error"]["type"] == "UndefinedDistributionError"
    assert "d has no value" in result["error"]["message"]


@pytest.mark.parametrize("source, message", [
    # a dist value in the state of a loop
    ("""
def f(n: nat) -> nat { return n; }
d: dist;
c: nat = 2;
while (c > 0) {
    d = infer(f, c);
    c = c - 1;
}
""", "dist values in the state of a loop are not implemented"),
    # infinite support and a normalising constant that depends on the argument
    ("""
def f(n: nat) -> nat {
    a: nat = bernoulli(1/2);
    observe(a + n >= 1);
    return a;
}
n: nat = geometric(1/2);
d: dist = infer(f, n);
""", "may depend on it (it has observe). Not implemented"),
    # Pr/Ex of a dist that depends on the state with infinite support
    ("""
def h(n: nat) -> nat { return n; }
n: nat = geometric(1/2);
d: dist = infer(h, n);
w: nat = 0;
if (Ex[d] > 1) { w = 1; }
""", "Pr and Ex of such a dist are not implemented"),
    # a loop that reads such a dist
    ("""
def h(n: nat) -> nat { return n; }
n: nat = geometric(1/2);
d: dist = infer(h, n);
c: nat = 1;
while (c == 1) { x: nat = sample(d); c = x % 2; }
""", "depends on the program state with infinite support"),
    # Pr[d == k] for a k that depends on the state
    (F + """
d: dist = infer(f, 0);
k: nat = unif(0, 1);
w: nat = bernoulli(Pr[d == k]);
""", "depends on the program state is not implemented"),
])
def test_not_implemented_cases_say_so(source, message):
    result, _ = analyse(source, decisions={"*": "unroll:iterations=5"})
    assert result["status"] == "error"
    assert result["error"]["type"] == "UnsupportedFeatureError"
    assert message in result["error"]["message"]
    assert "traceback" not in result["error"]


# Calls inside functions (#209). two(q) = coin(q) + coin(0) with coin(p) = bernoulli(1/2) + p, so
# x = two(1) is 1, 2, 3 with 1/4, 1/2, 1/4; cond(0) samples from infer(coin, 0) and observes w >= 1,
# so k = 1 and the observation fails with probability 1/2.
CALLS_INSIDE_FUNCTIONS = """
def coin(p: nat) -> nat {
    a: nat = bernoulli(1/2);
    return a + p;
}
def two(q: nat) -> nat {
    u: nat = coin(q);
    v: nat = coin(0);
    return u + v;
}
def cond(r: nat) -> nat {
    d: dist = infer(coin, r);
    w: nat = sample(d);
    observe(w >= 1);
    return w;
}
x: nat = two(1);
y: nat = bernoulli(1/2);
z: nat = two(y);
k: nat = cond(0);
"""


@pytest.mark.parametrize("engine", ENGINES)
def test_functions_call_functions(engine):
    result, _ = analyse(CALLS_INSIDE_FUNCTIONS, engine)
    # x = 1 + coin + coin; z = y + coin + coin with y = bernoulli(1/2); k = 1 after the observation
    expected = "k * x*(1/2 + x/2)^2 * (1/2 + z/2)^2 * (1/2 + y*z/2)"
    assert same(closed_form(result), expected)
    assert sym(result["distribution"]["error_probability"]) == sympy.Rational(1, 2)


@pytest.mark.parametrize("source, calls", [
    ("def f(a: nat) -> nat {\n    b: nat = 0;\n    if (a > 0) { b = f(a - 1); }\n    return b;\n}\n"
     "x: nat = f(2);\n", "f calls f"),
    ("def f(a: nat) -> nat { b: nat = g(a); return b; }\n"
     "def g(a: nat) -> nat { d: dist = infer(f, a); b: nat = sample(d); return b; }\n"
     "x: nat = g(1);\n", "g calls f, f calls g"),
])
def test_recursion_is_not_implemented(source, calls):
    assert isinstance(compile_pgcl(source, syntax="ppy"), Program)  # the type checker accepts it
    result, _ = analyse(source)
    assert result["status"] == "error"
    assert result["error"]["type"] == "UnsupportedFeatureError"
    assert result["error"]["message"] == f"Recursive functions are not implemented: {calls}."


def test_pgcl_syntax():
    source = """
dist d;
nat x;
nat w;
fun f := {
    nat a;
    nat b;
    a := bernoulli(1/2);
    b := bernoulli(1/2);
    observe(a + b >= 1);
    return a + b;
}
d := infer { f() }
x := sample { d }
if (Ex[d] > 1 & Pr[d = 1] = 2/3) { w := 1 } else { w := 0 }
?Pr[w = 1]
"""
    result, _ = analyse(source, syntax="pgcl")
    assert query_values(result) == ["1"]
    assert same(closed_form(result), "w*(2*x + x^2)/3")


@pytest.mark.parametrize("source, message", [
    ("x: nat = bernoulli(1/2);\nif (Ex[x] > 0) { x = 1; }\n", "wrap the code in a function"),
    ("x: nat = bernoulli(1/2);\nif (Pr[x == 1] > 0) { x = 1; }\n", "wrap the code in a function"),
    (F + "d: dist = infer(f, 0);\nquery Pr[d == 1];\n", "y = sample(d)"),
    (F + "d: dist = infer(f, 0);\nquery Ex[d];\n", "y = sample(d)"),
    (F + "d: dist = infer(f, 0);\nx: nat = Ex[d];\n", "Expected value of type"),
    (F + "d: dist = infer(f, 0);\nif (Pr[d + 1 == 1] > 0) { skip; }\n", "compare a variable d of type dist"),
    (F + "d: dist = infer(f, 0);\nx: nat = d + 1;\n", "Expected numeric value"),
])
def test_type_errors(source, message):
    checked = compile_pgcl(source, syntax="ppy")
    assert not isinstance(checked, Program)
    assert message in checked.message


def test_parser_and_printer_round_trip():
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.ppy_printer import program_to_ppy
    source = F + """
def g(q: dist) -> nat {
    a: nat = sample(q);
    return a;
}
d: dist = infer(f, n = 2);
x: nat = sample(d);
if (Ex[d] > 1 and Pr[d == 1] < 1/2) {
    x = g(d);
} else {
    skip;
}
"""
    program = parse_pgcl(source, "ppy")
    printed = program_to_ppy(program)
    assert "Ex[d] > 1 and Pr[d == 1] < 1/2" in printed
    assert parse_pgcl(printed, "ppy") == program
    pgcl = str(program)
    assert "(Ex[d] > 1) & (Pr[d = 1] < 1/2)" in pgcl
    assert parse_pgcl(pgcl) == program


@pytest.mark.parametrize("engine", ENGINES)
def test_function_call_with_state_dependent_argument(engine):
    # x = f(y) with f(a) = a: x == y (before the fix, x and y came out independent)
    source = "def f(a: nat) -> nat {\n    return a;\n}\ny: nat = unif(0, 1);\nx: nat = f(y);\n"
    result, _ = analyse(source, engine)
    assert same(closed_form(result), "1/2 + x*y/2")


@pytest.mark.parametrize("engine", ENGINES)
def test_observe_inside_a_called_function_conditions_the_program(engine):
    # P(c = 1, x = 1) ∝ 1/2 * 1/2, P(c = 0) ∝ 1/2: normalised c*x/3 + 2/3
    source = """
def f() -> nat {
    y: nat = bernoulli(1/2);
    observe(y == 1);
    return y;
}
c: nat = bernoulli(1/2);
x: nat;
if (c == 1) {
    x = f();
}
"""
    result, _ = analyse(source, engine)
    assert same(closed_form(result), "c*x/3 + 2/3")


def test_compute_discrete_distribution_without_session():
    program = compile_pgcl(F + "d: dist = infer(f, 0);\nx: nat = sample(d);\n", syntax="ppy")
    config = ForwardAnalysisConfig()
    dist, _ = compute_discrete_distribution(program, config.factory.one(*program.variables), config)
    assert same(str(dist), "(2*x + x^2)/3")
    assert dist.get_variables() == {"x"}


def test_gui_request_and_check():
    source = F + "d: dist = infer(f, 0);\nx: nat = sample(d);\nquery Ex[x];\n"
    result = run_analysis({"source": source, "syntax": "ppy"})
    assert result["status"] == "ok", result["error"]
    assert result["queries"][0]["value"] == "4/3"
    assert check_source(source, "ppy")["ok"]


@pytest.mark.parametrize("args", [["main"], ["analyze", "--json"]])
def test_cli_reports_undefined_dist_without_traceback(tmp_path, args):
    from prodigy.cli import cli  # pylint: disable=import-outside-toplevel
    program = tmp_path / "zero.ppy"
    program.write_text("def f() -> nat {\n    observe(false);\n    return 0;\n}\nd: dist = infer(f);\n")
    result = CliRunner().invoke(cli, [args[0], str(program), *args[1:]])
    assert "Z = 0" in result.output
    assert "Traceback" not in result.output
    if "--json" in args:
        assert json.loads(result.output)["status"] == "error"
