"""``Var[e]`` and ``Cov[a, b]`` queries (issue #185): closed forms on all engines, parameters,
both syntaxes, and the guarantee rules."""
import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.pgcl import CovarianceInstr, VarianceInstr, parse_pgcl
from prodigy.pgcl.converter import convert_pgcl
from prodigy.pgcl.ppy_parser import parse_ppy
from prodigy.pgcl.ppy_printer import instr_to_ppy

ENGINES = ["SYMPY", "GINAC", "SYMENGINE"]
R = sympy.Rational


def analyse(source, engine="SYMPY", syntax="ppy", decisions=None):
    if engine == "GINAC":
        pytest.importorskip("prodigy_ginac")
    config = ForwardAnalysisConfig(engine=getattr(ForwardAnalysisConfig.Engine, engine),
                                   use_simplification=True)
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config,
                              decisions=decisions or {}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    return result


def values(result):
    return {(q["kind"], q["expr"]): q for q in result["queries"]}


# distribution, query, closed form
CLOSED_FORMS = [
    pytest.param("geometric(1/3)", "Var[x]", R(2, 3) / R(1, 9), id="geometric"),
    pytest.param("binomial(10, 1/4)", "Var[x]", 10 * R(1, 4) * R(3, 4), id="binomial"),
    pytest.param("poisson(3)", "Var[x]", 3, id="poisson"),
    pytest.param("geometric(1/2) + geometric(1/2) + geometric(1/2)", "Var[x]", 3 * R(1, 2) / R(1, 4),
                 id="negative-binomial"),
    pytest.param("unif(1, 6)", "Var[x]", R(35, 12), id="uniform"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("draw, query, expected", CLOSED_FORMS)
def test_variance_closed_forms(engine, draw, query, expected):
    result = analyse(f"x: nat;\nx = {draw};\nquery {query};\n", engine)
    (answer,) = result["queries"]
    assert answer["kind"] == "variance" and answer["bound"] == Bound.EXACT
    assert sympy.simplify(sym(answer["value"]) - expected) == 0
    assert answer["value_float"] == pytest.approx(float(expected))


@pytest.mark.parametrize("engine", ENGINES)
def test_covariance_of_dependent_and_independent_variables(engine):
    source = """x: nat; y: nat; z: nat;
x = geometric(1/3);
y = x + binomial(3, 1/2);
z = poisson(2);
query Cov[x, y];
query Cov[y, x];
query Cov[x, z];
query Cov[x, x];
query Cov[x, 2 * y + 1];
query Var[x + y];
"""
    got = values(analyse(source, engine))
    assert sym(got[("covariance", "x, y")]["value"]) == 6  # = Var[x]: y = x + independent noise
    assert sym(got[("covariance", "y, x")]["value"]) == 6
    assert sym(got[("covariance", "x, z")]["value"]) == 0
    assert sym(got[("covariance", "x, x")]["value"]) == 6
    assert sym(got[("covariance", "x, 2 * y + 1")]["value"]) == 12
    # Var[x + y] = Var[x] + Var[y] + 2 Cov[x, y] = 6 + 27/4 + 12
    assert sym(got[("variance", "x + y")]["value"]) == R(6) + R(27, 4) + 12


@pytest.mark.parametrize("engine", ENGINES)
def test_negatively_correlated_variables(engine):
    # x + y = 3 with x binomial(3, 1/2): Cov[x, y] = -Var[x] = -3/4
    source = "x: nat; y: nat;\nx = binomial(3, 1/2);\ny = 3 - x;\nquery Cov[x, y];\n"
    (answer,) = analyse(source, engine)["queries"]
    assert sym(answer["value"]) == R(-3, 4)


@pytest.mark.parametrize("engine", ENGINES)
def test_parametric_variance_and_covariance(engine):
    source = """param p: real;
x: nat; y: nat;
x = geometric(p);
y = x + binomial(2, p);
query Var[x];
query Cov[x, y];
query Var[y];
"""
    got = values(analyse(source, engine))
    p = sym("p")
    var_x = (1 - p) / p ** 2
    assert sympy.simplify(sym(got[("variance", "x")]["value"]) - var_x) == 0
    assert sympy.simplify(sym(got[("covariance", "x, y")]["value"]) - var_x) == 0
    assert sympy.simplify(sym(got[("variance", "y")]["value"]) - (var_x + 2 * p * (1 - p))) == 0


def test_higher_moments_are_expectations_of_powers():
    # E[x^3] of geometric(1/2) (number of failures): 3! * sum ... = 26 (known: E[X^3] = 26 for q = 1/2)
    source = "x: nat;\nx = geometric(1/2);\nquery Ex[x ** 2];\nquery Ex[x ** 3];\n"
    source += "query Ex[x * (x - 1)];\n"  # the second factorial moment, 2 (q/p)^2 = 2
    got = values(analyse(source))
    assert sym(got[("expectation", "x * (x - 1)")]["value"]) == 2
    assert sym(got[("expectation", "x ** 2")]["value"]) == 3
    assert sym(got[("expectation", "x ** 3")]["value"]) == 13


def test_pgcl_syntax_and_a_loop():
    source = """nat x; nat c;
x := 0;
c := 1;
while (c = 1) { {c := 0} [1/2] {x := x + 1} }
?Var[x];
?Cov[x, c];
"""
    got = values(analyse(source, syntax="pgcl", decisions={"0": {"strategy": "auto"}}))
    assert sym(got[("variance", "x")]["value"]) == 2
    assert sym(got[("covariance", "x, c")]["value"]) == 0
    assert got[("variance", "x")]["bound"] == Bound.EXACT


def test_a_variance_of_a_bound_is_not_available():
    source = """x: nat; c: nat;
c = 1;
while (c == 1) { { c = 0; } [1/2] { x = x + 1; } }
query Ex[x];
query Var[x];
query Cov[x, c];
"""
    result = analyse(source, decisions={"0": {"strategy": "unroll", "params": {"iterations": 5}}})
    assert result["bounds"]["kind"] == Bound.LOWER
    got = values(result)
    for key in (("variance", "x"), ("covariance", "x, c")):
        assert got[key]["value"] is None and got[key]["bound"] == Bound.UNKNOWN
        assert "not a bound" in got[key]["error"]["message"]
    assert got[("expectation", "x")]["bound"] == Bound.LOWER  # Ex keeps its interval


def test_a_distribution_without_mass_one_is_refused():
    result = analyse("x: nat;\nx = bernoulli(1/2);\nif (x == 1) { abort; } else { skip; }\nquery Var[x];\n")
    (answer,) = result["queries"]
    assert answer["value"] is None and "mass" in answer["error"]["message"]


class _Stub:
    """A distribution that only answers what the variance query asks for."""

    def __init__(self, moments):
        self.moments = moments

    def get_probability_mass(self):
        return "1"

    def get_expected_value_of(self, expression):
        return self.moments[str(expression)]


def _query(source, moments):
    # pylint: disable=import-outside-toplevel,protected-access
    from prodigy.analysis.instructionhandler.query_handler import QueryHandler
    program = parse_pgcl(source, "ppy")
    session = AnalysisSession(program, source=source, syntax="ppy",
                              config=ForwardAnalysisConfig(), decisions={}, capture_output=True)
    with session.activate():
        QueryHandler._QueryHandler__query_spread(program.instructions[0], _Stub(moments), session)
    return session.queries[-1]


def test_infinite_moments_are_never_nan():
    # integrable, infinite second moment: the variance is infinite
    answer = _query("x: nat;\nquery Var[x];\n", {"x": "1", "x * x": "oo"})
    assert sym(answer.value) == sympy.oo and answer.bound == Bound.EXACT
    # infinite mean: the variance is undefined
    with pytest.raises(NonFiniteMarginalError):
        _query("x: nat;\nquery Var[x];\n", {"x": "oo", "x * x": "oo"})
    with pytest.raises(NonFiniteMarginalError):
        _query("x: nat;\nquery Cov[x, y];\n", {"x": "1", "y": "1", "x * y": "oo"})
    with pytest.raises(NonFiniteMarginalError):
        _query("x: nat;\nquery Var[x];\n", {"x": "1", "x * x": "nan"})


def test_parsing_and_printing_round_trip():
    ppy = "x: nat;\ny: nat;\nquery Var[x];\nquery Cov[x, y + 1];\n"
    program = parse_ppy(ppy)
    var, cov = program.instructions
    assert isinstance(var, VarianceInstr) and isinstance(cov, CovarianceInstr)
    assert [instr_to_ppy(instr) for instr in program.instructions] == ["query Var[x];", "query Cov[x, y + 1];"]
    old = parse_pgcl("nat x; nat y;\n?Var[x]\n?Cov[x, y + 1]\n")
    assert old.instructions == program.instructions
    assert parse_ppy(convert_pgcl("nat x; nat y;\n?Var[x]\n?Cov[x, y + 1]\n").text) == program
    assert str(var) == "?Var[x];" and str(cov) == "?Cov[x, y + 1];"


@pytest.mark.parametrize("text, message", [
    ("x: nat;\nquery Cov[x];\n", "two expressions"),
    ("x: nat;\nquery Var[x, x];\n", "one expression"),
    ("x: nat;\nquery Foo[x];\n", "Unknown query"),
])
def test_wrong_arity_is_a_syntax_error(text, message):
    with pytest.raises(Exception, match=message):
        parse_ppy(text)
