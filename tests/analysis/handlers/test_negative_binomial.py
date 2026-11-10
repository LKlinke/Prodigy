"""
The built-in distribution ``negative_binomial(r, p)`` (issue #174): the number of failures before the
``r``-th success, i.e. the sum of ``r`` independent samples of ``geometric(p)``, on every engine and in
both syntaxes. Arguments the analysis cannot handle soundly are rejected by the type checker.
"""
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.generating_function import SympyPGF
from prodigy.pgcl import FunctionCallExpr, NatLitExpr, RealLitExpr, parse_pgcl
from prodigy.pgcl.check import CheckFail
from prodigy.pgcl.compiler import compile_pgcl
from prodigy.pgcl.converter import convert_pgcl
from prodigy.pgcl.ppy_printer import program_to_ppy

try:
    import prodigy_ginac  # pylint: disable=unused-import
    HAS_GINAC = True
except ImportError:  # pragma: no cover - prodigy_ginac has no wheel for this platform
    HAS_GINAC = False

ENGINES = [
    pytest.param(ForwardAnalysisConfig.Engine.SYMPY, id="SymPy"),
    pytest.param(ForwardAnalysisConfig.Engine.SYMENGINE, id="SymEngine"),
    pytest.param(ForwardAnalysisConfig.Engine.GINAC, id="GiNaC",
                 marks=pytest.mark.skipif(not HAS_GINAC, reason="prodigy_ginac is not installed")),
]

EXAMPLE = (Path(__file__).resolve().parents[3]
           / "pgfexamples/inference/loop_free/conditioning/high_risk_policyholder.ppy")


def _run(source: str, engine: ForwardAnalysisConfig.Engine, syntax: str = "pgcl"):
    program = compile_pgcl(source, syntax=syntax)
    assert not isinstance(program, CheckFail), program
    config = ForwardAnalysisConfig(engine=engine)
    with config.activate():
        result, error_prob = compute_discrete_distribution(
            program, config.factory.one(*program.variables.keys()), config)
        assert error_prob.is_zero_dist()
        return result, config.factory


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("r, p", [(0, "1/2"), (1, "1/3"), (2, "1/2"), (3, "1/3"), (2, "0.25"), (2, "1")])
def test_equals_iid_geometric(engine, r, p):
    negative_binomial, factory = _run(f"nat x; nat y; y := 1; x := negative_binomial({r}, {p})", engine)
    iid, _ = _run(f"nat x; nat y; y := 1; x := iid(geometric({p}), {r})", engine)
    assert negative_binomial == iid
    assert negative_binomial == factory.from_expr(f"y * (({p}) / (1 - (1 - ({p})) * x))^{r}", "x", "y")


@pytest.mark.parametrize("engine", ENGINES)
def test_one_success_is_geometric(engine):
    negative_binomial, _ = _run("nat x; x := negative_binomial(1, 1/3)", engine)
    geometric, _ = _run("nat x; x := geometric(1/3)", engine)
    assert negative_binomial == geometric


@pytest.mark.parametrize("engine", ENGINES)
def test_inside_iid(engine):
    """The sum of two samples of NB(2, p) is NB(4, p)."""
    twice, _ = _run("nat x; x := iid(negative_binomial(2, 1/2), 2)", engine)
    once, _ = _run("nat x; x := negative_binomial(4, 1/2)", engine)
    assert twice == once


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("r, p", [(1, sympy.Rational(1, 2)), (3, sympy.Rational(1, 3)), (5, sympy.Rational(3, 4))])
def test_mean(engine, r, p):
    result, _ = _run(f"nat x; x := negative_binomial({r}, {p})", engine)
    with ForwardAnalysisConfig(engine=engine).activate():
        mean = result.get_expected_value_of("x")
    assert sympy.sympify(mean) == r * (1 - p) / p


@pytest.mark.parametrize("engine", ENGINES)
def test_parametric_success_probability(engine):
    result, factory = _run("nat x; rparam q; x := negative_binomial(2, q)", engine)
    assert result == factory.from_expr("(q / (1 - (1 - q) * x))^2", "x").set_parameters("q")
    with ForwardAnalysisConfig(engine=engine).activate():
        mean = sympy.sympify(result.get_expected_value_of("x"))
    q = sympy.Symbol("q")
    mean = mean.subs({s: sympy.Symbol(s.name) for s in mean.free_symbols})
    assert sympy.simplify(mean - 2 * (1 - q) / q) == 0


def test_factory_validates_its_arguments():
    assert SympyPGF.negative_binomial("x", 2, "1/2") == SympyPGF.geometric("x", "1/2") * SympyPGF.geometric("x", "1/2")
    with pytest.raises(ValueError, match="must be in"):
        SympyPGF.negative_binomial("x", 2, "3/2")
    with pytest.raises(ValueError, match="must be in"):
        SympyPGF.negative_binomial("x", 2, "0")
    with pytest.raises(ValueError, match="natural number"):
        SympyPGF.negative_binomial("x", "1/2", "1/2")


@pytest.mark.parametrize("syntax, source", [
    ("pgcl", "nat x; x := negative_binomial(2, 1/2)"),
    ("ppy", "x: nat;\nx = negative_binomial(2, 1/2);\n"),
])
def test_parsing(syntax, source):
    program = parse_pgcl(source, syntax)
    assert program.instructions[0].rhs == FunctionCallExpr(
        "negative_binomial", ([NatLitExpr(2), RealLitExpr("1/2")], {}))


def test_printing_and_conversion():
    program = parse_pgcl("nat x; x := negative_binomial(2, 1/2)")
    printed = program_to_ppy(program)
    assert "x = negative_binomial(2, 1/2);" in printed
    assert parse_pgcl(printed, "ppy").instructions == program.instructions
    converted = convert_pgcl("nat x;\nx := negative_binomial(3, 0.25);\n").text
    assert "x = negative_binomial(3, 0.25);" in converted


@pytest.mark.parametrize("syntax, source", [
    ("pgcl", "fun negative_binomial := { nat y; y := 1; return y; } nat x; x := 1"),
    ("ppy", "def negative_binomial() -> nat { return 1; }\nx: nat;\n"),
])
def test_name_is_reserved_for_functions(syntax, source):
    with pytest.raises(SyntaxError, match="negative_binomial"):
        parse_pgcl(source, syntax)


@pytest.mark.parametrize("source, message", [
    ("nat x; nparam n; x := negative_binomial(n, 1/2)", "must be a natural number"),
    ("nat x; nat y; x := negative_binomial(y, 1/2)", "must be a natural number"),
    ("nat x; x := negative_binomial(1 + 1, 1/2)", "must be a natural number"),
    ("nat x; x := negative_binomial(0.5, 1/2)", "must be a natural number"),
    ("nat x; x := negative_binomial(2, 0)", "must be in (0, 1]"),
    ("nat x; x := negative_binomial(2, 3/2)", "must be in (0, 1]"),
    ("nat x; x := negative_binomial(2, 1 - 3/2)", "must be in (0, 1]"),
    ("nat x; x := negative_binomial(2, 2)", "must be in (0, 1]"),
    ("nat x; x := negative_binomial(2)", "Too few parameters"),
    ("nat x; x := negative_binomial(2, 1/2, 1/2)", "Too many parameters"),
])
def test_rejected_arguments(source, message):
    result = compile_pgcl(source)
    assert isinstance(result, CheckFail)
    assert message in result.message


def test_constants_are_substituted():
    program = compile_pgcl("x: nat;\nconst R = 2;\nx = negative_binomial(R, 1/2);\n", syntax="ppy")
    assert not isinstance(program, CheckFail), program


@pytest.mark.parametrize("engine", ENGINES)
def test_high_risk_policyholder_example(engine):
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "negative_binomial" in source
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=config,
                              decisions={}, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    assert result["bounds"]["kind"] == Bound.EXACT
    (query,) = result["queries"]
    assert sym(query["value"]) == sympy.Rational(2, 11)
