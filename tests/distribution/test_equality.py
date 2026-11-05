"""
Equality of distributions decides the EVT fixed-point check I = Φ(I), so no engine may report equality
without proof (#117). Unproven equality must come out as False.
"""
import pytest
import sympy
from sympy.assumptions.assume import global_assumptions

from prodigy.distribution import CommonDistributionsFactory
from prodigy.distribution.equality import decide_equal, decide_equal_strings, is_rational_over_q
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import GeneratingFunction, SympyPGF, provably_equal
from prodigy.distribution.symengine_distribution import SymenginePGF

ENGINES = [SympyPGF, SymenginePGF, ProdigyPGF]

# 1/2 + 10^-20
NEARLY_HALF = "50000000000000000001/100000000000000000000"


def equal(factory: CommonDistributionsFactory, lhs: str, rhs: str, *variables: str) -> bool:
    return factory.from_expr(lhs, *variables) == factory.from_expr(rhs, *variables)


@pytest.mark.parametrize("factory", ENGINES)
@pytest.mark.parametrize("lhs, rhs", [
    ("1/(1-x)", "1 + x/(1-x)"),
    ("(x + 1/2)/(1 - c/2)", "(2*x + 1)/(2 - c)"),
    ("(1 - x^3)/(1 - x)", "1 + x + x^2"),
    ("p*x + (1-p)", "1 - p + p*x"),
    ("1/(1 - p*x)", "1 + p*x/(1 - p*x)"),
])
def test_equal_rational_functions_are_recognised(factory, lhs, rhs):
    assert equal(factory, lhs, rhs, "x", "c")


@pytest.mark.parametrize("factory", ENGINES)
@pytest.mark.parametrize("lhs, rhs", [
    ("(x + 1/2)/(1 - c/2)", f"(x + 1/2)/(1 - {NEARLY_HALF}*c)"),
    ("(x + 1/2)/(1 - c/2)", f"(x + {NEARLY_HALF})/(1 - c/2)"),
    ("1/(1-x)", "1 + x + x^2"),
    ("p*x + (1-p)", "q*x + (1-q)"),
    ("(1 - x^3)/(1 - x)", "1 + x + x^2 + 10^(-30)*x^3"),
])
def test_unequal_rational_functions_are_not_equal(factory, lhs, rhs):
    assert not equal(factory, lhs, rhs, "x", "c")


@pytest.mark.parametrize("factory", [
    SympyPGF,
    SymenginePGF,
    ProdigyPGF,
])
def test_decimal_literals_are_exact(factory):
    assert not equal(factory, "1.00000000000000000001*x", "x", "x")
    assert not equal(factory, "(x + 1/2)/(1 - 0.50000000000000000001*c)", "(x + 1/2)/(1 - c/2)", "x", "c")


def test_sympy_floats_are_only_equal_if_identical():
    # floating-point arithmetic is inexact (1 is absorbed into 1e20), so it proves nothing
    assert sympy.Float(1e20) + 1 - sympy.Float(1e20) == 0
    x = sympy.Symbol("x", real=True)
    assert not provably_equal(sympy.Float("0.5") * x, x / 2)
    assert provably_equal(sympy.Float("0.5") * x, sympy.Float("0.5") * x)


def test_sympy_non_rational_equality_needs_a_symbolic_proof():
    p, x = sympy.symbols("p x", real=True)
    assert provably_equal(sympy.exp(p * (x - 1)), sympy.exp(p * x) * sympy.exp(-p))
    assert provably_equal(sympy.exp(p * x) / (1 - x), sympy.exp(p * x) * (1 + x / (1 - x)))
    # equal only for nonnegative x: not proven
    assert not provably_equal(sympy.sqrt(x ** 2), x)
    assert not provably_equal(sympy.exp(p * (x - 1)), sympy.exp(p * x) * sympy.exp(-p) + 10 ** -30)


def test_symengine_non_rational_subterms_are_indeterminates():
    assert equal(SymenginePGF, "exp(p*x)/(1-x)", "exp(p*x)*(1 + x/(1-x))", "x")
    assert not equal(SymenginePGF, "exp(p*x)/(1-x)", "exp(p*x)*(1 + x)", "x")


def test_sympy_parameters_get_no_global_assumptions():
    # Template parameters may be negative; the old Q.positive assumptions were wrong for them and
    # accumulated in the process-wide global_assumptions.
    before = set(global_assumptions)
    GeneratingFunction("a*x + b", "x")
    assert set(global_assumptions) == before


# Three-valued equality (#123): True = proven equal, False = proven different, None = neither.

@pytest.mark.parametrize("factory", ENGINES)
@pytest.mark.parametrize("lhs, rhs, verdict", [
    ("1/(1-x)", "1 + x/(1-x)", True),
    ("(x + 1/2)/(1 - c/2)", "(2*x + 1)/(2 - c)", True),
    ("1/(1-x)", "1 + x + x^2", False),
    ("(x + 1/2)/(1 - c/2)", f"(x + 1/2)/(1 - {NEARLY_HALF}*c)", False),
    ("p*x + (1-p)", "q*x + (1-q)", False),
    # non-rational: the difference -exp(x) is nonzero, but cancellation does not decide that
    ("x*exp(x)", "(x + 1)*exp(x)", None),
    ("exp(x)", "1/(1-x)", None),
])
def test_decide_equal(factory, lhs, rhs, verdict):
    assert factory.from_expr(lhs, "x", "c").decide_equal(factory.from_expr(rhs, "x", "c")) is verdict


@pytest.mark.parametrize("factory, verdict", [(SympyPGF, True), (SymenginePGF, None), (ProdigyPGF, None)])
def test_decide_equal_reports_unproven_equality_as_unknown(factory, verdict):
    # equal, but only SymPy's simplify proves it; the other engines must not call it different
    lhs = factory.from_expr("sin(c)^2 + cos(c)^2", "x", "c")
    assert lhs.decide_equal(factory.from_expr("1", "x", "c")) is verdict


@pytest.mark.parametrize("factory, verdict", [(SympyPGF, False), (SymenginePGF, False), (ProdigyPGF, False)])
def test_decide_equal_floats_prove_nothing(factory, verdict):
    # all engines parse decimals as exact rationals (GiNaC since prodigy-ppl-ginac 0.0.3), so 0.5 = 1/2
    # decides; only a genuine floating-point number proves nothing
    assert factory.from_expr("0.5*x", "x").decide_equal(factory.from_expr("x/3", "x")) is verdict
    x = sympy.Symbol("x")
    assert decide_equal(sympy.Float("0.5") * x, x / 3) is None


def test_decide_equal_never_refutes_with_irrational_constants():
    # sqrt(2) is not an indeterminate: cancellation over the rationals decides nothing about it
    assert decide_equal(sympy.sqrt(2) * sympy.sqrt(2), sympy.Integer(3)) is False  # SymPy evaluates to 2 first
    x = sympy.Symbol("x")
    assert decide_equal((1 + sympy.sqrt(2)) ** 2 * x, (3 + 2 * sympy.sqrt(2)) * x) is True
    assert decide_equal(sympy.pi * x, 3 * x) is None
    assert not is_rational_over_q(sympy.sqrt(2) * x)
    assert is_rational_over_q((x + sympy.Rational(1, 2)) / (1 - x) ** 2)


def test_decide_equal_strings_keeps_names_as_symbols():
    # E, S and I would otherwise be Euler's number, SymPy's singleton registry and the imaginary unit
    assert decide_equal_strings("E + S*I", "I*S + E", ["E", "S", "I"]) is True
    assert decide_equal_strings("E + S*I", "E", ["E", "S", "I"]) is False
    assert decide_equal_strings("x +", "x", ["x"]) is None
