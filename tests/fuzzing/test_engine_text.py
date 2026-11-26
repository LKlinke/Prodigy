"""
The text of expressions that the engines read (v0.6.1): ``str()`` of an arithmetic expression parses to the
expression itself. A fraction or negative literal as an operand is in parentheses; ``1/2 / 3/4`` meant
``((1/2)/3)/4``. The oracle is :func:`_parse_to_sympy` on the tree, which builds the SymPy expression without text.
"""
from decimal import Decimal
from fractions import Fraction

import pytest
import sympy
from hypothesis import given, settings
from hypothesis import strategies as st

from prodigy.distribution.generating_function import _parse_to_sympy
from prodigy.pgcl.ast import Binop, BinopExpr, FunctionCallExpr, NatLitExpr, RealLitExpr, Unop, UnopExpr, VarExpr

pytestmark = [pytest.mark.fuzz, pytest.mark.filterwarnings("ignore:Generating overly large repr")]

_NAMES = st.sampled_from(["x", "y", "n", "p"])
_NATS = st.integers(min_value=0, max_value=50).map(NatLitExpr)
_REALS = st.one_of(
    st.fractions(min_value=Fraction(-5), max_value=Fraction(5), max_denominator=12).map(RealLitExpr),
    st.decimals(min_value=Decimal("0"), max_value=Decimal("9.99"), places=2).map(lambda d: RealLitExpr(str(d))))
_OPERATORS = st.sampled_from([Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.DIVIDE, Binop.POWER])


def _arithmetic():
    leaves = st.one_of(_NAMES.map(VarExpr), _NATS, _REALS)
    return st.recursive(leaves, lambda inner: st.one_of(
        st.builds(BinopExpr, _OPERATORS, inner, inner),
        st.builds(lambda e: BinopExpr(Binop.TIMES, RealLitExpr("-1"), e), inner),
        st.builds(lambda name, arg: FunctionCallExpr(name, ([arg], {})), st.sampled_from(["exp", "log"]), inner)),
        max_leaves=8)


@settings(max_examples=300, deadline=None)
@given(_arithmetic())
def test_engines_read_the_text_of_an_expression_as_the_expression(expr):
    try:
        expected = _parse_to_sympy(expr)
    except (ZeroDivisionError, OverflowError, ValueError):
        return
    parsed = _parse_to_sympy(str(expr))
    assert parsed == expected or sympy.simplify(parsed - expected) == 0


def test_unary_minus_is_not_a_literal():
    """``-x`` stays ``-x``; only literals and compound operands get parentheses."""
    assert str(BinopExpr(Binop.TIMES, UnopExpr(Unop.MINUS, VarExpr("x")), RealLitExpr("1/2"))) == "-x * (1/2)"
