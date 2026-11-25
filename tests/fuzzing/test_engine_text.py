"""
The engine text of expressions (P5a, T6): :func:`~prodigy.language.engine_text.to_engine_text` writes exactly the
text the engines read before ``str()`` of the AST became ``.ppy`` (:func:`_legacy_text`, a frozen copy of the old
``__str__`` methods).
"""
from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prodigy.language import (Binop, BinopExpr, BoolLitExpr, CategoricalExpr, ExpectationExpr, FunctionCallExpr,
                              InferExpr, NatLitExpr, ProbabilityExpr, RealLitExpr, SampleExpr, SubstExpr, Unop,
                              UnopExpr, VarExpr)
from prodigy.language.engine_text import to_engine_text

pytestmark = pytest.mark.fuzz

_OPERATORS = {Binop.OR: "||", Binop.AND: "&", Binop.LEQ: "<=", Binop.LT: "<", Binop.GT: ">", Binop.GEQ: ">=",
              Binop.EQ: "=", Binop.PLUS: "+", Binop.MINUS: "-", Binop.TIMES: "*", Binop.POWER: "^",
              Binop.DIVIDE: "/", Binop.MODULO: "%", Binop.FLOOR_DIVIDE: "//"}


# pylint: disable=too-many-return-statements
def _legacy_text(expr) -> str:
    """``str()`` of an expression before P5a PR 2 (the ``__str__`` methods of ``ast/expressions.py``)."""
    def parens(inner) -> str:
        simple = (VarExpr, BoolLitExpr, NatLitExpr, RealLitExpr, UnopExpr, ProbabilityExpr, ExpectationExpr)
        return _legacy_text(inner) if isinstance(inner, simple) else f"({_legacy_text(inner)})"

    if isinstance(expr, VarExpr):
        return expr.var
    if isinstance(expr, BoolLitExpr):
        return str(expr.value).lower()
    if isinstance(expr, (NatLitExpr, RealLitExpr)):
        return str(expr.value)
    if isinstance(expr, UnopExpr):
        return f"not {parens(expr.expr)}" if expr.operator == Unop.NEG else f"-{parens(expr.expr)}"
    if isinstance(expr, BinopExpr):
        if expr.operator == Binop.POWER:
            return f"({_legacy_text(expr.lhs)}) ^ ({_legacy_text(expr.rhs)})"
        return f"{parens(expr.lhs)} {_OPERATORS[expr.operator]} {parens(expr.rhs)}"
    if isinstance(expr, CategoricalExpr):
        return " + ".join(f"{parens(e)} : {parens(p)}" for e, p in expr.exprs)
    if isinstance(expr, SubstExpr):
        substs = ", ".join(f"{key}/{_legacy_text(value)}" for key, value in expr.subst.items())
        return f"({_legacy_text(expr.expr)})[{substs}]"
    if isinstance(expr, FunctionCallExpr):
        params = ", ".join([_legacy_text(p) for p in expr.params[0]]
                           + [f"{var} := {_legacy_text(e)}" for var, e in expr.params[1].items()])
        return f"{expr.function}({params})"
    if isinstance(expr, InferExpr):
        return f"infer {{ {_legacy_text(expr.function)} }}"
    if isinstance(expr, SampleExpr):
        return f"sample {{ {expr.dist} }}"
    if isinstance(expr, ProbabilityExpr):
        return f"Pr[{_legacy_text(expr.cond)}]"
    assert isinstance(expr, ExpectationExpr)
    return f"Ex[{_legacy_text(expr.expr)}]"


_NAMES = st.sampled_from(["x", "y", "n", "p", "E", "S", "pi"])
_NATS = st.integers(min_value=0, max_value=50).map(NatLitExpr)
_REALS = st.one_of(
    st.fractions(min_value=Fraction(-5), max_value=Fraction(5), max_denominator=12).map(RealLitExpr),
    st.decimals(min_value=Decimal("0"), max_value=Decimal("9.99"), places=2).map(lambda d: RealLitExpr(str(d))))
_ARITHMETIC_OPS = st.sampled_from([Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.DIVIDE, Binop.POWER])
_ALL_OPS = st.sampled_from(list(_OPERATORS))


def _arithmetic():
    leaves = st.one_of(_NAMES.map(VarExpr), _NATS, _REALS)
    return st.recursive(leaves, lambda inner: st.one_of(
        st.builds(BinopExpr, _ARITHMETIC_OPS, inner, inner),
        st.builds(lambda e: UnopExpr(Unop.MINUS, e), inner),
        st.builds(lambda name, args: FunctionCallExpr(name, (args, {})), st.sampled_from(["exp", "log", "sqrt"]),
                  st.lists(inner, min_size=1, max_size=1))), max_leaves=8)


def _any_expr():
    leaves = st.one_of(_NAMES.map(VarExpr), _NATS, _REALS, st.booleans().map(BoolLitExpr))
    return st.recursive(leaves, lambda inner: st.one_of(
        st.builds(BinopExpr, _ALL_OPS, inner, inner),
        st.builds(UnopExpr, st.sampled_from([Unop.NEG, Unop.MINUS]), inner),
        st.builds(lambda name, args, named: FunctionCallExpr(name, (args, named)), st.sampled_from(["f", "unif"]),
                  st.lists(inner, max_size=2), st.dictionaries(st.sampled_from(["a", "b"]), inner, max_size=1)),
        st.builds(lambda e, name, value: SubstExpr({name: value}, e), inner, _NAMES, inner),
        st.builds(lambda e: ProbabilityExpr(BinopExpr(Binop.EQ, VarExpr("d"), e)), _NATS),
        st.just(ExpectationExpr(VarExpr("d"))),
        st.just(SampleExpr("d")),
        st.builds(lambda args: InferExpr(FunctionCallExpr("f", (args, {}))), st.lists(inner, max_size=2)),
        st.builds(lambda a, b: CategoricalExpr([(a, RealLitExpr("1/3")), (b, RealLitExpr("2/3"))]), inner, inner),
    ), max_leaves=10)


@settings(max_examples=400, deadline=None)
@given(_any_expr())
def test_engine_text_is_the_text_the_engines_read(expr):
    assert to_engine_text(expr) == _legacy_text(expr) == str(expr)
