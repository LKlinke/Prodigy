"""
-------------------------
Expression Simplification
-------------------------

Expressions and expectations become ugly after backward calculation?
Your program has dumb expressions a toddler could simplify?
This is the right place for you!
"""

from typing import Dict, Union, Optional

import attr

from probably.pgcl.ast.expressions import expr_str_parens

from probably.pgcl.ast import *
from probably.pgcl.typechecker.check import CheckFail, get_type
from probably.pgcl.ast.walk import Walk, walk_expr
from probably.analysis.backward.wp import ExpectationTransformer


def simplifying_and(lhs: Expr, rhs: Expr) -> Expr:
    """
    Combine with `Binop.AND`, but simplify when one operand is `BoolLitExpr(True)`.

    .. doctest::

        >>> simplifying_and(BoolLitExpr(True), VarExpr('x'))
        VarExpr('x')
    """
    if lhs == BoolLitExpr(True):
        return rhs
    if rhs == BoolLitExpr(True):
        return lhs
    return BinopExpr(Binop.AND, lhs, rhs)


def simplifying_plus(lhs: Expr, rhs: Expr) -> Expr:
    """
    Combine with ``Binop.PLUS``, but simplify when one operand is
    ``NatLitExpr(0)`` or both operands are ``NatLitExpr``.

    .. doctest::

        >>> simplifying_plus(NatLitExpr(0), VarExpr('x'))
        VarExpr('x')
        >>> simplifying_plus(NatLitExpr(5), NatLitExpr(2))
        NatLitExpr(7)
    """
    zero = NatLitExpr(0)
    if lhs == zero:
        return rhs
    elif rhs == zero:
        return lhs
    if isinstance(lhs, NatLitExpr) and isinstance(rhs, NatLitExpr):
        return NatLitExpr(lhs.value + rhs.value)
    return BinopExpr(Binop.PLUS, lhs, rhs)


def simplifying_times(lhs: Expr, rhs: Expr) -> Expr:
    """
    Combine with ``Binop.TIMES``, but simplify when one operand is ``RealLitExpr("1.0")``.

    .. doctest::

        >>> simplifying_times(RealLitExpr("1.0"), VarExpr('x'))
        VarExpr('x')
    """
    one = RealLitExpr("1.0")
    if lhs == one:
        return rhs
    if rhs == one:
        return lhs
    return BinopExpr(Binop.TIMES, lhs, rhs)


def simplifying_subst(subst: Dict[Var, Expr], expr: Expr) -> Expr:
    """
    Only create a ``SubstExpr`` around `expr` if `expr` is not a simple literal.

    .. doctest::

        >>> simplifying_subst(dict(), RealLitExpr("1.0"))
        RealLitExpr("1.0")
    """
    if isinstance(expr, (BoolLitExpr, RealLitExpr, NatLitExpr)):
        return expr
    return SubstExpr(subst, expr)


def simplifying_neg(expr: Expr) -> Expr:
    """
    Negate the given Boolean expression. If we already have a negation, unwrap it.

    .. doctest::

        >>> simplifying_neg(UnopExpr(Unop.NEG, VarExpr("x")))
        VarExpr('x')
        >>> simplifying_neg(VarExpr("x"))
        UnopExpr(operator=Unop.NEG, expr=VarExpr('x'))
    """
    if isinstance(expr, UnopExpr) and expr.operator == Unop.NEG:
        return expr.expr
    return UnopExpr(Unop.NEG, expr)


@attr.s
class SnfExpectationTransformerProduct:
    """
    A :class:`SnfExpectationTransformer` represents a list of
    :class:`SnfExpectationTransformerProduct`.

    The multiplication operator (``*``) is implemented, and returns the product
    of two objects. However, only at most one of the operands must have a
    non-empty set of substitutions.

    .. doctest::

        >>> a = SnfExpectationTransformerProduct.from_iverson(BoolLitExpr(True))
        >>> b = SnfExpectationTransformerProduct(guard=BoolLitExpr(True), prob=RealLitExpr("5.0"), subst=None, ticks=TickExpr(NatLitExpr(1)))
        >>> c = SnfExpectationTransformerProduct(guard=BoolLitExpr(False), prob=RealLitExpr("2.0"), subst=None, ticks=TickExpr(NatLitExpr(5)))
        >>> print(a * b)
        [true] * 5.0 * tick(1)
        >>> print(a * b * c)
        [false] * (5.0 * 2.0) * tick(6)
        >>> print(SnfExpectationTransformerProduct(guard=BoolLitExpr(True), prob=RealLitExpr("2.0"), subst={"x": VarExpr("y")}, ticks=TickExpr(NatLitExpr(0))))
        [true] * 2.0 * (𝑋)[x/y]
    """

    guard: Expr = attr.ib()
    """The boolean guard expression."""

    prob: Expr = attr.ib()
    """
    The probability without any Iverson brackets or
    :class:`probably.pgcl.ast.TickExpr`. It has types
    :class:`probably.pgcl.ast.RealType` or :class:`probably.pgcl.ast.NatType`.
    """

    subst: Optional[Dict[Var, Expr]] = attr.ib()
    """
    Representing the (optional) post-expectation and substitutions applied to it.
    """

    ticks: TickExpr = attr.ib()
    """Sum of ticks in this part."""
    @staticmethod
    def from_iverson(guard: Expr) -> 'SnfExpectationTransformerProduct':
        """Create a new value ``guard * 1.0 * tick(1)``."""
        return SnfExpectationTransformerProduct(guard=guard,
                                                prob=RealLitExpr("1.0"),
                                                subst=None,
                                                ticks=TickExpr(NatLitExpr(0)))

    @staticmethod
    def from_prob(expr: Expr) -> 'SnfExpectationTransformerProduct':
        """Create a new value from just the probability."""
        return SnfExpectationTransformerProduct(guard=BoolLitExpr(True),
                                                prob=expr,
                                                subst=None,
                                                ticks=TickExpr(NatLitExpr(0)))

    @staticmethod
    def from_subst(
            subst: Dict[Var, Expr]) -> 'SnfExpectationTransformerProduct':
        """Create a new value from the substitution."""
        return SnfExpectationTransformerProduct(guard=BoolLitExpr(True),
                                                prob=RealLitExpr("1.0"),
                                                subst=subst,
                                                ticks=TickExpr(NatLitExpr(0)))

    @staticmethod
    def from_ticks(ticks: TickExpr) -> 'SnfExpectationTransformerProduct':
        """Create a new value from ticks."""
        return SnfExpectationTransformerProduct(guard=BoolLitExpr(True),
                                                prob=RealLitExpr("1.0"),
                                                subst=None,
                                                ticks=ticks)

    def substitute(
            self, subst: Dict[Var,
                              Expr]) -> 'SnfExpectationTransformerProduct':
        """
        Apply the substitution to this value.

        This object's :data:`subst` must be empty!
        """
        assert self.subst is None
        return SnfExpectationTransformerProduct(
            guard=simplifying_subst(subst, self.guard),
            prob=SubstExpr(subst, self.prob),
            subst=subst,
            ticks=TickExpr(simplifying_subst(subst, self.ticks.expr)))

    def __mul__(
        self, other: 'SnfExpectationTransformerProduct'
    ) -> 'SnfExpectationTransformerProduct':
        assert not (self.subst is not None and other.subst
                    is not None), "only one operand may include substitutions"
        subst = self.subst if self.subst is not None else other.subst
        return SnfExpectationTransformerProduct(
            guard=simplifying_and(self.guard, other.guard),
            prob=simplifying_times(self.prob, other.prob),
            subst=subst,
            ticks=TickExpr(simplifying_plus(self.ticks.expr,
                                            other.ticks.expr)))

    def __str__(self) -> str:
        terms = []
        if self.subst is not None:
            subst = SubstExpr(self.subst, VarExpr('𝑋'))
            terms.append(str(subst))
        if not self.ticks.is_zero():
            terms.append(str(self.ticks))
        if len(terms) == 2:
            value = f"({terms[0]} + {terms[1]})"
        elif len(terms) == 1:
            value = str(terms[0])
        else:
            value = ""
        terms = [f"[{self.guard}]", expr_str_parens(self.prob)]
        if len(value) != 0:
            terms.append(value)
        return " * ".join(terms)


@attr.s(init=False, repr=False)
class SnfExpectationTransformer:
    """
    An expectation transformer
    (:class:`probably.pgcl.backward.ExpectationTransformer`) is in *summation normal
    form (SNF)* if it is a sum of products
    (:class:`SnfExpectationTransformerProduct`). This class can be accessed just
    like a list over :class:`SnfExpectationTransformerProduct`.

    You can create a summation normal form expectation transformer from an
    :class:`Expr` using :func:`normalize_expectation_transformer`.

    The multiplication operator is implemented (``*``), but only one operand may
    include substitutions. This is always the case for transformers generated
    from weakest pre-expectation semantics (see :mod:`probably.pgcl.backward`).

    .. doctest::

        >>> a = SnfExpectationTransformerProduct.from_iverson(BoolLitExpr(True))
        >>> b = SnfExpectationTransformerProduct.from_prob(RealLitExpr("0.5"))
        >>> c = SnfExpectationTransformerProduct.from_subst({"x": VarExpr("y")})
        >>> print(SnfExpectationTransformer([a, b, c]))
        λ𝑋. [true] * 1.0 + [true] * 0.5 + [true] * 1.0 * (𝑋)[x/y]
    """

    _values: List[SnfExpectationTransformerProduct] = attr.ib()

    def __init__(self, values: List[SnfExpectationTransformerProduct]):
        self._values = values

        # To catch bugs, verify that the probability expressions do not contain Iverson brackets.
        for value in values:
            for child_ref in walk_expr(Walk.DOWN, Mut.alloc(value.prob)):
                child = child_ref.val
                assert not (isinstance(child, UnopExpr)
                            and child.operator == Unop.IVERSON)

    @staticmethod
    def from_iverson(guard: Expr) -> 'SnfExpectationTransformer':
        """Create a new value ``guard * 1.0 * tick(1)``."""
        return SnfExpectationTransformer(
            [SnfExpectationTransformerProduct.from_iverson(guard)])

    @staticmethod
    def from_prob(expectation: Expr) -> 'SnfExpectationTransformer':
        """Create a new value with just the probability."""
        return SnfExpectationTransformer(
            [SnfExpectationTransformerProduct.from_prob(expectation)])

    @staticmethod
    def from_subst(subst: Dict[Var, Expr]) -> 'SnfExpectationTransformer':
        """Create a new value with just the substitution."""
        return SnfExpectationTransformer(
            [SnfExpectationTransformerProduct.from_subst(subst)])

    @staticmethod
    def from_ticks(ticks: TickExpr) -> 'SnfExpectationTransformer':
        """Create a new value with just the ticks value."""
        return SnfExpectationTransformer(
            [SnfExpectationTransformerProduct.from_ticks(ticks)])

    def substitute(self, subst: Dict[Var,
                                     Expr]) -> 'SnfExpectationTransformer':
        """Apply the subtitution to this expectation."""
        return SnfExpectationTransformer(
            [value.substitute(subst) for value in self])

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __add__(
            self,
            other: 'SnfExpectationTransformer') -> 'SnfExpectationTransformer':
        return SnfExpectationTransformer(self._values + other._values)

    def __mul__(
            self,
            other: 'SnfExpectationTransformer') -> 'SnfExpectationTransformer':
        return SnfExpectationTransformer(
            [lhs * rhs for lhs in self for rhs in other])

    def __repr__(self) -> str:
        return repr(self._values)

    def __str__(self) -> str:
        terms = " + ".join(map(str, self._values))
        return f"λ𝑋. {terms}"


def normalize_expectation_transformer(
    program: Program, expectation_transformer: ExpectationTransformer
) -> Union[SnfExpectationTransformer, CheckFail]:
    """
    Given a well-typed expectation transformer, return a
    :class:`SnfExpectationTransformer`, i.e. an expectation transformer in
    *summation normal form* (SNF).

    Important: A guard statement may only appear in sums and products, but not
    anywhere in the operands of a minus.

    Simplifying methods are used whenever possible: :func:`simplifying_and`,
    :func:`simplifying_plus`, :func:`simplifying_times`, and
    :func:`simplifying_subst`.

    See :func:`normalize_expectation` for a lot of examples.
    """
    def recurse(expr: Expr) -> Union[SnfExpectationTransformer, CheckFail]:
        if isinstance(expr, BinopExpr):
            if expr.operator == Binop.PLUS:
                lhs = recurse(expr.lhs)
                if isinstance(lhs, CheckFail):
                    return lhs

                # addition with a TickExpr is handled explicitely to get a nicer
                # result. without the following branch (it is entirely an
                # optimization), we'd get something like: `[x < n] * 1/2 *
                # (𝑋)[r/0, x/x + 0] + [x < n] * 1/2 * tick(1)` instead of `[x <
                # n] * 1/2 * ((𝑋)[r/0, x/x + 0] + tick(1))`.
                if isinstance(expr.rhs, TickExpr):
                    for value in lhs:
                        value.ticks = TickExpr(
                            simplifying_plus(value.ticks.expr, expr.rhs.expr))
                    return lhs
                else:
                    rhs = recurse(expr.rhs)
                    if isinstance(rhs, CheckFail):
                        return rhs
                return lhs + rhs
            if expr.operator == Binop.TIMES:
                lhs = recurse(expr.lhs)
                if isinstance(lhs, CheckFail):
                    return lhs
                rhs = recurse(expr.rhs)
                if isinstance(rhs, CheckFail):
                    return rhs
                return lhs * rhs

        if isinstance(expr, UnopExpr):
            if expr.operator == Unop.IVERSON:
                return SnfExpectationTransformer.from_iverson(expr.expr)
            elif expr.operator == Unop.NEG:
                raise Exception(
                    "negation operator is not applicable to expectations")

        if isinstance(expr, SubstExpr):
            assert isinstance(
                expr.expr,
                VarExpr) and expr.expr.var == expectation_transformer.variable
            return SnfExpectationTransformer.from_subst(expr.subst)

        if isinstance(expr, TickExpr):
            return SnfExpectationTransformer.from_ticks(expr)

        # the last remaining cases are only VarExprs and literals.
        expr_type = get_type(program, expr, check=False)
        if isinstance(expr_type, CheckFail):
            return expr_type
        if isinstance(expr_type, (RealType, NatType)):
            return SnfExpectationTransformer.from_prob(expr)

        raise Exception("unreachable")

    expectation_transformer.substitute()
    return recurse(expectation_transformer.expectation)


def normalize_expectation(
        program: Program,
        expectation: Expr) -> Union[SnfExpectationTransformer, CheckFail]:
    """
    Specialized version of :func:`normalize_expectation_transformer`.

    Translate this expectation into summation normal form. The result is a
    :class:`SnfExpectationTransformer` where
    :data:`SnfExpectationTransformerProduct.subst` are always empty.

    .. doctest::

        >>> from .ast import *
        >>> from .parser import parse_pgcl, parse_expectation
        >>> program = parse_pgcl("bool x; nat c")

        >>> expectation = parse_expectation("[true] * ([false] + 1.0)")
        >>> print(normalize_expectation(program, expectation))
        λ𝑋. [false] * 1.0 + [true] * 1.0

        >>> prob = RealLitExpr("10")
        >>> print(normalize_expectation(program, BinopExpr(Binop.TIMES, prob, prob)))
        λ𝑋. [true] * (10 * 10)

        >>> normalize_expectation(program, BinopExpr(Binop.TIMES, UnopExpr(Unop.IVERSON, VarExpr('x')), prob))
        [SnfExpectationTransformerProduct(guard=VarExpr('x'), prob=RealLitExpr("10"), subst=None, ticks=TickExpr(expr=NatLitExpr(0)))]

        >>> expectation = parse_expectation("[c < 3] * c")
        >>> print(normalize_expectation(program, expectation))
        λ𝑋. [c < 3] * c

        >>> expectation = parse_expectation("1 - 5")
        >>> print(normalize_expectation(program, expectation))
        λ𝑋. [true] * (1 - 5)

        >>> expectation = parse_expectation("c - 0.5")
        >>> print(normalize_expectation(program, expectation))
        λ𝑋. [true] * (c - 0.5)
    """
    variable = '𝑋'
    assert variable not in program.variables, "can't use 𝑋 as a variable name"
    transformer = ExpectationTransformer(variable, expectation)
    return normalize_expectation_transformer(program, transformer)
