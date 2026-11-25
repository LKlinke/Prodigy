# pylint: disable=broad-exception-raised  # style as taken over from probably
from __future__ import annotations

import itertools
from decimal import Decimal
from enum import Enum, auto
from fractions import Fraction
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple, Union

import attr

from .ast import Node, Var


class ExprClass(Node):
    """
    Superclass for all expressions.
    See :obj:`Expr`.

    All expressions can be transformed into human-readable strings using the `str` operator.
    """
    def cast(self) -> Expr:
        """Cast to Expr. This is sometimes necessary to satisfy the type checker."""
        return self  # type: ignore[return-value]


@attr.s(repr=False)
class VarExpr(ExprClass):
    """A variable is an expression."""
    var: Var = attr.ib()

    def __attrs_post_init__(self) -> None:
        assert self.var not in {'true', 'false'}

    def __repr__(self) -> str:
        return f'VarExpr({repr(self.var)})'


@attr.s(repr=False)
class BoolLitExpr(ExprClass):
    """A bool literal is an expression."""
    value: bool = attr.ib()

    def __repr__(self) -> str:
        return f'BoolLitExpr({self.value})'


@attr.s(repr=False)
class NatLitExpr(ExprClass):
    """A natural number is an expression."""
    value: int = attr.ib()

    def __repr__(self) -> str:
        return f'NatLitExpr({self.value})'


def _validate_real_lit_value(_object: RealLitExpr, _attribute: object,
                             value: Union[Decimal, Fraction]) -> None:
    if not isinstance(value, Decimal) and not isinstance(value, Fraction):
        raise ValueError(
            f"Expected a Decimal or Fraction value, got: {value!r}")


def _parse_real_lit_expr(
        value: Union[str, Decimal, Fraction]) -> Union[Decimal, Fraction]:
    if isinstance(value, str):
        if "/" in value:
            return Fraction(value)
        else:
            res = Decimal(value)
            assert value == str(res)
            return res
    return value


@attr.s(repr=False, frozen=True)
class RealLitExpr(ExprClass):
    """
    A decimal literal (used for probabilities) is an expression.

    It is represented by either a :class:`Decimal` (created from decimal
    literals), or by a :class:`Fraction` (created from a fraction of natural numbers).

    Infinity is represented by ``Decimal('Infinity')``.

    .. warning::

        Note that the :class:`Decimal` representation is not exact under arithmetic operations.
        That is fine if it is used just as the representation of a decimal literal.
        For calculations, please use :meth:`to_fraction()`.
    """
    value: Union[Decimal,
                 Fraction] = attr.ib(validator=_validate_real_lit_value,
                                     converter=_parse_real_lit_expr)

    @staticmethod
    def infinity() -> RealLitExpr:
        """
        Create a new infinite value.

        .. doctest::

            >>> RealLitExpr.infinity().is_infinite()
            True
        """
        return RealLitExpr(Decimal('Infinity'))

    def is_infinite(self) -> bool:
        """
        Whether this expression represents infinity.
        """
        return isinstance(self.value, Decimal) and self.value.is_infinite()

    def to_fraction(self) -> Fraction:
        """
        Convert this value to a :class:`Fraction`.
        Throws an exception if the value :meth:`is_infinite()`!

        .. doctest::

            >>> expr = RealLitExpr("0.1")
            >>> expr.to_fraction()
            Fraction(1, 10)
        """
        assert not self.is_infinite()
        if isinstance(self.value, Fraction):
            return self.value
        return Fraction(self.value)

    def __repr__(self) -> str:
        return f'RealLitExpr("{str(self.value)}")'


class Unop(Enum):
    """What unary operator is it?"""
    NEG = auto()
    MINUS = auto()

    def __repr__(self) -> str:
        # pylint: disable=no-member
        return f'Unop.{self._name_}'


@attr.s
class UnopExpr(ExprClass):
    """A unary operator is an expression."""
    operator: Unop = attr.ib()
    expr: Expr = attr.ib()


class Binop(Enum):
    """What binary operator is it?"""
    OR = auto()
    AND = auto()
    LEQ = auto()
    LT = auto()
    GT = auto()
    GEQ = auto()
    EQ = auto()
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    POWER = auto()
    DIVIDE = auto()
    MODULO = auto()
    FLOOR_DIVIDE = auto()
    """Floor division ``//``, only in the ``.ppy`` syntax. Not implemented by the analysis yet."""

    def is_associative(self) -> bool:
        """Is this operator associative?"""
        return self in [Binop.OR, Binop.AND, Binop.PLUS, Binop.TIMES]

    def returns_boolean(self) -> bool:
        """Is the result of this operator a boolean (if not, it's a number)?"""
        return self in [
            Binop.OR, Binop.AND, Binop.LEQ, Binop.LT, Binop.GT, Binop.GEQ,
            Binop.EQ
        ]

    def __repr__(self) -> str:
        # pylint: disable=no-member
        return f'Binop.{self._name_}'

    def __str__(self) -> str:
        return ({
            Binop.OR: "||",
            Binop.AND: "&",
            Binop.LEQ: "<=",
            Binop.LT: "<",
            Binop.GT: ">",
            Binop.GEQ: ">=",
            Binop.EQ: "=",
            Binop.PLUS: "+",
            Binop.MINUS: "-",
            Binop.TIMES: "*",
            Binop.POWER: "^",
            Binop.DIVIDE: "/",
            Binop.MODULO: "%",
            Binop.FLOOR_DIVIDE: "//",
        })[self]


@attr.s
class BinopExpr(ExprClass):
    """A binary operator is an expression."""
    operator: Binop = attr.ib()
    lhs: Expr = attr.ib()
    rhs: Expr = attr.ib()

    @staticmethod
    def reduce(operator: Binop,
               iterable: Sequence[Expr],
               default: Optional[Expr] = None) -> Expr:
        """
        Builds a :class:`BinopExpr` using :func:`functools.reduce`.

        If the list is empty, ``default`` is returned.

        :raises AssertionError: if list is empty and ``default`` is ``None``.
        """
        gen = iter(iterable)
        try:
            peeked = next(gen)
            gen = itertools.chain([peeked], gen)
            return reduce(lambda x, y: BinopExpr(operator, x, y), iterable)
        except StopIteration:
            assert default is not None
            return default

    def flatten(self) -> List[Expr]:
        """
        Return a list of all recursive operands of the same operator.
        This really only makes sense if the operator is associative (see
        :meth:`Binop.is_associative`). Throws an error if the operator is not
        associative.

        .. doctest::

            >>> x = VarExpr('x')
            >>> times = BinopExpr(Binop.TIMES, x, x)
            >>> expr = BinopExpr(Binop.PLUS, BinopExpr(Binop.PLUS, x, times), x)
            >>> expr.flatten()
            [VarExpr('x'), BinopExpr(operator=Binop.TIMES, lhs=VarExpr('x'), rhs=VarExpr('x')), VarExpr('x')]
        """
        assert self.operator.is_associative()

        def flatten_expr(expr: Expr) -> List[Expr]:
            if isinstance(expr, BinopExpr) and expr.operator == self.operator:
                return flatten_expr(expr.lhs) + flatten_expr(expr.rhs)
            else:
                return [expr]

        return flatten_expr(self)


def _check_categorical_exprs(_self: CategoricalExpr, _attribute: object,
                             value: List[Tuple[Expr, RealLitExpr]]) -> None:
    probabilities = (prob.to_fraction() for _, prob in value)
    if sum(probabilities) != 1:
        raise ValueError("Probabilities need to sum up to 1!")


@attr.s
class CategoricalExpr(ExprClass):
    """
    Chooses one of a list of expressions to evaluate, where each expression has
    some assigned probability. The sum of probabilities must always be exactly one.

    It is represented by a `(expression, probability)` pair.

    As *monadic expressions* (see :ref:`expressions`), categorical choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    exprs: List[Tuple[Expr, RealLitExpr]] = attr.ib(
        validator=_check_categorical_exprs)

    def distribution(self) -> List[Tuple[RealLitExpr, Expr]]:
        """
        Return the distribution of possible values as a list along with
        probabilities.
        """
        return [(prob, expr) for expr, prob in self.exprs]


@attr.s
class SubstExpr(ExprClass):
    """
    A substition expression that applies a mapping from variables to expressions
    to the target expression.

    An important invariant is that all substitutions must be well-typed, i.e.
    assign expressions to variables of the same type. Thus the expression with
    the substitutions applied has the same type as the expression before.

    It is not available in the program language, but can be generated by
    program transformations to represent expectations, such as the weakest
    preexpectation of a program.

    Substitutions can be applied using the :mod:`prodigy.language.substitute`
    module.
    """
    subst: Dict[Var, Expr] = attr.ib()
    expr: Expr = attr.ib()


@attr.s
class FunctionCallExpr(ExprClass):
    """
    Performs the body of the corresponding function and then samples a value
    from the resulting distribution (according to the function's `return`
    statement).
    """

    function: Var = attr.ib()
    """The name of the function that is called."""

    params: Tuple[List[Expr], Dict[Var, Expr]] = attr.ib()
    """
    The parameters of the function call. First come the positional parameters,
    then the named ones.
    """


@attr.s
class InferExpr(ExprClass):
    function: FunctionCallExpr = attr.ib()
    """The function call to infer a distribution from"""


@attr.s
class SampleExpr(ExprClass):
    dist: Var = attr.ib()
    """The distribution from which to sample a value"""


@attr.s
class ProbabilityExpr(ExprClass):
    """
    ``Pr[d == k]`` inside a program: the probability that the distribution of the ``dist``
    variable ``d`` gives a value that satisfies the condition, a ``real`` number. The condition
    compares ``d`` with a natural number (``==``, ``<``, ``<=``, ``>``, ``>=``); the type checker
    enforces this form.
    """
    cond: Expr = attr.ib()
    """The condition, e.g. ``d = 3``"""


@attr.s
class ExpectationExpr(ExprClass):
    """
    ``Ex[d]`` inside a program: the expected value of the distribution of the ``dist`` variable
    ``d``, a ``real`` number that may be infinite.
    """
    expr: Expr = attr.ib()
    """The ``dist`` variable as a :class:`VarExpr`"""



Expr = Union[VarExpr, BoolLitExpr, NatLitExpr, RealLitExpr, UnopExpr,
             BinopExpr, CategoricalExpr, SubstExpr, FunctionCallExpr,
             InferExpr, SampleExpr, ProbabilityExpr, ExpectationExpr]
"""Union type for all expression objects. See :class:`ExprClass` for use with isinstance."""
