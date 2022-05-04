import itertools
from collections.abc import Sequence
from decimal import Decimal
from enum import Enum, auto
from fractions import Fraction
from functools import reduce
from typing import Optional, List, Any, Union, Tuple, Dict
import attr
from abc import abstractmethod

from . import Var
from .ast import Node

""" Helper methods for dealing with expressions."""


def _validate_real_lit_value(_object: 'RealLitExpr', _attribute: Any,
                             value: Any):
    if not isinstance(value, Decimal) and not isinstance(value, Fraction):
        raise ValueError(
            f"Expected a Decimal or Fraction value, got: {value!r}")


def _parse_real_lit_expr(value: Union[str, Decimal, Fraction]) -> Union[Decimal, Fraction]:
    if isinstance(value, str):
        if "/" in value:
            return Fraction(value)
        else:
            res = Decimal(value)
            assert value == str(res)
            return res
    return value


def expr_str_parens(expr: 'ExprClass') -> str:
    """Wrap parentheses around an expression, but not for simple expressions."""
    if isinstance(expr,
                  (VarExpr, BoolLitExpr, NatLitExpr, RealLitExpr, UnopExpr)):
        return str(expr)
    else:
        return f'({expr})'


def _check_categorical_exprs(_self: "CategoricalExpr", _attribute: Any,
                             value: List[Tuple["Expr", 'RealLitExpr']]):
    probabilities = (prob.to_fraction() for _, prob in value)
    if sum(probabilities) != 1:
        raise ValueError("Probabilities need to sum up to 1!")


""" Operator Types """


class Binop(Enum):
    """What binary operator is it?"""
    OR = auto()
    AND = auto()
    LEQ = auto()
    LE = auto()
    GE = auto()
    GEQ = auto()
    EQ = auto()
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    POWER = auto()
    DIVIDE = auto()
    MODULO = auto()

    def is_associative(self) -> bool:
        """Is this operator associative?"""
        return self in [Binop.OR, Binop.AND, Binop.PLUS, Binop.TIMES]

    def __repr__(self) -> str:
        # pylint: disable=no-member
        return f'Binop.{self._name_}'

    def __str__(self) -> str:
        return ({
            Binop.OR: "||",
            Binop.AND: "&",
            Binop.LEQ: "<=",
            Binop.LE: "<",
            Binop.GE: ">",
            Binop.GEQ: ">=",
            Binop.EQ: "=",
            Binop.PLUS: "+",
            Binop.MINUS: "-",
            Binop.TIMES: "*",
            Binop.POWER: "^",
            Binop.DIVIDE: "/",
            Binop.MODULO: "%"
        })[self]


class Unop(Enum):
    """What unary operator is it?"""
    NEG = auto()
    IVERSON = auto()

    def __repr__(self) -> str:
        # pylint: disable=no-member
        return f'Unop.{self._name_}'


""" The types of Expressions."""


class ExprClass(Node):
    """
    Superclass for all expressions.
    See :obj:`Expr`.

    All expressions can be transformed into human-readable strings using the `str` operator.
    """

    def cast(self) -> 'Expr':
        """Cast to Expr. This is sometimes necessary to satisfy the type checker."""
        return self  # type: ignore

    @abstractmethod
    def __str__(self) -> str:
        """
        Convert this expression to corresponding source code in pGCL.
        """


""" Atomic Expressions."""


@attr.s(repr=False)
class VarExpr(ExprClass):
    """A variable is an expression."""
    var: Var = attr.ib()
    is_parameter: bool = attr.ib(default=False)

    def __str__(self) -> str:
        return self.var

    def __repr__(self) -> str:
        if self.is_parameter:
            return f"ParamExpr({repr(self.var)})"
        else:
            return f'VarExpr({repr(self.var)})'


@attr.s(repr=False)
class BoolLitExpr(ExprClass):
    """A bool literal is an expression."""
    value: bool = attr.ib()

    def __str__(self) -> str:
        return str(self.value).lower()

    def __repr__(self) -> str:
        return f'BoolLitExpr({self.value})'


@attr.s(repr=False)
class NatLitExpr(ExprClass):
    """A natural number is an expression."""
    value: int = attr.ib()

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f'NatLitExpr({self.value})'


@attr.s(repr=False, frozen=True)
class RealLitExpr(ExprClass):
    """
    A real number literals. Used for both probabilities and for values in the
    program if real number variables are enabled (see
    :py:data:`ProgramConfig.allow_real_vars`).

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
    def infinity() -> 'RealLitExpr':
        """
        Create a new infinite value.

        .. doctest::

            >>> RealLitExpr.infinity().is_infinite()
            True
        """
        return RealLitExpr(Decimal('Infinity'))

    def is_infinite(self):
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

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f'RealLitExpr("{str(self.value)}")'


""" Non-Atomic Expressions."""


@attr.s
class UnopExpr(ExprClass):
    """A unary operator is an expression."""
    operator: Unop = attr.ib()
    expr: 'Expr' = attr.ib()

    def __str__(self) -> str:
        if self.operator == Unop.NEG:
            return f'not {expr_str_parens(self.expr)}'
        elif self.operator == Unop.IVERSON:
            return f'[{self.expr}]'
        raise Exception("Invalid Unop operator")


@attr.s
class BinopExpr(ExprClass):
    """A binary operator is an expression."""
    operator: Binop = attr.ib()
    lhs: 'Expr' = attr.ib()
    rhs: 'Expr' = attr.ib()

    @staticmethod
    def reduce(operator: Binop,
               iterable: Sequence['Expr'],
               default: Optional['Expr'] = None) -> 'Expr':
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

    def flatten(self) -> List['Expr']:
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

    def __str__(self) -> str:
        if self.operator == Binop.POWER:
            return f'({self.lhs}) {self.operator} ({self.rhs})'
        return f'{expr_str_parens(self.lhs)} {self.operator} {expr_str_parens(self.rhs)}'


""" Distribution Expressions."""


@attr.s
class DUniformExpr(ExprClass):
    """
    Chooses a random integer within the (inclusive) interval.

    As *monadic expressions* (see :ref:`expressions`), uniform choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    start: 'Expr' = attr.ib()
    end: 'Expr' = attr.ib()

    def distribution(self) -> List[Tuple[RealLitExpr, NatLitExpr]]:
        r"""
        Return the distribution of possible values as a list along with
        probabilities. For the uniform distribution, all probabilites are equal
        to :math:`\frac{1}{\text{end} - \text{start} + 1}`.
        """
        if isinstance(self.start, NatLitExpr) and isinstance(self.end, NatLitExpr):
            width = self.end.value - self.start.value + 1
            prob = RealLitExpr(Fraction(1, width))
            return [(prob, NatLitExpr(i))
                    for i in range(self.start.value, self.end.value + 1)]
        else:
            NotImplementedError("Parameters not implemented yet.")

    def __str__(self) -> str:
        return f'unif({expr_str_parens(self.start)}, {expr_str_parens(self.end)})'



@attr.s
class CUniformExpr(ExprClass):
    """
    Chooses a random real number within the (inclusive) interval.

    As *monadic expressions* (see :ref:`expressions`), uniform choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    start: 'Expr' = attr.ib()
    end: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'unif({expr_str_parens(self.start)}, {expr_str_parens(self.end)})'


@attr.s
class BernoulliExpr(ExprClass):
    """
    Chooses a random bernoulli distributed integer.

    As *monadic expressions* (see :ref:`expressions`), geometric choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    param: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'bernoulli({expr_str_parens(self.param)})'


@attr.s
class GeometricExpr(ExprClass):
    """
    Chooses a random geometrically distributed integer.

    As *monadic expressions* (see :ref:`expressions`), geometric choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    param: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'geometric({expr_str_parens(self.param)})'


@attr.s
class PoissonExpr(ExprClass):
    """
    Chooses a random poisson distributed integer.

    As *monadic expressions* (see :ref:`expressions`), geometric choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    param: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'poisson({expr_str_parens(self.param)})'


@attr.s
class LogDistExpr(ExprClass):
    """
    Chooses a random logarithmically distributed integer.
    """
    param: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'logdist({expr_str_parens(self.param)})'


@attr.s
class BinomialExpr(ExprClass):
    """
    Chooses a random logarithmically distributed integer.

    As *monadic expressions* (see :ref:`expressions`), geometric choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    n: 'Expr' = attr.ib()
    p: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'binomial({expr_str_parens(self.n)}, {expr_str_parens(self.p)})'


@attr.s
class IidSampleExpr(ExprClass):
    """ Independently sampling from identical distributions"""
    sampling_dist: 'Expr' = attr.ib()
    variable: VarExpr = attr.ib()

    def __str__(self) -> str:
        return f"iid({self.sampling_dist}, {self.variable})"


""" Other Expressions. """


@attr.s
class SubstExpr(ExprClass):
    """
    A substition expression that applies a mapping from variables to expressions
    to the target expression.

    An important invariant is that all substitutions must be well-typed, i.e.
    assign expressions to variables of the same type. Thus the expression with
    the substitutions applied has the same type as the expression before.

    It is not available in the pGCL program language, but can be generated by
    program transformations to represent expectations, such as the weakest
    preexpectation of a program (see :mod:`prodigy.pgcl.backward`).

    Substitutions can be applied using the :mod:`prodigy.pgcl.substitute`
    module.
    """
    subst: Dict[Var, 'Expr'] = attr.ib()
    expr: 'Expr' = attr.ib()

    def __str__(self) -> str:
        substs = ", ".join(
            (f'{key}/{value}' for key, value in self.subst.items()))
        return f'({self.expr})[{substs}]'


@attr.s
class TickExpr(ExprClass):
    """
    Generated only by the weakest pre-expectation semantics of
    :class:`TickInstr`.
    """
    expr: 'Expr' = attr.ib()

    def is_zero(self) -> bool:
        """Whether this expression represents exactly zero ticks."""
        return self.expr == NatLitExpr(0)

    def __str__(self) -> str:
        return f"tick({self.expr})"


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
    exprs: List[Tuple['Expr', RealLitExpr]] = attr.ib(
        validator=_check_categorical_exprs)

    def distribution(self) -> List[Tuple[RealLitExpr, 'Expr']]:
        r"""
        Return the distribution of possible values as a list along with
        probabilities.
        """
        return [(prob, expr) for expr, prob in self.exprs]

    def __str__(self) -> str:
        return " + ".join((f"{expr_str_parens(expr)} : {expr_str_parens(prob)}"
                           for expr, prob in self.exprs))


""" A type combining all sampling expressions"""
DistrExpr = Union[DUniformExpr, CUniformExpr, BernoulliExpr, GeometricExpr, PoissonExpr, LogDistExpr, BinomialExpr,
                  IidSampleExpr]

"""Union type for all expression objects. See :class:`ExprClass` for use with isinstance."""
Expr = Union[VarExpr, BoolLitExpr, NatLitExpr, RealLitExpr,
             UnopExpr, BinopExpr, CategoricalExpr,
             SubstExpr, TickExpr, DistrExpr]
