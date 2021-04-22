r"""
---
AST
---

All data types to represent a pGCL program.

.. note::

    For our AST types, most "sum types" have a normal Python superclass (e.g. :class:`TypeClass` for types), but also a Union type (e.g. :data:`Type`).
    **Prefer using the Union type, as it allows for exhaustiveness checking by mypy.**

    In some rare cases, it's necessary to tell mypy that an object of a superclass type is actually in the respective union.
    Use ``cast`` methods, e.g. :meth:`InstrClass.cast` to go from :class:`InstrClass` to :data:`Instr`.

    Also note that ``isinstance`` checks must be done against the superclasses, because Python's Union does not yet work with ``isinstance``.

Program
#######
.. autoclass:: Program

    .. automethod:: __str__

Types
#####
.. autodata:: Type
.. autoclass:: BoolType
.. autoclass:: NatType
.. autoclass:: Bounds
.. autoclass:: FloatType

Declarations
############
.. autodata:: Decl
.. autoclass:: VarDecl
.. autoclass:: ConstDecl


.. _expressions:

Expressions
###########

In the AST, a bunch of different things that look like program expressions are
lumped together. Formally, let a state :math:`\sigma \in \Sigma` consists of a
bunch of values for each variable, and be represented by a function of type
:math:`\sigma : \text{Var} \to \text{Value}`.

First, there are *state expressions* that, given a state :math:`\sigma`, compute
some value to be used later in the program itself: :math:`\Sigma \to
\text{Value}`. There are two types of expressions we call *monadic expressions*:
:class:`UniformExpr` and :class:`CategoricalExpr`. They map a state to a
distribution of values: :math:`\Sigma \to \text{Dist}[\Sigma]`. And
*expectations* are also expressed using :data:`Expr`, but they are actually a
mapping of states to *expected values*: :math:`\Sigma \to \mathbb{R}`.

.. autodata:: Expr
.. autoclass:: VarExpr
.. autoclass:: BoolLitExpr
.. autoclass:: NatLitExpr
.. autoclass:: FloatLitExpr
.. autoclass:: Unop
.. autoclass:: UnopExpr
.. autoclass:: Binop
.. autoclass:: BinopExpr
.. autoclass:: UniformExpr
.. autoclass:: CategoricalExpr
.. autoclass:: SubstExpr
.. autoclass:: TickExpr

Instructions
############
.. autodata:: Instr
.. autoclass:: SkipInstr
.. autoclass:: WhileInstr
.. autoclass:: IfInstr
.. autoclass:: AsgnInstr
.. autoclass:: ChoiceInstr
.. autoclass:: TickInstr

Superclasses
############
These are only used for use with isinstance.
Otherwise use corresponding Union types instead.

.. autoclass:: TypeClass
.. autoclass:: DeclClass

    .. automethod:: __str__

.. autoclass:: ExprClass

    .. automethod:: __str__

.. autoclass:: InstrClass

    .. automethod:: __str__

.. autoclass:: Node
"""
import copy
import itertools
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum, auto
from fractions import Fraction
from functools import reduce
from textwrap import indent
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import attr

Var = str


@attr.s
class Node(ABC):
    """Superclass for all node types in the AST."""


@attr.s
class Bounds:
    """
    Bounds for a natural number type.

    The bounds can contain constant expressions, therefore bounds have type :class:`Expr`.
    """
    lower: "Expr" = attr.ib()
    upper: "Expr" = attr.ib()

    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"


@attr.s
class TypeClass(Node):
    """Superclass for all types. See :obj:`Type`."""


@attr.s
class BoolType(TypeClass):
    """Boolean type."""


@attr.s
class NatType(TypeClass):
    """
    Natural number types with optional bounds.

    Bounds are only preserved for variables.
    Values of bounded types are considered as unbounded until they are assigned to a bounded variable.
    That is to say, bounds are lost in expressions such as in the example below:

    .. doctest::

        >>> from .parser import parse_pgcl, parse_expr
        >>> from .check import get_type
        >>> program = parse_pgcl("nat x [1,5]")
        >>> get_type(program, parse_expr("x + 5"))
        NatType(bounds=None)
    """

    bounds: Optional[Bounds] = attr.ib()


@attr.s
class FloatType(TypeClass):
    """Floating numbers, used for probabilities."""


Type = Union[BoolType, NatType, FloatType]
"""Union type for all type objects. See :class:`TypeClass` for use with isinstance."""


class DeclClass(Node):
    """Superclass for all declarations. See :obj:`Decl`."""
    @abstractmethod
    def __str__(self) -> str:
        """
        Convert this declaration to corresponding source code in pGCL.

        .. doctest::

            >>> str(VarDecl('x', NatType(Bounds(1, 10))))
            'nat x [1, 10];'
        """


@attr.s
class VarDecl(DeclClass):
    """A variable declaration with a name and a type."""
    var: Var = attr.ib()
    typ: Type = attr.ib()

    def __str__(self) -> str:
        if isinstance(self.typ, BoolType):
            return f"bool {self.var};"
        elif isinstance(self.typ, NatType):
            res = f"nat {self.var}"
            if self.typ.bounds is not None:
                res += " " + str(self.typ.bounds)
            return res + ";"
        raise ValueError(f"invalid type: {self.typ}")


@attr.s
class ConstDecl(DeclClass):
    """A constant declaration with a name and an expression."""
    var: Var = attr.ib()
    value: "Expr" = attr.ib()

    def __str__(self) -> str:
        return f"const {self.var} := {self.value};"


Decl = Union[VarDecl, ConstDecl]
"""Union type for all declaration objects. See :class:`DeclClass` for use with isinstance."""


class ExprClass(Node):
    """
    Superclass for all expressions.
    See :obj:`Expr`.

    All expressions can be transformed into human-readable strings using the `str` operator.
    """
    def cast(self) -> "Expr":
        """Cast to Expr. This is sometimes necessary to satisfy the type checker."""
        return self  # type: ignore

    @abstractmethod
    def __str__(self) -> str:
        """
        Convert this expression to corresponding source code in pGCL.

        .. doctest::

            >>> from .parser import parse_expr
            >>> str(parse_expr("x < 2 & not true"))
            '(x < 2) & not true'
        """


@attr.s(repr=False)
class VarExpr(ExprClass):
    """A variable is an expression."""
    var: Var = attr.ib()

    def __str__(self) -> str:
        return self.var

    def __repr__(self) -> str:
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


def _validate_float_lit_value(_object: 'FloatLitExpr', _attribute: Any,
                              value: Any):
    if not isinstance(value, Decimal) and not isinstance(value, Fraction):
        raise ValueError(
            f"Expected a Decimal or Fraction value, got: {value!r}")


def _parse_float_lit_expr(
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
class FloatLitExpr(ExprClass):
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
                 Fraction] = attr.ib(validator=_validate_float_lit_value,
                                     converter=_parse_float_lit_expr)

    @staticmethod
    def infinity() -> 'FloatLitExpr':
        """
        Create a new infinite value.

        .. doctest::

            >>> FloatLitExpr.infinity().is_infinite()
            True
        """
        return FloatLitExpr(Decimal('Infinity'))

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

            >>> expr = FloatLitExpr("0.1")
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
        return f'FloatLitExpr("{str(self.value)}")'


class Unop(Enum):
    """What unary operator is it?"""
    NEG = auto()
    IVERSON = auto()

    def __repr__(self) -> str:
        # pylint: disable=no-member
        return f'Unop.{self._name_}'


@attr.s
class UnopExpr(ExprClass):
    """A unary operator is an expression."""
    operator: Unop = attr.ib()
    expr: "Expr" = attr.ib()

    def __str__(self) -> str:
        if self.operator == Unop.NEG:
            return f'not {expr_str_parens(self.expr)}'
        elif self.operator == Unop.IVERSON:
            return f'[{self.expr}]'
        raise Exception("Invalid Unop operator")


class Binop(Enum):
    """What binary operator is it?"""
    OR = auto()
    AND = auto()
    LEQ = auto()
    LE = auto()
    EQ = auto()
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIVIDE = auto()

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
            Binop.EQ: "=",
            Binop.PLUS: "+",
            Binop.MINUS: "-",
            Binop.TIMES: "*",
            Binop.DIVIDE: "/",
        })[self]


@attr.s
class BinopExpr(ExprClass):
    """A binary operator is an expression."""
    operator: Binop = attr.ib()
    lhs: "Expr" = attr.ib()
    rhs: "Expr" = attr.ib()

    @staticmethod
    def reduce(operator: Binop,
               iterable: Sequence["Expr"],
               default: Optional["Expr"] = None) -> "Expr":
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

    def flatten(self) -> List["Expr"]:
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

        def flatten_expr(expr: "Expr") -> List["Expr"]:
            if isinstance(expr, BinopExpr) and expr.operator == self.operator:
                return flatten_expr(expr.lhs) + flatten_expr(expr.rhs)
            else:
                return [expr]

        return flatten_expr(self)

    def __str__(self) -> str:
        return f'{expr_str_parens(self.lhs)} {self.operator} {expr_str_parens(self.rhs)}'


@attr.s
class UniformExpr(ExprClass):
    """
    Chooses a random integer within the (inclusive) interval.

    As *monadic expressions* (see :ref:`expressions`), uniform choice
    expressions are only allowed as the right-hand side of an assignment
    statement and not somewhere in a nested expression.
    """
    start: NatLitExpr = attr.ib()
    end: NatLitExpr = attr.ib()

    def distribution(self) -> List[Tuple[FloatLitExpr, NatLitExpr]]:
        r"""
        Return the distribution of possible values as a list along with
        probabilities. For the uniform distribution, all probabilites are equal
        to :math:`\frac{1}{\text{end} - \text{start} + 1}`.
        """
        width = self.end.value - self.start.value + 1
        prob = FloatLitExpr(Fraction(1, width))
        return [(prob, NatLitExpr(i))
                for i in range(self.start.value, self.end.value + 1)]

    def __str__(self) -> str:
        return f'unif({expr_str_parens(self.start)}, {expr_str_parens(self.end)})'


def _check_categorical_exprs(_self: "CategoricalExpr", _attribute: Any,
                             value: List[Tuple["Expr", FloatLitExpr]]):
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
    exprs: List[Tuple["Expr", FloatLitExpr]] = attr.ib(
        validator=_check_categorical_exprs)

    def distribution(self) -> List[Tuple[FloatLitExpr, "Expr"]]:
        r"""
        Return the distribution of possible values as a list along with
        probabilities.
        """
        return [(prob, expr) for expr, prob in self.exprs]

    def __str__(self) -> str:
        return " + ".join((f"{expr_str_parens(expr)} : {expr_str_parens(prob)}"
                           for expr, prob in self.exprs))


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
    preexpectation of a program (see :mod:`probably.pgcl.wp`).

    Substitutions can be applied using the :mod:`probably.pgcl.substitute`
    module.
    """
    subst: Dict[Var, "Expr"] = attr.ib()
    expr: "Expr" = attr.ib()

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
    expr: "Expr" = attr.ib()

    def is_zero(self) -> bool:
        """Whether this expression represents exactly zero ticks."""
        return self.expr == NatLitExpr(0)

    def __str__(self) -> str:
        return f"tick({self.expr})"


def expr_str_parens(expr: ExprClass) -> str:
    """Wrap parentheses around an expression, but not for simple expressions."""
    if isinstance(expr,
                  (VarExpr, BoolLitExpr, NatLitExpr, FloatLitExpr, UnopExpr)):
        return str(expr)
    else:
        return f'({expr})'


Expr = Union[VarExpr, BoolLitExpr, NatLitExpr, FloatLitExpr, UnopExpr,
             BinopExpr, UniformExpr, CategoricalExpr, SubstExpr, TickExpr]
"""Union type for all expression objects. See :class:`ExprClass` for use with isinstance."""


class InstrClass(Node):
    """Superclass for all instructions. See :obj:`Instr`."""
    def cast(self) -> "Instr":
        """Cast to Instr. This is sometimes necessary to satisfy the type checker."""
        return self  # type: ignore

    @abstractmethod
    def __str__(self) -> str:
        """
        Convert this instruction to corresponding source code in pGCL.

        .. doctest::

            >>> print(SkipInstr())
            skip;
            >>> print(WhileInstr(BoolLitExpr(True), [SkipInstr()]))
            while (true) {
                skip;
            }
            >>> print(IfInstr(BoolLitExpr(False), [SkipInstr()], []))
            if (false) {
                skip;
            }
        """


def _str_block(instrs: List["Instr"]) -> str:
    if len(instrs) == 0:
        return "{ }"
    lines = indent("\n".join(map(str, instrs)), "    ")
    return "{\n" + lines + "\n}"


@attr.s
class SkipInstr(InstrClass):
    """The skip instruction does nothing."""
    def __str__(self) -> str:
        return "skip;"


@attr.s
class WhileInstr(InstrClass):
    """A while loop with a condition and a body."""
    cond: Expr = attr.ib()
    body: List["Instr"] = attr.ib()

    def __str__(self) -> str:
        return f"while ({self.cond}) {_str_block(self.body)}"


@attr.s
class IfInstr(InstrClass):
    """A conditional expression with two branches."""
    cond: Expr = attr.ib()
    true: List["Instr"] = attr.ib()
    false: List["Instr"] = attr.ib()

    def __str__(self) -> str:
        if len(self.false) > 0:
            else_str = f" else {_str_block(self.false)}"
        else:
            else_str = ""
        return f"if ({self.cond}) {_str_block(self.true)}{else_str}"


@attr.s
class AsgnInstr(InstrClass):
    """An assignment instruction with a left- and right-hand side."""
    lhs: Var = attr.ib()
    rhs: Expr = attr.ib()

    def __str__(self) -> str:
        return f'{self.lhs} := {self.rhs};'


@attr.s
class ChoiceInstr(InstrClass):
    """A probabilistic choice instruction with a probability expression and two branches."""
    prob: Expr = attr.ib()
    lhs: List["Instr"] = attr.ib()
    rhs: List["Instr"] = attr.ib()

    def __str__(self) -> str:
        return f"{_str_block(self.lhs)} [{self.prob}] {_str_block(self.rhs)}"


@attr.s
class TickInstr(InstrClass):
    """
    An instruction that does not modify the program state, but only increases
    the runtime by the value of the expression in the current state. Its only
    use is its translation to :class:`TickExpr` by weakest pre-expectations.

    The type of ``expr`` must be :class:`NatType`.
    """
    expr: Expr = attr.ib()

    def __str__(self) -> str:
        return f"tick({self.expr});"


Instr = Union[SkipInstr, WhileInstr, IfInstr, AsgnInstr, ChoiceInstr,
              TickInstr]
"""Union type for all instruction objects. See :class:`InstrClass` for use with isinstance."""


@attr.s
class Program:
    """
    A pGCL program has a bunch of variables with types, constants with defining expressions, and a list of instructions.
    """
    declarations: List[Decl] = attr.ib(repr=False)
    """The original list of declarations."""

    variables: Dict[Var, Type] = attr.ib()
    """
    A dict of variables to their type.
    Only valid if the declarations are well-typed.
    """

    constants: Dict[Var, Expr] = attr.ib()
    """
    A dict of constant names to their defining expression.
    Only valid if the declarations are well-typed.
    """

    instructions: List[Instr] = attr.ib()

    @staticmethod
    def from_parse(declarations: List[Decl],
                   instructions: List[Instr]) -> "Program":
        """Create a program from the parser's output."""
        variables: Dict[Var, Type] = dict()
        constants: Dict[Var, Expr] = dict()

        for decl in declarations:
            if isinstance(decl, VarDecl):
                variables[decl.var] = decl.typ
            elif isinstance(decl, ConstDecl):
                constants[decl.var] = decl.value

        return Program(declarations, variables, constants, instructions)

    def add_variable(self, var: Var, typ: Type):
        """
        Add a new variable declaration to the program's list of declarations and
        to the dict of variables.

        :raises AssertionError: if the variable is already declared
        """
        for decl in self.declarations:
            assert decl.var != var, f"name {var} is already declared in program"
        assert var not in self.variables, f"variable {var} is already declared in program"
        self.declarations.append(VarDecl(var, typ))
        self.variables[var] = typ

    def to_skeleton(self) -> 'Program':
        """
        Return a (shallow) copy of this program with just the declarations, but
        without any instructions.

        .. doctest::

            >>> from .parser import parse_pgcl
            >>> program = parse_pgcl("nat x; nat y; while (x < 2) {}")
            >>> program.to_skeleton()
            Program(variables={'x': NatType(bounds=None), 'y': NatType(bounds=None)}, constants={}, instructions=[])
        """
        return Program(declarations=copy.copy(self.declarations),
                       variables=copy.copy(self.variables),
                       constants=copy.copy(self.constants),
                       instructions=[])

    def __str__(self) -> str:
        """
        Convert this program to corresponding source code in pGCL.

        .. doctest::

            >>> from .parser import parse_pgcl
            >>> program = parse_pgcl("nat x; nat y; while (x < 2) {}")
            >>> print(program)
            nat x;
            nat y;
            while (x < 2) { }
        """
        instrs: List[Any] = list(self.declarations)
        instrs.extend(self.instructions)
        return "\n".join(map(str, instrs))
