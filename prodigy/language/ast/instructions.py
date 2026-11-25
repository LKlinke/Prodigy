from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Union

import attr

from .ast import Node, Var
from .expressions import Expr, NatLitExpr, RealLitExpr, VarExpr

if TYPE_CHECKING:
    from .declarations import VarDecl


class InstrClass(Node):
    """Superclass for all instructions. See :obj:`Instr`."""
    def cast(self) -> Instr:
        """Cast to Instr. This is sometimes necessary to satisfy the type checker."""
        return self  # type: ignore[return-value]



@attr.s
class SkipInstr(InstrClass):
    """The skip instruction does nothing."""

@attr.s
class AbortInstr(InstrClass):
    """The abort instruction is syntactic sugar for non-termination."""

@attr.s
class WhileInstr(InstrClass):
    """A while loop with a condition and a body."""
    cond: Expr = attr.ib()
    body: List[Instr] = attr.ib()


@attr.s
class IfInstr(InstrClass):
    """A conditional expression with two branches."""
    cond: Expr = attr.ib()
    true: List[Instr] = attr.ib()
    false: List[Instr] = attr.ib()


@attr.s
class AsgnInstr(InstrClass):
    """An assignment instruction with a left- and right-hand side."""
    lhs: Var = attr.ib()
    rhs: Expr = attr.ib()


@attr.s
class ChoiceInstr(InstrClass):
    """A probabilistic choice instruction with a probability expression and two branches."""
    prob: Expr = attr.ib()
    lhs: List[Instr] = attr.ib()
    rhs: List[Instr] = attr.ib()


@attr.s
class LoopInstr(InstrClass):
    """Iterating a block a constant amount of times."""
    iterations: NatLitExpr = attr.ib()
    body: List[Instr] = attr.ib()


@attr.s
class ObserveInstr(InstrClass):
    """
    Updates the current distribution according to the observation (forward analysis only).
    May result in an error if the observed condition has probability zero.

    The type of ``cond`` must be :class:`BoolType`.
    """
    cond: Expr = attr.ib()


@attr.s
class ExpectationInstr(InstrClass):
    """
    Allows for expectation queries inside of a program.
    """
    expr: Expr = attr.ib()


@attr.s
class VarianceInstr(InstrClass):
    """
    The variance ``Var[e]`` of an expression, a query inside of a program.
    """
    expr: Expr = attr.ib()


@attr.s
class CovarianceInstr(InstrClass):
    """
    The covariance ``Cov[lhs, rhs]`` of two expressions, a query inside of a program.
    """
    lhs: Expr = attr.ib()
    rhs: Expr = attr.ib()


class OptimizationType(Enum):
    MAXIMIZE = auto()
    MINIMIZE = auto()


@attr.s
class OptimizationQuery(InstrClass):
    expr: Expr = attr.ib()
    parameter: Var = attr.ib()
    type: OptimizationType = attr.ib()


@attr.s
class ProbabilityQueryInstr(InstrClass):
    expr: Expr = attr.ib()


@attr.s
class PrintInstr(InstrClass):
    """Prints the current distribution."""


@attr.s
class PlotInstr(InstrClass):
    var_1: VarExpr = attr.ib()
    var_2: VarExpr = attr.ib(default=None)
    prob: RealLitExpr = attr.ib(default=None)
    term_count: NatLitExpr = attr.ib(default=None)


@attr.s
class QueryInstr(InstrClass):
    instrs: List[Instr] = attr.ib()


@attr.s
class LocalDeclInstr(InstrClass):
    """
    A declaration inside a block (``.ppy`` syntax only), visible until the end of
    the enclosing block, with an optional initial value. Without a value the
    variable starts at 0. At the end of the block the variable is marginalised
    out.

    :func:`prodigy.language.scope.lower_block_scope` lowers it into plain
    assignments (done by :func:`prodigy.language.parser.parse_program`); the type
    checker and the analysis never see it.
    """
    decl: VarDecl = attr.ib()
    value: Optional[Expr] = attr.ib(default=None)


Query = Union[ProbabilityQueryInstr, ExpectationInstr, VarianceInstr, CovarianceInstr, PlotInstr, PrintInstr,
              OptimizationQuery]
"""Union type for all query objects. See :class:`QueryInstr` for use with isinstance."""

Instr = Union[SkipInstr, AbortInstr, WhileInstr, IfInstr, AsgnInstr, ChoiceInstr,
              LoopInstr, ObserveInstr, QueryInstr, LocalDeclInstr,
              Query]
"""Union type for all instruction objects. See :class:`InstrClass` for use with isinstance."""
