from enum import Enum, auto

import attr
from abc import abstractmethod
from typing import List, Union
from textwrap import indent
from .ast import Node
from .expressions import NatLitExpr, VarExpr, RealLitExpr, Expr
from . import Var


def _str_block(instrs: List["Instr"]) -> str:
    if len(instrs) == 0:
        return "{ }"
    lines = indent("\n".join(map(str, instrs)), "    ")
    return "{\n" + lines + "\n}"


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


@attr.s
class SkipInstr(InstrClass):
    """The skip instruction does nothing."""

    def __str__(self) -> str:
        return "skip;"


@attr.s
class WhileInstr(InstrClass):
    """A while loop with a condition and a body."""
    cond: 'Expr' = attr.ib()
    body: List["Instr"] = attr.ib()

    def __str__(self) -> str:
        return f"while ({self.cond}) {_str_block(self.body)}"


@attr.s
class IfInstr(InstrClass):
    """A conditional expression with two branches."""
    cond: 'Expr' = attr.ib()
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
    rhs: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f'{self.lhs} := {self.rhs};'


@attr.s
class ChoiceInstr(InstrClass):
    """A probabilistic choice instruction with a probability expression and two branches."""
    prob: 'Expr' = attr.ib()
    lhs: List["Instr"] = attr.ib()
    rhs: List["Instr"] = attr.ib()

    def __str__(self) -> str:
        return f"{_str_block(self.lhs)} [{self.prob}] {_str_block(self.rhs)}"


@attr.s
class LoopInstr(InstrClass):
    """ iterating a block a constant amount of times"""
    iterations: NatLitExpr = attr.ib()
    body: List["Instr"] = attr.ib()

    def __str__(self) -> str:
        return f"loop({self.iterations}){_str_block(self.body)}"


@attr.s
class TickInstr(InstrClass):
    """
    An instruction that does not modify the program state, but only increases
    the runtime by the value of the expression in the current state. Its only
    use is its translation to :class:`TickExpr` by weakest pre-expectations.

    The type of ``expr`` must be :class:`NatType`.
    """
    expr: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f"tick({self.expr});"


@attr.s
class ObserveInstr(InstrClass):
    """
    Updates the current distribution according to the observation (forward analysis only).
    May result in an error if the observed condition has probability zero.

    The type of ``expr`` must be :class:`BoolType`.
    """
    cond: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f"observe({self.cond});"


@attr.s
class ExpectationInstr(InstrClass):
    """
    Allows for expectation queries inside of a pgcl program.
    """
    expr: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f"?Ex[{self.expr}];"


class OptimizationType(Enum):
    MAXIMIZE = auto()
    MINIMIZE = auto()


@attr.s
class OptimizationQuery(InstrClass):
    expr: 'Expr' = attr.ib()
    parameter: Var = attr.ib()
    type: OptimizationType = attr.ib()

    def __str__(self) -> str:
        return f"?Opt[{self.expr}, {self.parameter}, {'MAX' if self.type == OptimizationType.MAXIMIZE else 'MIN'}];"


@attr.s
class ProbabilityQueryInstr(InstrClass):
    expr: 'Expr' = attr.ib()

    def __str__(self) -> str:
        return f"?Pr[{self.expr}];"


@attr.s
class PrintInstr(InstrClass):
    def __str__(self) -> str:
        return f"!Print;"


@attr.s
class PlotInstr(InstrClass):
    var_1: VarExpr = attr.ib()
    var_2: VarExpr = attr.ib(default=None)
    prob: RealLitExpr = attr.ib(default=None)
    term_count: NatLitExpr = attr.ib(default=None)

    def __str__(self) -> str:
        output = str(self.var_1)
        if self.var_2:
            output += f", {str(self.var_2)}"
        if self.prob:
            output += f", {str(self.prob)}"
        if self.term_count:
            output += f", {str(self.term_count)}"
        return f"!Plot[{output}]"


Query = Union[ProbabilityQueryInstr, ExpectationInstr, PlotInstr, PrintInstr, OptimizationQuery]
"""Union type for all query objects. See :class:`QueryInstr` for use with isinstance."""

Instr = Union[SkipInstr, WhileInstr, IfInstr, AsgnInstr, ChoiceInstr, LoopInstr,
              TickInstr, ObserveInstr, Query]
"""Union type for all instruction objects. See :class:`InstrClass` for use with isinstance."""
