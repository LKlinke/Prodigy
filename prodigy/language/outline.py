"""
-------------------
Outline of programs
-------------------

The outline of a program: its variables, parameters, constants and functions, its ``while`` loops with stable ids
(:class:`LoopInfo`, the key of loop decisions) and its queries, each with the position of its source text. The
positions come from the parser (:mod:`prodigy.language.positions`); a program built without source has none, and
its loops and queries are numbered in program order all the same.

Loops are numbered in pre-order: first the loops of the functions (in the order of ``program.functions``), then
those of the program; a loop nested in another follows it.

.. doctest::

    >>> from prodigy.language.parser import parse_program
    >>> program = parse_program("x: nat;\\nwhile (x < 3) {\\n    x = x + 1;\\n}\\nquery Pr[x == 3];")
    >>> [loop.describe() for loop in outline(program).loops]
    ['loop 0 (line 2): while (x < 3)']
    >>> outline(program).queries[0].line
    5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from prodigy.language.ast import (ChoiceInstr, CovarianceInstr, ExpectationInstr, IfInstr, Instr, LoopInstr,
                                  OptimizationQuery, PlotInstr, PrintInstr, ProbabilityQueryInstr, Program, QueryInstr,
                                  VarExpr, VarianceInstr, WhileInstr)
from prodigy.language.positions import span_of
from prodigy.language.ppy_printer import spell

__all__ = ["QUERY_INSTRUCTIONS", "LoopInfo", "ProgramOutline", "QueryInfo", "is_query", "outline",
           "query_instructions", "type_name"]

QUERY_INSTRUCTIONS = (CovarianceInstr, VarianceInstr, ExpectationInstr, OptimizationQuery, PlotInstr, PrintInstr,
                      ProbabilityQueryInstr)
"""The instructions that are queries (they report on the distribution and do not change it)."""


def is_query(instr: Instr) -> bool:
    """Whether ``instr`` is one of :data:`QUERY_INSTRUCTIONS`."""
    return isinstance(instr, QUERY_INSTRUCTIONS)


def query_instructions(program: Program) -> List[Instr]:
    """The queries among the top-level instructions of ``program``."""
    return [instr for instr in program.instructions if is_query(instr)]


Position = Tuple[Optional[int], Optional[int]]


def _position(instr: Instr) -> Position:
    span = span_of(instr)
    return (span.line, span.column) if span is not None else (None, None)


@dataclass(frozen=True)
class LoopInfo:
    """A while loop of the analysed program, identified by its position in the text."""

    id: int  # pylint: disable=invalid-name
    """Index of the loop in textual order (0-based); the key for decisions."""
    cond: str
    """The loop guard as ``.ppy`` text."""
    depth: int
    """Number of enclosing while loops."""
    function: Optional[str] = None
    """Name of the enclosing function declaration, if any."""
    line: Optional[int] = None
    column: Optional[int] = None

    def describe(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"loop {self.id}{where}: while ({self.cond})"

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cond": self.cond,
            "depth": self.depth,
            "function": self.function,
            "line": self.line,
            "column": self.column,
        }


@dataclass(frozen=True)
class QueryInfo:
    """A top-level query of the program."""

    index: int
    kind: str
    """``expectation``, ``variance``, ``covariance``, ``marginal``, ``probability``, ``print``, ``plot`` or
    ``optimization``."""
    text: str
    """The query as ``.ppy`` text."""
    line: Optional[int] = None
    column: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "text": self.text, "line": self.line, "column": self.column}


@dataclass
class ProgramOutline:
    """Variables, parameters, loops and queries of a program (for editors, forms and loop decisions)."""

    program: Program
    loops: List[LoopInfo] = field(default_factory=list)
    loop_instructions: List[WhileInstr] = field(default_factory=list)
    """The loops in the order of :attr:`loops`."""
    queries: List[QueryInfo] = field(default_factory=list)
    query_positions: List[Position] = field(default_factory=list)
    """The positions of all queries of the program (also those in blocks and functions), in textual order."""

    def to_json(self) -> Dict[str, Any]:
        program = self.program
        return {
            "variables": {name: type_name(typ) for name, typ in program.variables.items()},
            "parameters": {name: type_name(typ) for name, typ in program.parameters.items()},
            "constants": {name: str(value) for name, value in program.constants.items()},
            "functions": list(program.functions.keys()),
            "loops": [loop.to_json() for loop in self.loops],
            "queries": [query.to_json() for query in self.queries],
        }


def _children(instr: Instr) -> List[List[Instr]]:
    if isinstance(instr, WhileInstr):
        return [instr.body]
    if isinstance(instr, IfInstr):
        return [instr.true or [], instr.false or []]
    if isinstance(instr, ChoiceInstr):
        return [instr.lhs, instr.rhs]
    if isinstance(instr, LoopInstr):
        return [instr.body]
    if isinstance(instr, QueryInstr):
        return [instr.instrs]
    return []


def _collect_loops(instrs: List[Instr], depth: int, function: Optional[str],
                   out: List[Tuple[WhileInstr, int, Optional[str]]]) -> None:
    for instr in instrs:
        if isinstance(instr, WhileInstr):
            out.append((instr, depth, function))
            _collect_loops(instr.body, depth + 1, function, out)
        else:
            for block in _children(instr):
                _collect_loops(block, depth, function, out)


def _all_instructions(instrs: List[Instr]) -> Iterator[Instr]:
    for instr in instrs:
        yield instr
        for block in _children(instr):
            yield from _all_instructions(block)


def outline(program: Program) -> ProgramOutline:
    """The outline of ``program``; positions only if the parser recorded them for every loop (and every query)."""
    found: List[Tuple[WhileInstr, int, Optional[str]]] = []
    for name, function in program.functions.items():
        _collect_loops(function.instructions, 0, name, found)
    _collect_loops(program.instructions, 0, None, found)
    positions = [_position(instr) for instr, _, _ in found]
    if any(line is None for line, _ in positions):
        positions = [(None, None)] * len(found)
    result = ProgramOutline(program)
    for index, ((instr, depth, function_name), (line, column)) in enumerate(zip(found, positions, strict=True)):
        result.loops.append(LoopInfo(id=index, cond=spell(instr.cond), depth=depth, function=function_name,
                                     line=line, column=column))
        result.loop_instructions.append(instr)
    every = [instr for function in program.functions.values() for instr in _all_instructions(function.instructions)]
    every += list(_all_instructions(program.instructions))
    query_positions = [_position(instr) for instr in every if is_query(instr)]
    if not any(line is None for line, _ in query_positions):
        result.query_positions = sorted(query_positions)
    for index, query in enumerate(query_instructions(program)):
        line, column = _position(query)
        result.queries.append(QueryInfo(index, _query_kind(query), spell(query), line, column))
    return result


def type_name(typ: Any) -> str:
    """Spelling of a declared type in ``.ppy`` (``nat``, ``nat[0, 5]``, ``bool``, ``real``, ...)."""
    name = type(typ).__name__
    if name == "NatType":
        bounds = getattr(typ, "bounds", None)
        if bounds is None:
            return "nat"
        return f"nat[{bounds.lower}, {bounds.upper}]"
    return {"BoolType": "bool", "RealType": "real", "DistributionType": "dist"}.get(name, str(typ))


def _query_kind(instr: Instr) -> str:
    if isinstance(instr, ExpectationInstr):
        return "expectation"
    if isinstance(instr, VarianceInstr):
        return "variance"
    if isinstance(instr, CovarianceInstr):
        return "covariance"
    if isinstance(instr, ProbabilityQueryInstr):
        return "marginal" if isinstance(instr.expr, VarExpr) else "probability"
    if isinstance(instr, PrintInstr):
        return "print"
    if isinstance(instr, PlotInstr):
        return "plot"
    return "optimization"
