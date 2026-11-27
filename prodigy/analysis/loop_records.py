"""
The loops of the analysed program and what each of them produced during a run.

:class:`LoopRegistry` gives every loop of the program its id and finds the loop of an instruction, also of a
copy of it; :class:`LoopRecord` collects what one loop's strategy produced, certified and reported. The
:class:`~prodigy.analysis.session.AnalysisSession` keeps one registry and one record per loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from prodigy.language import Program, WhileInstr
from prodigy.language.outline import LoopInfo, outline
from prodigy.language.positions import span_of

from prodigy.analysis.loops.api import Bound
from prodigy.analysis.values import exact_and_float

__all__ = ["LoopRecord", "LoopRegistry"]


class LoopRegistry:
    """The loops of the analysed program with their ids (:func:`~prodigy.language.outline.outline`); finds the
    :class:`LoopInfo` of a loop instruction, also of a copy of it."""

    def __init__(self, program: Optional[Program]):
        self.loops: List[LoopInfo] = []
        self._by_identity: Dict[int, LoopInfo] = {}
        self._by_position: Dict[Tuple[int, int], LoopInfo] = {}
        self._by_structure: Dict[str, List[LoopInfo]] = {}
        self._instructions: List[WhileInstr] = []
        if program is None:
            return
        found = outline(program)
        for info, instr in zip(found.loops, found.loop_instructions, strict=True):
            self.loops.append(info)
            self._instructions.append(instr)
            self._by_identity[id(instr)] = info
            if info.line is not None and info.column is not None:
                self._by_position[(info.line, info.column)] = info
            self._by_structure.setdefault(repr(instr), []).append(info)

    def lookup(self, instr: WhileInstr) -> Optional[LoopInfo]:
        """The loop ``instr``, or the loop it is a copy of: found by identity, else by its source position (a
        deep copy keeps it, :mod:`prodigy.language.positions`), else by its structure (the ``repr`` of the tree) if
        that is unique (a copy without position, e.g. of a program built in Python)."""
        info = self._by_identity.get(id(instr))
        if info is not None:
            return info
        span = span_of(instr)
        if span is not None and (span.line, span.column) in self._by_position:
            return self._by_position[(span.line, span.column)]
        candidates = self._by_structure.get(repr(instr), [])
        return candidates[0] if len(candidates) == 1 else None

    def instruction(self, loop_id: int) -> Optional[WhileInstr]:
        """The loop with id ``loop_id`` (``None`` if there is none)."""
        return self._instructions[loop_id] if 0 <= loop_id < len(self._instructions) else None

    def alias(self, copy: WhileInstr, original: WhileInstr) -> None:
        """Looks up ``copy`` as ``original`` (see :meth:`AnalysisSession.loop_alias`)."""
        info = self.lookup(original)
        if info is not None:
            self._by_identity[id(copy)] = info

    def unalias(self, copy: WhileInstr) -> None:
        self._by_identity.pop(id(copy), None)

    def __iter__(self) -> Iterator[LoopInfo]:
        return iter(self.loops)

    def __len__(self) -> int:
        return len(self.loops)


@dataclass
class LoopRecord:
    """Everything one loop produced during a run (a loop may be entered several times)."""

    loop: Optional[LoopInfo]
    cond: str
    strategy: str
    params: Dict[str, Any]
    invocations: int = 0
    kind: str = Bound.EXACT
    residual: Any = 0
    time: float = 0.0
    certificates: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    progress: Optional[Tuple[int, Optional[int]]] = None
    failed: Optional[str] = None
    """Error message if the strategy failed (e.g. a refuted invariant)."""
    inside_checks: int = 0
    """How many of the invocations happened inside the check of another loop's invariant."""
    suppressed_messages: int = 0
    execution_kinds: Dict[str, int] = field(default_factory=dict)
    """How many executions that returned an outcome gave each kind of guarantee."""
    execution_messages: Dict[str, int] = field(default_factory=dict)
    """The messages of the outcomes, each with the number of executions that gave it (in order)."""

    MAX_MESSAGES = 4

    def add_message(self, message: str) -> None:
        """A message that does not belong to one outcome (a note, the error of a failed execution)."""
        if message in self.messages:
            return
        if len(self.messages) >= self.MAX_MESSAGES:
            self.suppressed_messages += 1
            return
        self.messages.append(message)

    def add_outcome(self, kind: str, messages: Sequence[str]) -> None:
        """Books the guarantee and the messages of one execution of the loop."""
        self.execution_kinds[kind] = self.execution_kinds.get(kind, 0) + 1
        for message in dict.fromkeys(messages):
            self.execution_messages[message] = self.execution_messages.get(message, 0) + 1

    def outcome_messages(self) -> List[str]:
        """The messages of the outcomes. A loop that ran several times (e.g. inside another loop) gets
        the messages that all its executions gave and, if the others differ between executions (e.g.
        the residual mass of each), one summary instead of them (#229)."""
        executions = sum(self.execution_kinds.values())
        common = [message for message, count in self.execution_messages.items() if count == executions]
        if len(common) == len(self.execution_messages):
            return common
        return common + [self._summary(executions)]

    def _summary(self, executions: int) -> str:
        order = (Bound.UNKNOWN, Bound.UPPER, Bound.LOWER, Bound.EXACT)
        labels = {Bound.UNKNOWN: "no guarantee", Bound.UPPER: "upper bound", Bound.LOWER: "lower bound",
                  Bound.EXACT: "exact"}
        counts = [f"{labels[kind]} in {self.execution_kinds[kind]}" for kind in order if kind in self.execution_kinds]
        summary = f"{executions} executions: {', '.join(counts)}"
        if self.execution_kinds.get(Bound.LOWER) and self.residual != 0:
            summary += f"; in total, mass {self.residual} is still inside the loop"
        return summary + "."

    def to_json(self) -> Dict[str, Any]:
        residual, residual_float = exact_and_float(self.residual)
        return {
            "id": self.loop.id if self.loop else None,
            "line": self.loop.line if self.loop else None,
            "depth": self.loop.depth if self.loop else None,
            "cond": self.cond,
            "strategy": self.strategy,
            "params": dict(self.params),
            "invocations": self.invocations,
            "kind": self.kind,
            "residual_mass": residual,
            "residual_mass_float": residual_float,
            "time": round(self.time, 6),
            "certificates": self.certificates,
            "messages": self.outcome_messages() + self.messages
            + ([f"… {self.suppressed_messages} similar message(s) omitted."] if self.suppressed_messages else []),
            "failed": self.failed,
            "inside_checks": self.inside_checks,
        }
