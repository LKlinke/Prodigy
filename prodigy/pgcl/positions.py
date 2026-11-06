"""
Source positions of instructions.

Both parsers (pGCL and ``.ppy``) attach to every instruction and query the span of its source
text: :func:`span_of` returns a :class:`SourceSpan` with 1-based line and column of its start and
end (the end column is the one after the last character, as in Lark). The span is kept outside of
the AST's attributes, so it never takes part in equality, ``repr`` or ``str``; ``copy.deepcopy``
keeps it, while rewriting a node into a new one (e.g. ``attr.evolve``) has to copy it with
:func:`copy_span`. Instructions that a transformation creates (e.g. the assignments that end the
scope of a block variable in ``.ppy``) have no span.

The step trace of the analysis (:mod:`prodigy.analysis.trace`) uses the spans to show where each
step happened.

.. doctest::

    >>> from prodigy.pgcl.parser import parse_pgcl
    >>> program = parse_pgcl("nat x;\\nx := 1;\\nwhile (x < 3) {\\n  x := x + 1\\n}")
    >>> span_of(program.instructions[1])
    SourceSpan(line=3, column=1, end_line=5, end_column=2)
    >>> span_of(program.instructions[1].body[0]).line
    4
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar

_ATTRIBUTE = "_prodigy_source_span"

T = TypeVar("T")


@dataclass(frozen=True)
class SourceSpan:
    """Where an instruction stands in the source text (1-based; ``end_column`` is exclusive)."""

    line: int
    column: int
    end_line: int
    end_column: int

    def to_json(self) -> Dict[str, int]:
        return {"line": self.line, "column": self.column, "end_line": self.end_line,
                "end_column": self.end_column}


def span_from_meta(meta: Any) -> Optional[SourceSpan]:
    """The span of a Lark node's ``meta`` (``None`` if Lark recorded no position)."""
    if meta is None or getattr(meta, "empty", True):
        return None
    try:
        return SourceSpan(int(meta.line), int(meta.column), int(meta.end_line), int(meta.end_column))
    except (AttributeError, TypeError, ValueError):
        return None


def set_span(node: T, span: Optional[SourceSpan]) -> T:
    """Attaches ``span`` to ``node`` (no effect for ``None`` or nodes that take no attributes)."""
    if span is not None and node is not None:
        try:
            object.__setattr__(node, _ATTRIBUTE, span)
        except (AttributeError, TypeError):
            pass
    return node


def span_of(node: Any) -> Optional[SourceSpan]:
    """The span the parser attached to ``node``, if any."""
    span = getattr(node, _ATTRIBUTE, None)
    return span if isinstance(span, SourceSpan) else None


def copy_span(source: Any, target: T) -> T:
    """Gives ``target`` the span of ``source`` unless it has one already; returns ``target``."""
    if span_of(target) is None:
        set_span(target, span_of(source))
    return target
