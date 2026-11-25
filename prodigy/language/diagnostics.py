"""
-----------------------------
Diagnostics of .ppy programs
-----------------------------

Diagnostics with positions in the source: syntax errors, type errors (a failure of the type checker) and the
language warnings (monus, implicit casts, :mod:`prodigy.language.language_warnings`), each a dictionary with
``severity``, ``message``, ``hint`` and ``line``/``column``/``end_line``/``end_column``. The GUI
(:mod:`prodigy.web.diagnostics`), the CLI and the analysis session report them.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from prodigy.language.ast import Expr, Instr
from prodigy.language.check import CheckFail
from prodigy.language.language_warnings import LanguageWarning, language_warnings
from prodigy.language.positions import span_of
from prodigy.language.ppy_printer import spell


def _sentence(text: str) -> str:
    return text if text.endswith(("…", ".")) else text + "."


def _position(source: str, offset: int) -> Tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    column = offset - (source.rfind("\n", 0, offset) + 1) + 1
    return line, column


def end_of_source(source: str) -> Tuple[int, int]:
    stripped = source.rstrip()
    return _position(source, len(stripped))


def _locate_compact(source: str, text: str) -> Optional[Tuple[int, int, int, int]]:
    """Finds ``text`` in ``source`` ignoring all whitespace; returns 1-based start/end."""
    needle = re.sub(r"\s+", "", text)
    if not needle:
        return None
    compact_chars, offsets = [], []
    for index, char in enumerate(source):
        if not char.isspace():
            compact_chars.append(char)
            offsets.append(index)
    found = "".join(compact_chars).find(needle)
    if found < 0:
        return None
    start = offsets[found]
    end = offsets[found + len(needle) - 1] + 1
    return (*_position(source, start), *_position(source, end))


def _locate_name(source: str, name: str) -> Optional[Tuple[int, int, int, int]]:
    match = re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", _blank_comments(source))
    if match is None:
        return None
    return (*_position(source, match.start()), *_position(source, match.end()))


def _blank_comments(source: str) -> str:
    # '//' is floor division; comments are '#' and '/* … */' (keep their line breaks)
    return re.sub(r"#.*$|/\*[\s\S]*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), source, flags=re.M)


def diagnostic(severity: str, message: str, where: Optional[Tuple[int, int, int, int]],
                source: str, hint: str = "") -> Dict[str, Any]:
    if where is None:
        where = (1, 1, 1, max(2, len(source.split("\n", 1)[0]) + 1))
    line, column, end_line, end_column = where
    return {"severity": severity, "message": message, "hint": hint, "line": line, "column": column,
            "end_line": end_line, "end_column": end_column}


def parse_error_diagnostic(source: str, exc: Exception) -> Dict[str, Any]:
    name = type(exc).__name__
    if name in ("PpySyntaxError", "ProgramSyntaxError", "ProgramTypeError"):
        line, column = getattr(exc, "line", None), getattr(exc, "column", None)
        message = _sentence(getattr(exc, "message", str(exc)))
        if line is None:
            return diagnostic("error", message, None, source)
        return diagnostic("error", message, (line, column or 1, line, (column or 1) + 1), source)
    message = str(exc) or name
    where = None
    match = re.search(r":\s*(\w+)\s*$", message)
    if match:
        where = _locate_name(source, match.group(1))
    return diagnostic("error", message, where, source)


def check_failure_diagnostic(source: str, failure: CheckFail) -> Dict[str, Any]:
    message = failure.message.replace("NatType(bounds=None)", "nat").replace("BoolType()", "bool") \
        .replace("RealType()", "real")
    blanked = _blank_comments(source)
    assignment = r"=(?!=)"
    where = None
    location = failure.location
    if location is not None:
        lhs = getattr(location, "lhs", None)
        if type(location).__name__ == "AsgnInstr" and isinstance(lhs, str):
            match = re.search(rf"(?<![\w]){re.escape(lhs)}\s*{assignment}", blanked)
            if match:
                where = (*_position(source, match.start()), *_position(source, match.start() + len(lhs)))
        if where is None:
            var = getattr(location, "var", None)
            if isinstance(var, str):
                where = _locate_name(source, var)
        if where is None:
            where = _locate_compact(blanked, spell(cast(Union[Expr, Instr], location)))
    if where is None:
        match = re.match(r"(\w+) is not", failure.message)
        if match:
            where = _locate_name(source, match.group(1))
    return diagnostic("warning", message, where, source,
                       hint="Reported by the type checker; Prodigy may still be able to analyse the program.")


_LANGUAGE_HINTS = {
    "monus": "The program is valid and keeps its meaning; `-` on `nat` is truncated subtraction (monus).",
    "cast": "The program is valid and keeps its meaning; write `if (…) { x = 1 } else { x = 0 }` to make it explicit.",
}


def _locate_in_instruction(source: str, warning: LanguageWarning) -> Optional[Tuple[int, int, int, int]]:
    """The place of the warning's expression: inside the instruction's source text if the expression is found
    as printed, otherwise the whole instruction (expressions carry no spans of their own)."""
    span = span_of(warning.instr)
    if span is None:
        return _locate_compact(_blank_comments(source), spell(warning.expr))
    lines = source.split("\n")
    start = sum(len(line) + 1 for line in lines[:span.line - 1]) + span.column - 1
    end = sum(len(line) + 1 for line in lines[:span.end_line - 1]) + span.end_column - 1
    segment = _blank_comments(source)[start:end]
    found = _locate_compact(segment, spell(warning.expr))
    if found is None:
        return (span.line, span.column, span.end_line, span.end_column)
    first_line, first_column, last_line, last_column = found
    base_line, base_column = _position(source, start)
    return (base_line + first_line - 1, first_column + (base_column - 1 if first_line == 1 else 0),
            base_line + last_line - 1, last_column + (base_column - 1 if last_line == 1 else 0))


def language_warning_diagnostics(program: Any, source: str) -> List[Dict[str, Any]]:
    """Warnings for valid programs that are likely to surprise (monus, implicit casts), as diagnostics
    with the position of the expression; also used by the CLI."""
    try:
        warnings = language_warnings(program)
    except Exception:  # pylint: disable=broad-except  # a heuristic check must never hide the real diagnostics
        return []
    return [diagnostic("warning", warning.message, _locate_in_instruction(source, warning), source,
                        hint=_LANGUAGE_HINTS[warning.kind]) for warning in warnings]
