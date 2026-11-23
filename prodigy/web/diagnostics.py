"""
Diagnostics for editors: parses and type-checks a program (pGCL or ``.ppy``) and reports
problems with positions, plus an outline of the program (variables, loops with their ids,
queries).

Used by the web GUI; a language server can use it as well.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import copy

from prodigy.pgcl import parse_pgcl, substitute
from prodigy.pgcl.check import CheckFail, check_program
from prodigy.pgcl.language_warnings import LanguageWarning, language_warnings
from prodigy.pgcl.positions import span_of

from prodigy.analysis.result import outline_program

_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_SIZE = 64
_LOCK = threading.Lock()


def _sentence(text: str) -> str:
    return text if text.endswith(("…", ".")) else text + "."


def _position(source: str, offset: int) -> Tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    column = offset - (source.rfind("\n", 0, offset) + 1) + 1
    return line, column


def _end_of_source(source: str) -> Tuple[int, int]:
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


def _locate_name(source: str, name: str, syntax: str = "ppy") -> Optional[Tuple[int, int, int, int]]:
    match = re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", _blank_comments(source, syntax))
    if match is None:
        return None
    return (*_position(source, match.start()), *_position(source, match.end()))


def _blank_comments(source: str, syntax: str = "ppy") -> str:
    # in .ppy, '//' is floor division; comments are '#' and '/* … */' (keep their line breaks)
    pattern = r"#.*$|/\*[\s\S]*?\*/" if syntax == "ppy" else r"(#|//).*$"
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m.group(0)), source, flags=re.M)


def _diagnostic(severity: str, message: str, where: Optional[Tuple[int, int, int, int]],
                source: str, hint: str = "") -> Dict[str, Any]:
    if where is None:
        where = (1, 1, 1, max(2, len(source.split("\n", 1)[0]) + 1))
    line, column, end_line, end_column = where
    return {"severity": severity, "message": message, "hint": hint, "line": line, "column": column,
            "end_line": end_line, "end_column": end_column}


def _parse_error(source: str, exc: Exception) -> Dict[str, Any]:
    name = type(exc).__name__
    if name in ("PpySyntaxError", "ProgramSyntaxError", "ProgramTypeError"):
        line, column = getattr(exc, "line", None), getattr(exc, "column", None)
        message = _sentence(getattr(exc, "message", str(exc)))
        if line is None:
            return _diagnostic("error", message, None, source)
        return _diagnostic("error", message, (line, column or 1, line, (column or 1) + 1), source)
    message = str(exc) or name
    where = None
    match = re.search(r":\s*(\w+)\s*$", message)
    if match:
        where = _locate_name(source, match.group(1))
    return _diagnostic("error", message, where, source)


def _spelling(node: Any, syntax: str) -> str:
    """``node`` as it is written in ``syntax`` (for finding it in the source)."""
    if syntax == "ppy":
        # pylint: disable=import-outside-toplevel
        from prodigy.pgcl.ppy_printer import expr_to_ppy, instr_to_ppy
        printer = expr_to_ppy if type(node).__name__.endswith("Expr") else instr_to_ppy
        try:
            return printer(node)
        except Exception:  # pylint: disable=broad-except  # not printable: search the pGCL spelling
            pass
    return str(node)


def _check_failure(source: str, failure: CheckFail, syntax: str = "ppy") -> Dict[str, Any]:
    message = failure.message.replace("NatType(bounds=None)", "nat").replace("BoolType()", "bool") \
        .replace("RealType()", "real")
    blanked = _blank_comments(source, syntax)
    assignment = r"=(?!=)" if syntax == "ppy" else ":="
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
                where = _locate_name(source, var, syntax)
        if where is None:
            where = _locate_compact(blanked, _spelling(location, syntax))
    if where is None:
        match = re.match(r"(\w+) is not", failure.message)
        if match:
            where = _locate_name(source, match.group(1), syntax)
    return _diagnostic("warning", message, where, source,
                       hint="Reported by the pGCL type checker; Prodigy may still be able to analyse the "
                            "program.")


_LANGUAGE_HINTS = {
    "monus": "The program is valid and keeps its meaning; `-` on `nat` is truncated subtraction (monus).",
    "cast": "The program is valid and keeps its meaning; write `if (…) { x = 1 } else { x = 0 }` to make it explicit.",
}


def _locate_in_instruction(source: str, warning: LanguageWarning, syntax: str) -> Optional[Tuple[int, int, int, int]]:
    """The place of the warning's expression: inside the instruction's source text if the expression is found
    as printed, otherwise the whole instruction (expressions carry no spans of their own)."""
    span = span_of(warning.instr)
    if span is None:
        return _locate_compact(_blank_comments(source, syntax), _spelling(warning.expr, syntax))
    lines = source.split("\n")
    start = sum(len(line) + 1 for line in lines[:span.line - 1]) + span.column - 1
    end = sum(len(line) + 1 for line in lines[:span.end_line - 1]) + span.end_column - 1
    segment = _blank_comments(source, syntax)[start:end]
    found = _locate_compact(segment, _spelling(warning.expr, syntax))
    if found is None:
        return (span.line, span.column, span.end_line, span.end_column)
    first_line, first_column, last_line, last_column = found
    base_line, base_column = _position(source, start)
    return (base_line + first_line - 1, first_column + (base_column - 1 if first_line == 1 else 0),
            base_line + last_line - 1, last_column + (base_column - 1 if last_line == 1 else 0))


def language_warning_diagnostics(program: Any, source: str, syntax: str = "ppy") -> List[Dict[str, Any]]:
    """Warnings for valid programs that are likely to surprise (monus, implicit casts), as diagnostics
    with the position of the expression; also used by the CLI."""
    try:
        warnings = language_warnings(program, syntax)
    except Exception:  # pylint: disable=broad-except  # a heuristic check must never hide the real diagnostics
        return []
    return [_diagnostic("warning", warning.message, _locate_in_instruction(source, warning, syntax), source,
                        hint=_LANGUAGE_HINTS[warning.kind]) for warning in warnings]


def type_error(program: Any, source: str, syntax: str = "ppy") -> Optional[Dict[str, Any]]:
    """The GUI's diagnostic for the first type error of ``program`` (None if it type checks); also used by the CLI."""
    checked_program = copy.deepcopy(program)
    substitute.substitute_constants(checked_program)
    failure = check_program(checked_program, spell=lambda node: _spelling(node, syntax))
    return _check_failure(source, failure, syntax) if isinstance(failure, CheckFail) else None


def check_source(source: str, syntax: str = "ppy") -> Dict[str, Any]:
    """Diagnostics and outline for ``source`` in ``syntax`` (``pgcl`` or ``ppy``; cached,
    thread-safe)."""
    key = f"{syntax}\0{source}"
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    result = _check(source, syntax)
    with _LOCK:
        _CACHE[key] = result
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return result


def _nested_inference_limits(program: Any, source: str) -> List[Dict[str, Any]]:
    """Warnings for valid programs with ``dist`` values that the analysis cannot compute yet,
    as far as they can be seen without running it: a loop that assigns a ``dist`` variable."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.nested_inference import assigned_dist_variables
    from prodigy.pgcl import LoopInstr, WhileInstr
    from prodigy.pgcl.ast.types import DistributionType
    from prodigy.pgcl.ast.walk import Walk, walk_instrs
    found = []
    bodies = [(program.instructions, program.variables)]
    bodies += [(function.instructions, {decl.var: decl.typ for decl in function.declarations})
               for function in program.functions.values()]
    for instructions, variables in bodies:
        dist_vars = {var for var, typ in variables.items() if isinstance(typ, DistributionType)}
        if not dist_vars:
            continue
        for ref in walk_instrs(Walk.DOWN, instructions):
            loop = ref.val
            if not isinstance(loop, (WhileInstr, LoopInstr)):
                continue
            assigned = assigned_dist_variables(loop.body, dist_vars)
            if not assigned:
                continue
            span = span_of(loop)
            where = (span.line, span.column, span.line, span.column + 5) if span else None
            found.append(_diagnostic(
                "warning", f"The loop assigns the dist variable {assigned[0]}; dist values in the state "
                "of a loop are not implemented yet.", where, source,
                hint="The program is valid, but the analysis will stop here. Assign it before the loop."))
    return found


def _check(source: str, syntax: str) -> Dict[str, Any]:
    diagnostics: List[Dict[str, Any]] = []
    try:
        program = parse_pgcl(source, syntax)
    except Exception as exc:  # pylint: disable=broad-except
        diagnostics.append(_parse_error(source, exc))
        return {"ok": False, "diagnostics": diagnostics, "outline": None}
    try:
        outline = outline_program(program, source, syntax)
    except Exception as exc:  # pylint: disable=broad-except
        outline = None
        diagnostics.append(_diagnostic("warning", f"Could not outline the program: {exc}", None, source))
    try:
        # same as prodigy.pgcl.compiler.compile_pgcl, without parsing a second time
        checked_program = copy.deepcopy(program)
        substitute.substitute_constants(checked_program)
        checked = check_program(checked_program, spell=lambda node: _spelling(node, syntax))
        if isinstance(checked, CheckFail):
            diagnostics.append(_check_failure(source, checked, syntax))
    except Exception as exc:  # pylint: disable=broad-except
        diagnostics.append(_diagnostic("warning", f"Type check failed: {exc}", None, source,
                                       hint="Reported by the pGCL type checker; Prodigy may still be "
                                            "able to analyse the program."))
    diagnostics.extend(language_warning_diagnostics(program, source, syntax))
    diagnostics.extend(_nested_inference_limits(program, source))
    if outline is not None and not outline["queries"]:
        line, _ = _end_of_source(source)
        diagnostics.append({"severity": "info", "line": line, "column": 1, "end_line": line, "end_column": 1,
                            "message": "No queries: add " + (
                                "query Pr[...], Ex[...], Var[...] or Cov[...]" if syntax == "ppy"
                                else "?Pr[...], ?Ex[...], ?Var[...] or ?Cov[...]")
                                       + " to get values with bounds.",
                            "hint": ""})
    return {"ok": not any(d["severity"] == "error" for d in diagnostics), "diagnostics": diagnostics,
            "outline": outline}
