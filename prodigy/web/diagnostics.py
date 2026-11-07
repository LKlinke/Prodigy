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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import copy

from prodigy.pgcl import parse_pgcl, substitute
from prodigy.pgcl.check import CheckFail, check_program

from prodigy.analysis.result import outline_program

_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_SIZE = 64
_LOCK = threading.Lock()

_FRIENDLY = {"CNAME": "a name", "INT": "a number", "FLOAT": "a number"}
_FIRST = ("RPAR", "RBRACE", "RSQB", "LBRACE", "__ANON_0")


def _terminal_texts() -> Dict[str, str]:
    try:
        from prodigy.pgcl.parser import _PARSER  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return {}
    texts = {}
    for terminal in _PARSER.terminals:
        if terminal.name in _FRIENDLY:
            texts[terminal.name] = _FRIENDLY[terminal.name]
        elif type(terminal.pattern).__name__ == "PatternStr":
            texts[terminal.name] = f"'{terminal.pattern.value.strip()}'"
    return texts


_TERMINALS = _terminal_texts()


def _expected(names: Iterable[Any], limit: int = 4) -> str:
    names = [getattr(n, "name", str(n)) for n in names]
    ordered = [n for n in _FIRST if n in names] + sorted(n for n in names if n not in _FIRST)
    texts: List[str] = []
    for name in ordered:
        text = _TERMINALS.get(name)
        if text and text not in texts:
            texts.append(text)
    if not texts:
        return ""
    shown = ", ".join(texts[:limit]) + (", …" if len(texts) > limit else "")
    return f"expected {shown}"


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


def _locate_name(source: str, name: str, syntax: str = "pgcl") -> Optional[Tuple[int, int, int, int]]:
    match = re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", _blank_comments(source, syntax))
    if match is None:
        return None
    return (*_position(source, match.start()), *_position(source, match.end()))


def _blank_comments(source: str, syntax: str = "pgcl") -> str:
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
    if name in ("PpySyntaxError", "ProgramSyntaxError"):
        line, column = getattr(exc, "line", None), getattr(exc, "column", None)
        message = _sentence(getattr(exc, "message", str(exc)))
        if line is None:
            return _diagnostic("error", message, None, source)
        return _diagnostic("error", message, (line, column or 1, line, (column or 1) + 1), source)
    if name == "UnexpectedCharacters":
        line, column = exc.line, exc.column  # type: ignore[attr-defined]
        offset = exc.pos_in_stream  # type: ignore[attr-defined]
        char = source[offset] if offset is not None and offset < len(source) else ""
        expected = _expected(getattr(exc, "allowed", None) or [])
        message = _sentence(f"Unexpected '{char}'" + (f"; {expected}" if expected else ""))
        return _diagnostic("error", message, (line, column, line, column + 1), source)
    if name == "UnexpectedToken":
        token = exc.token  # type: ignore[attr-defined]
        expected = _expected(getattr(exc, "expected", None) or [])
        end_column = getattr(token, "end_column", None) or token.column + len(str(token))
        message = _sentence(f"Unexpected '{token}'" + (f"; {expected}" if expected else ""))
        return _diagnostic("error", message, (token.line, token.column, token.line, end_column), source)
    if name == "UnexpectedEOF":
        expected = _expected(getattr(exc, "expected", None) or [])
        line, column = _end_of_source(source)
        message = _sentence("The program ends unexpectedly" + (f"; {expected}" if expected else ""))
        return _diagnostic("error", message, (line, max(1, column - 1), line, column), source)
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


def _check_failure(source: str, failure: CheckFail, syntax: str = "pgcl") -> Dict[str, Any]:
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


def check_source(source: str, syntax: str = "pgcl") -> Dict[str, Any]:
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
        checked = check_program(checked_program)
        if isinstance(checked, CheckFail):
            diagnostics.append(_check_failure(source, checked, syntax))
    except Exception as exc:  # pylint: disable=broad-except
        diagnostics.append(_diagnostic("warning", f"Type check failed: {exc}", None, source,
                                       hint="Reported by the pGCL type checker; Prodigy may still be "
                                            "able to analyse the program."))
    if outline is not None and not outline["queries"]:
        line, _ = _end_of_source(source)
        diagnostics.append({"severity": "info", "line": line, "column": 1, "end_line": line, "end_column": 1,
                            "message": "No queries: add " + ("query Pr[...] or query Ex[...]" if syntax == "ppy"
                                                             else "?Pr[...] or ?Ex[...]")
                                       + " to get values with bounds.",
                            "hint": ""})
    return {"ok": not any(d["severity"] == "error" for d in diagnostics), "diagnostics": diagnostics,
            "outline": outline}
