"""
Diagnostics for editors: parses and type-checks a ``.ppy`` program and reports
problems with positions (:mod:`prodigy.language.diagnostics`, :mod:`prodigy.analysis.checks`), plus an
outline of the program (variables, loops with their ids, queries).

Used by the web GUI; a language server can use it as well.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List

import copy

from prodigy.language import parse_program, substitute
from prodigy.language.check import CheckFail
from prodigy.language.diagnostics import (check_failure_diagnostic, diagnostic, end_of_source,
                                          language_warning_diagnostics, parse_error_diagnostic)
from prodigy.language.outline import outline as outline_of
from prodigy.language.positions import span_of

from prodigy.analysis.checks import first_failure

_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_SIZE = 64
_LOCK = threading.Lock()


def check_source(source: str) -> Dict[str, Any]:
    """Diagnostics and outline for the ``.ppy`` program ``source`` (cached, thread-safe)."""
    key = source
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    result = _check(source)
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
    from prodigy.language import LoopInstr, WhileInstr
    from prodigy.language.ast.types import DistributionType
    from prodigy.language.ast.walk import Walk, walk_instrs
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
            found.append(diagnostic(
                "warning", f"The loop assigns the dist variable {assigned[0]}; dist values in the state "
                "of a loop are not implemented yet.", where, source,
                hint="The program is valid, but the analysis will stop here. Assign it before the loop."))
    return found


def _check(source: str) -> Dict[str, Any]:
    diagnostics: List[Dict[str, Any]] = []
    try:
        program = parse_program(source)
    except Exception as exc:  # pylint: disable=broad-except
        diagnostics.append(parse_error_diagnostic(source, exc))
        return {"ok": False, "diagnostics": diagnostics, "outline": None}
    try:
        outline = outline_of(program).to_json()
    except Exception as exc:  # pylint: disable=broad-except
        outline = None
        diagnostics.append(diagnostic("warning", f"Could not outline the program: {exc}", None, source))
    try:
        # same as prodigy.language.compiler.compile_program, without parsing a second time
        checked_program = copy.deepcopy(program)
        substitute.substitute_constants(checked_program)
        checked = first_failure(checked_program)
        if isinstance(checked, CheckFail):
            diagnostics.append(check_failure_diagnostic(source, checked))
    except Exception as exc:  # pylint: disable=broad-except
        diagnostics.append(diagnostic("warning", f"Type check failed: {exc}", None, source,
                                       hint="Reported by the type checker; Prodigy may still be "
                                            "able to analyse the program."))
    diagnostics.extend(language_warning_diagnostics(program, source))
    diagnostics.extend(_nested_inference_limits(program, source))
    if outline is not None and not outline["queries"]:
        line, _ = end_of_source(source)
        diagnostics.append({"severity": "info", "line": line, "column": 1, "end_line": line, "end_column": 1,
                            "message": "No queries: add query Pr[...], Ex[...], Var[...] or Cov[...] to get "
                                       "values with bounds.",
                            "hint": ""})
    return {"ok": not any(d["severity"] == "error" for d in diagnostics), "diagnostics": diagnostics,
            "outline": outline}
