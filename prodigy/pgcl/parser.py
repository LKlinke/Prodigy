"""
------
Parser
------

:func:`parse_pgcl` reads a program in the ``.ppy`` syntax (:mod:`prodigy.pgcl.ppy_parser`), lowers
its block-scoped declarations into plain assignments (:func:`prodigy.pgcl.scope.lower_block_scope`)
and rejects a guard that is no ``bool``. The original pGCL syntax (``.pgcl``) and its parser, taken
over from probably, were removed in v0.6.0 (#433); the function keeps its name until the package is
renamed (#446).
"""
import os
from pathlib import Path
from typing import Optional, Union

from prodigy.pgcl.ast import Program
from prodigy.pgcl.syntax_errors import ProgramTypeError

PGCL_REMOVED = "The .pgcl syntax was removed in v0.6.0; programs are written in .ppy."
"""The answer of :func:`parse_pgcl` to a ``.pgcl`` program (the CLI, the GUI and the API answer with
:data:`prodigy.analysis.deprecations.PGCL_REMOVED`, which says how to convert one)."""


def parse_pgcl(code: str,
               syntax: Optional[str] = None,
               *,
               path: Union[str, os.PathLike, None] = None,
               check_guards: bool = True) -> Program:
    """
    Parse a program in the ``.ppy`` syntax. ``syntax`` (``"ppy"``) and ``path`` are only checked:
    ``"pgcl"`` or a path ending in ``.pgcl`` raise ``ValueError``.

    .. doctest::

        >>> parse_pgcl("x = y;").instructions
        [AsgnInstr(lhs='x', rhs=VarExpr('y'))]

        >>> parse_pgcl("x: nat;", "pgcl")
        Traceback (most recent call last):
            ...
        ValueError: The .pgcl syntax was removed in v0.6.0; programs are written in .ppy.
    """
    if syntax == "pgcl" or (path is not None and Path(path).suffix == ".pgcl"):
        raise ValueError(PGCL_REMOVED)
    if syntax not in (None, "ppy"):
        raise ValueError(f"Unknown syntax {syntax!r}, expected ppy.")
    # imported here: ppy_parser builds its own Lark parser on import
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.ppy_parser import parse_ppy
    from prodigy.pgcl.scope import lower_block_scope
    program = lower_block_scope(parse_ppy(code))
    if check_guards:
        _check_guards(program)
    return program


def _check_guards(program: Program) -> None:
    """Rejects a guard that is no ``bool`` (:func:`prodigy.pgcl.check.non_bool_guard`)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.check import non_bool_guard, type_name
    from prodigy.pgcl.draws import find_draw
    from prodigy.pgcl.positions import span_of
    from prodigy.pgcl.ppy_printer import PpyPrintError, expr_to_ppy
    found = non_bool_guard(program)
    if found is None:
        return
    instr, guard, what, typ = found
    try:
        spelled = expr_to_ppy(guard)
    except PpyPrintError:
        spelled = str(guard)
    if find_draw(guard) is not None:
        hint = "Draw the value into a variable first and compare the variable, e.g. with > 0."
    else:
        hint = f"Write a comparison such as {spelled} > 0."
    span = span_of(instr)
    raise ProgramTypeError(
        f"Type error: {what} must be of type bool, but {spelled} is of type {type_name(typ)}. "
        + hint,
        None if span is None else span.line, None if span is None else span.column)
