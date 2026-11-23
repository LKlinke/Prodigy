"""
The tests of the ``.pgcl`` front end (taken over from probably) read programs in the old syntax. Programs
are ``.ppy`` by default since v0.6.0, and the CLI, the GUI and the API reject ``.pgcl`` (#433); the
parser itself goes in a later step of #433, and these tests are converted or removed with it.
"""
from typing import Any, Optional

from prodigy.pgcl import compiler, parser


def _syntax(syntax: Optional[str], kwargs: Any) -> Optional[str]:
    return "pgcl" if syntax is None and kwargs.get("path") is None else syntax


def parse_pgcl(code: str, syntax: Optional[str] = None, **kwargs: Any) -> Any:
    """:func:`prodigy.pgcl.parser.parse_pgcl` with ``pgcl`` as the default syntax (without a path)."""
    return parser.parse_pgcl(code, _syntax(syntax, kwargs), **kwargs)


def compile_pgcl(code: str, *, syntax: Optional[str] = None, **kwargs: Any) -> Any:
    """:func:`prodigy.pgcl.compiler.compile_pgcl` with ``pgcl`` as the default syntax (without a path)."""
    return compiler.compile_pgcl(code, syntax=_syntax(syntax, kwargs), **kwargs)
