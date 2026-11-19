"""
The allowlist for generating functions given as text from outside: typed in by a person (input distributions,
GUI fields, CLI options) or proposed by a language model (:mod:`prodigy.analysis.ai_candidates`).

SymPy's parser evaluates Python code, so such a text reaches a computer algebra system only after
:func:`admissible_expression` accepted it: numbers, names, ``+ - * /``, powers, parentheses and calls of a few
mathematical functions, and it must parse as an expression of the program's syntax (pGCL or ``.ppy``).
"""
from __future__ import annotations

import re
from typing import AbstractSet, Optional, Sequence

from prodigy.pgcl.parser import parse_expectation

FUNCTIONS = frozenset({"sqrt", "exp"})
"""Functions a candidate of a language model may use."""
MATH_FUNCTIONS = frozenset({"sqrt", "exp", "log", "sin", "cos", "tan"})
"""Functions a generating function typed in by a person may use (see :func:`admissible_expression`)."""


class InadmissibleExpression(ValueError):
    """A text is not an admissible generating function (:func:`admissible_expression`)."""


_ALLOWED = re.compile(r"[0-9A-Za-z_+\-*/^().\s]+")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
"""A name in an expression (variables, parameters and functions)."""
_CALL = re.compile(r"([A-Za-z_]\w*)\s*\(")
_BOOLEAN_WORDS = frozenset({"and", "or", "not", "true", "false", "True", "False"})
"""Words both parsers read as Boolean operators or constants; a generating function has none."""


def to_syntax(expression: str, syntax: str) -> str:
    """``expression`` with the power operator of ``syntax`` (``^`` in pGCL, ``**`` in ``.ppy``)."""
    return expression.replace("^", "**") if syntax == "ppy" else expression.replace("**", "^")


def admissible_expression(text: str, names: Optional[Sequence[str]] = None, syntax: str = "pgcl",
                          functions: AbstractSet[str] = FUNCTIONS) -> str:
    """Checks that ``text`` is a generating function that is safe to give to a computer algebra
    system and returns it in the engines' spelling (powers as ``^``).

    ``text`` must consist of numbers, names, ``+ - * /``, powers (``^`` or ``**``, both
    syntaxes accept both), parentheses and calls of ``functions``, and parse as an expression of
    ``syntax``. With ``names``, every other name is rejected as well.

    Raises :class:`InadmissibleExpression` otherwise."""
    text = to_syntax(text, syntax)
    if not _ALLOWED.fullmatch(text):
        raise InadmissibleExpression(f"{text!r} contains characters other than numbers, names, "
                             f"+ - * / {'**' if syntax == 'ppy' else '^'} and parentheses.")
    if "//" in text:
        # floor division in .ppy and Python, a comment in pGCL: neither belongs into a generating function
        raise InadmissibleExpression(f"{text!r} contains '//', which is not an operation on generating functions.")
    called = sorted({match.group(1) for match in _CALL.finditer(text)} - set(functions))
    if called:
        raise InadmissibleExpression(f"{text!r} calls {', '.join(called)}; allowed functions are "
                             f"{', '.join(sorted(functions))}.")
    if any(name.startswith("__") for name in IDENTIFIER.findall(text)):
        raise InadmissibleExpression(f"{text!r} uses a name that starts with '__'.")
    boolean = sorted(set(IDENTIFIER.findall(text)) & _BOOLEAN_WORDS)
    if boolean:
        raise InadmissibleExpression(f"{text!r} uses {', '.join(boolean)}, which is Boolean, "
                                     "not a generating function.")
    if names is not None:
        unknown = sorted(set(IDENTIFIER.findall(text)) - set(names) - set(functions))
        if unknown:
            raise InadmissibleExpression(f"{text!r} uses {', '.join(unknown)}, which are neither program "
                                 "variables nor parameters.")
    try:
        if syntax == "ppy":
            # pylint: disable=import-outside-toplevel
            from prodigy.pgcl.ppy_parser import parse_ppy_expr
            parse_ppy_expr(text)
        else:
            parse_expectation(text)
    except Exception as exc:  # pylint: disable=broad-except  # lark raises several error types
        raise InadmissibleExpression(f"{text!r} is not a well-formed expression: {exc}") from exc
    return text.replace("**", "^")
