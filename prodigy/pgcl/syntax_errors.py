"""
-------------
Syntax errors
-------------

:class:`ProgramSyntaxError` is a syntax error with line and column; both parsers raise it (the
``.ppy`` parser as its subclass :class:`~prodigy.pgcl.ppy_parser.PpySyntaxError`), and
:class:`ProgramTypeError` for the type errors they reject (a guard that is no ``bool``).

:func:`removed_construct` recognizes constructs that are no longer part of the language, so that a
failed parse can say so instead of reporting an unexpected character: ``tick(e)`` and Iverson
brackets ``[g]`` (a condition as the number 0 or 1). Both came from probably and were never
supported by Prodigy's analysis.

.. doctest::

    >>> removed_construct("x := 1;\\ntick(x);", 12, expects_operand=False)
    ('tick(e) is not supported', 8)
    >>> removed_construct("x := [x = 1];", 5, expects_operand=True)[0]
    "Iverson brackets '[g]' are not supported"
    >>> removed_construct("x := a[1];", 6, expects_operand=False) is None
    True
"""
import re
from typing import Optional, Tuple

TICK_NOT_SUPPORTED = "tick(e) is not supported"
"""The message for a program that uses ``tick(e)``."""

IVERSON_NOT_SUPPORTED = "Iverson brackets '[g]' are not supported"
"""The message for a program that uses an Iverson bracket."""

_TICK_BEFORE = re.compile(r"\btick\s*$")


class ProgramSyntaxError(SyntaxError):
    """A syntax error in a program, with line and column (1-based)."""
    def __init__(self, message: str, line: Optional[int],
                 column: Optional[int]):
        self.message = message
        self.line = line
        self.column = column
        where = "" if line is None else f"line {line}, column {column}: "
        self.text_message = where + message
        super().__init__(self.text_message)
        self.lineno = line
        self.offset = column

    def __str__(self) -> str:
        return self.text_message


class ProgramTypeError(ProgramSyntaxError):
    """A type error that the parser reports with line and column, e.g. a guard that is no
    ``bool`` (see :func:`prodigy.pgcl.check.check_guards`)."""


def removed_construct(source: str, offset: Optional[int], expects_operand: bool) -> Optional[Tuple[str, int]]:
    """
    The message and the start offset of a removed construct, if the parse of ``source`` failed at
    ``offset`` because of one; otherwise ``None``. ``expects_operand`` says whether the parser
    expected the start of an operand (a number, a name, …) at ``offset``.
    """
    if offset is None or not 0 <= offset < len(source):
        return None
    if source[offset] == "(":
        match = _TICK_BEFORE.search(source, 0, offset)
        if match is not None:
            return TICK_NOT_SUPPORTED, match.start()
    if source[offset] == "[" and expects_operand:
        return IVERSON_NOT_SUPPORTED, offset
    return None


def position(source: str, offset: int) -> Tuple[int, int]:
    """Line and column (1-based) of ``offset`` in ``source``."""
    line = source.count("\n", 0, offset) + 1
    return line, offset - (source.rfind("\n", 0, offset) + 1) + 1
