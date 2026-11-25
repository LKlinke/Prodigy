from abc import ABC
from typing import Callable, List

import attr

Var = str

_PRINTER: List[Callable[[object], str]] = []


def install_printer(printer: Callable[[object], str]) -> None:
    """
    Makes ``printer`` the ``str()`` of every node. :mod:`prodigy.language` installs the ``.ppy`` printer
    (:func:`prodigy.language.ppy_printer.node_to_ppy`) when it is imported; the AST itself imports no printer,
    so that the printer can import the AST (``tests/test_layering.py``).
    """
    _PRINTER[:] = [printer]


def printed(node: object) -> str:
    """``node`` as the installed printer writes it (``repr`` if none is installed yet)."""
    if _PRINTER:
        return _PRINTER[0](node)
    return repr(node)


@attr.s
class Node(ABC):
    """Superclass for all node types in the AST."""

    def __str__(self) -> str:
        """
        The node in the ``.ppy`` syntax (the installed printer); ``repr`` while :mod:`prodigy.language` is still
        being imported. Expressions go to an engine only as engine text
        (:func:`prodigy.language.engine_text.to_engine_text`), never through ``str()``.

        .. doctest::

            >>> from prodigy.language.ppy_parser import parse_ppy_expr
            >>> str(parse_ppy_expr("x < 2 and not true"))
            'x < 2 and not true'
        """
        return printed(self)
