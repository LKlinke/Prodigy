"""
----------------------------
Converter from .pgcl to .ppy
----------------------------

Translates a program in the original pGCL syntax into the Python-like ``.ppy``
syntax (``prodigy convert``). The program is parsed into the AST and printed by
:class:`prodigy.pgcl.ppy_printer.PpyPrinter`, so parsing the result with
:func:`prodigy.pgcl.ppy_parser.parse_ppy` gives the same AST, up to renamed names.

Names that are keywords in ``.ppy`` (e.g. a variable ``switch``) are renamed by
appending ``_`` until the name is neither a keyword nor used elsewhere in the
program (:func:`keyword_renamings`). Every renaming is reported.

Comments are not part of the AST. The converter carries them over by their
lines: a comment on a line of its own is placed before the next statement or
declaration, a comment after code is placed after the statement that ends on its
line. Their position is therefore approximate, e.g. a comment at the end of a
block moves behind the block. ``//`` comments become ``#`` comments.

.. doctest::

    >>> result = convert_pgcl("nat switch; // the door\\nswitch := 1\\n?Pr[switch = 1]")
    >>> print(result.text, end="")
    switch_: nat;  # the door
    switch_ = 1;
    query Pr[switch_ == 1];
    >>> result.renamings
    {'switch': 'switch_'}
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar

import attr

from prodigy.pgcl.ast import (ChoiceInstr, Decl, FunctionDecl, IfInstr, Instr,
                              LoopInstr, Program, QueryInstr, WhileInstr)
from prodigy.pgcl.parser import parse_pgcl_with_lines
from prodigy.pgcl.ppy_parser import PPY_KEYWORDS
from prodigy.pgcl.ppy_printer import PpyPrinter

T = TypeVar("T")

_COMMENT = re.compile(r"#|//")


@dataclass
class Conversion:
    """The result of :func:`convert_pgcl`."""

    text: str
    """The program in the ``.ppy`` syntax."""

    renamings: Dict[str, str] = field(default_factory=dict)
    """Old name -> new name for every name that is a keyword in ``.ppy``."""

    comments: int = 0
    """Number of comments carried over."""

    def messages(self) -> List[str]:
        """One line for every renaming, for the user."""
        return [f"renamed '{old}' to '{new}' ('{old}' is a keyword in .ppy)"
                for old, new in self.renamings.items()]


def _names(node: Any, out: Set[str]) -> None:
    """Collects all names of an AST. Every string in the AST is a name."""
    if isinstance(node, str):
        out.add(node)
    elif isinstance(node, (list, tuple, set, frozenset)):
        for item in node:
            _names(item, out)
    elif isinstance(node, dict):
        for key, value in node.items():
            _names(key, out)
            _names(value, out)
    elif attr.has(type(node)):
        for attribute in attr.fields(type(node)):
            _names(getattr(node, attribute.name), out)


def keyword_renamings(program: Program) -> Dict[str, str]:
    """
    New names for the names of ``program`` that are keywords in ``.ppy``: the name
    with ``_`` appended, as often as needed to avoid keywords and the other names
    of the program. The result does not depend on the order of the declarations.

    .. doctest::

        >>> from prodigy.pgcl.parser import parse_pgcl
        >>> keyword_renamings(parse_pgcl("nat switch; nat switch_; nat case; switch := case + switch_"))
        {'case': 'case_', 'switch': 'switch__'}
    """
    names: Set[str] = set()
    _names(program, names)
    taken = set(names)
    renamings: Dict[str, str] = {}
    for name in sorted(names & PPY_KEYWORDS):
        new = name + "_"
        while new in taken or new in PPY_KEYWORDS:
            new += "_"
        taken.add(new)
        renamings[name] = new
    return renamings


def rename(node: T, renamings: Dict[str, str]) -> T:
    """
    A copy of an AST node (or a program) with the names renamed by ``renamings``.

    .. doctest::

        >>> from prodigy.pgcl.parser import parse_pgcl
        >>> rename(parse_pgcl("nat a; a := a + 1"), {"a": "b"}).instructions
        [AsgnInstr(lhs='b', rhs=BinopExpr(operator=Binop.PLUS, lhs=VarExpr('b'), rhs=NatLitExpr(1)))]
    """
    if not renamings:
        return copy.deepcopy(node)
    return _rename(node, renamings)


def _rename(node: Any, renamings: Dict[str, str]) -> Any:
    # pylint: disable=too-many-return-statements
    if isinstance(node, str):
        return renamings.get(node, node)
    if isinstance(node, list):
        return [_rename(item, renamings) for item in node]
    if isinstance(node, tuple):
        return tuple(_rename(item, renamings) for item in node)
    if isinstance(node, (set, frozenset)):
        return type(node)(_rename(item, renamings) for item in node)
    if isinstance(node, dict):
        return {_rename(key, renamings): _rename(value, renamings) for key, value in node.items()}
    if attr.has(type(node)):
        result = copy.copy(node)
        for attribute in attr.fields(type(node)):
            # object.__setattr__ also works for frozen classes
            object.__setattr__(result, attribute.name, _rename(getattr(node, attribute.name), renamings))
        return result
    return node


# ---------------------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------------------

@dataclass
class _Comment:
    line: int
    text: str
    trailing: bool  # after code on the same line


def _comments(source: str) -> List[_Comment]:
    comments = []
    for number, line in enumerate(source.split("\n"), start=1):
        match = _COMMENT.search(line)
        if match is None:
            continue
        text = line[match.start():].rstrip()
        if text.startswith("//"):
            text = "#" + text[2:]
        comments.append(_Comment(number, text, line[:match.start()].strip() != ""))
    return comments


def _children(instr: Instr) -> Iterator[List[Instr]]:
    if isinstance(instr, (WhileInstr, LoopInstr)):
        yield instr.body
    elif isinstance(instr, IfInstr):
        yield instr.true
        yield instr.false
    elif isinstance(instr, ChoiceInstr):
        yield instr.lhs
        yield instr.rhs
    elif isinstance(instr, QueryInstr):
        yield instr.instrs


def _preorder_instrs(instrs: List[Instr]) -> Iterator[Instr]:
    for instr in instrs:
        yield instr
        for block in _children(instr):
            yield from _preorder_instrs(block)


def _anchors(program: Program) -> Iterator[Any]:
    """The declarations and instructions that the printer prints on their own, in source order."""
    decl: Decl
    for decl in program.declarations:
        yield decl
        if isinstance(decl, FunctionDecl):
            # the parameters are printed in the header, comments go to the body
            yield from _preorder_instrs(decl.body.instructions)
    yield from _preorder_instrs(program.instructions)


class _CommentPrinter(PpyPrinter):
    """Prints the comments attached to a node before or after its text."""

    def __init__(self, before: Dict[int, List[str]], after: Dict[int, List[str]]):
        self.before = before
        self.after = after

    def _decorate(self, node: Any, text: str) -> str:
        lines = self.before.get(id(node), [])
        trailing = self.after.get(id(node), [])
        if trailing:
            text += "  " + "  ".join(trailing)
        return "\n".join(lines + [text])

    def instr(self, instr: Instr) -> str:
        return self._decorate(instr, super().instr(instr))

    def decl(self, decl: Decl) -> str:
        return self._decorate(decl, super().decl(decl))


def _place_comments(program: Program, lines: Dict[int, Tuple[int, int]], source: str) \
        -> Tuple[Dict[int, List[str]], Dict[int, List[str]], List[str], int]:
    """
    The lines to print before and after a node (by ``id()``), the lines after the
    program and the number of comments. An empty line stands for an empty line in
    the source before a comment or a node.
    """
    source_lines = source.split("\n")

    def blank_before(line: int) -> List[str]:
        return [""] if line > 1 and source_lines[line - 2].strip() == "" else []

    anchors = [(node, *lines[id(node)]) for node in _anchors(program) if id(node) in lines]
    before: Dict[int, List[str]] = {}
    after: Dict[int, List[str]] = {}
    tail: List[str] = []
    comments = _comments(source)
    for comment in comments:
        if comment.trailing:
            ending = [node for node, _, last in anchors if last == comment.line]
            if ending:
                after.setdefault(id(ending[-1]), []).append(comment.text)
                continue
        following: Optional[Any] = next((node for node, first, _ in anchors if first > comment.line), None)
        target = tail if following is None else before.setdefault(id(following), [])
        target.extend(blank_before(comment.line) + [comment.text])
    for node, first, _ in anchors:
        if not before.get(id(node)):
            before[id(node)] = blank_before(first)
    return before, after, tail, len(comments)


def _tidy(text: str) -> str:
    """Drops empty lines at the start, after an opening brace and after another empty line."""
    lines: List[str] = []
    for line in text.split("\n"):
        if line == "" and (not lines or lines[-1] == "" or lines[-1].endswith("{")):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip("\n") + "\n"


def convert_pgcl(source: str) -> Conversion:
    """
    Convert a program from the original pGCL syntax into the ``.ppy`` syntax,
    renaming names that are keywords in ``.ppy`` and keeping the comments.

    Raises:
        lark.exceptions.LarkError: if ``source`` is not a valid pGCL program.
        prodigy.pgcl.ppy_printer.PpyPrintError: if the program has no ``.ppy`` form.
    """
    original, lines = parse_pgcl_with_lines(source)
    renamings = keyword_renamings(original)
    # rename returns new nodes; translate the line table to them
    program = rename(original, renamings)
    new_lines: Dict[int, Tuple[int, int]] = {}
    for old_node, new_node in zip(_anchors(original), _anchors(program)):
        entry = lines.get(id(old_node))
        if entry is not None and entry[0] is old_node:
            new_lines[id(new_node)] = (entry[1], entry[2])
    before, after, tail, comments = _place_comments(program, new_lines, source)
    text = _CommentPrinter(before, after).program(program) + "\n".join(tail)
    return Conversion(text=_tidy(text), renamings=renamings, comments=comments)
