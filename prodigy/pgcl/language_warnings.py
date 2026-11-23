"""
Language warnings: programs that are valid but likely to surprise (#377, #335).

The type checker (:mod:`prodigy.pgcl.check`) rejects programs without a well-defined semantics. This
module reports programs that are accepted and keep their meaning, but where the reader may expect
something else. Every warning carries the instruction it stands in (whose source span the caller
can use) and the expression it is about. The messages spell expressions in the syntax of the
program. The CLI prints the warnings on stderr; the GUI shows them as editor hints
(:mod:`prodigy.web.diagnostics`).

Two checks exist:

* ``monus``: on ``nat``, ``-`` is truncated at 0 in every intermediate result, so ``x - y + z`` is
  ``(x - y) + z`` and ``a - (b - c)`` is not ``a - b + c``. Chains of subtractions
  (``a - b - c``) and ``(a - b) * c`` are harmless and not reported, nor are subtractions of
  constants, where the numbers decide. Explicit parentheses around the subtraction
  (``(x - y) + z``) are the user's statement that the grouping is intended: no warning
  (:func:`prodigy.pgcl.positions.is_parenthesized`). Since ``a - (b - c)`` needs its parentheses,
  a parsed program never gets the subtrahend warning; it applies to programs built as ASTs.
* ``cast``: assigning a ``bool`` to a ``nat`` variable converts ``true`` to 1 and ``false`` to 0.

.. doctest::

    >>> from prodigy.pgcl.parser import parse_pgcl
    >>> program = parse_pgcl("x: nat; y: nat; z: nat; x = x - y + z; x = (x - y) + z;")
    >>> [w.message.split(":")[0] for w in language_warnings(program)]
    ['`x - y + z`']
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List

from .ast import (AsgnInstr, Binop, BinopExpr, BoolLitExpr, BoolType, Expr, Instr, NatLitExpr, NatType,
                  Program, RealLitExpr, UnopExpr, VarExpr)
from .ast.walk import Walk, walk_instr_exprs, walk_instrs
from .check import get_type
from .positions import is_parenthesized
from .ppy_printer import expr_to_ppy


@dataclass(frozen=True)
class LanguageWarning:
    """One warning: ``kind`` (``monus`` or ``cast``), the text, the instruction and the expression it is about."""

    kind: str
    message: str
    instr: Instr
    expr: Expr


def _printer(syntax: str) -> Callable[[Expr], str]:
    return expr_to_ppy if syntax == "ppy" else str


def _is_constant(program: Program, expr: Expr) -> bool:
    """Whether ``expr`` consists of literals and constants only, so its value does not depend on the state."""
    if isinstance(expr, (NatLitExpr, RealLitExpr, BoolLitExpr)):
        return True
    if isinstance(expr, VarExpr):
        return expr.var in program.constants
    if isinstance(expr, UnopExpr):
        return _is_constant(program, expr.expr)
    if isinstance(expr, BinopExpr):
        return _is_constant(program, expr.lhs) and _is_constant(program, expr.rhs)
    return False


def _truncating_subtraction(program: Program, expr: Expr) -> bool:
    """Whether ``expr`` is a ``-`` of natural numbers that can be truncated (not a constant)."""
    if not (isinstance(expr, BinopExpr) and expr.operator == Binop.MINUS):
        return False
    if _is_constant(program, expr):
        return False
    return isinstance(get_type(program, expr, check=False), NatType)


def _operand(expr: Expr, spell: Callable[[Expr], str]) -> str:
    """``expr`` as an operand of ``+``/``-``: sums and differences get parentheses."""
    text = spell(expr)
    if isinstance(expr, BinopExpr) and expr.operator in (Binop.PLUS, Binop.MINUS):
        return f"({text})"
    return text


def _monus_warnings(program: Program, instr: Instr, expr: Expr,
                    spell: Callable[[Expr], str]) -> Iterator[LanguageWarning]:
    """Explicit parentheses around the subtraction state that the grouping is intended: no warning."""
    if not isinstance(expr, BinopExpr):
        return
    if expr.operator == Binop.PLUS and _truncating_subtraction(program, expr.lhs) \
            and not is_parenthesized(expr.lhs):
        assert isinstance(expr.lhs, BinopExpr)
        keep = f"({spell(expr.lhs)}) + {_operand(expr.rhs, spell)}"
        better = f"({_operand(expr.lhs.lhs, spell)} + {_operand(expr.rhs, spell)}) - {_operand(expr.lhs.rhs, spell)}"
        yield LanguageWarning(
            "monus", f"`{spell(expr)}`: `-` is truncated at 0 before `+` is applied; write `{keep}` to keep it, "
            f"or `{better}` for integer arithmetic.", instr, expr)
    elif expr.operator == Binop.MINUS and _truncating_subtraction(program, expr.rhs) \
            and not is_parenthesized(expr.rhs):
        assert isinstance(expr.rhs, BinopExpr)
        better = f"({_operand(expr.lhs, spell)} + {_operand(expr.rhs.rhs, spell)}) - {_operand(expr.rhs.lhs, spell)}"
        yield LanguageWarning(
            "monus", f"`{spell(expr)}`: the subtraction on the right of `-` is truncated at 0 first, so this is "
            f"not `a - b + c`; write `{better}` for integer arithmetic.", instr, expr)


def _cast_warnings(program: Program, instr: Instr) -> Iterator[LanguageWarning]:
    if not isinstance(instr, AsgnInstr):
        return
    if not isinstance(program.variables.get(instr.lhs), NatType):
        return
    if isinstance(get_type(program, instr.rhs, check=False), BoolType):
        yield LanguageWarning("cast", "implicit cast from bool to nat (true → 1, false → 0)", instr, instr.rhs)


def _warnings_of(program: Program, instructions: List[Instr], syntax: str) -> Iterator[LanguageWarning]:
    spell = _printer(syntax)
    for ref in walk_instrs(Walk.DOWN, instructions):
        instr = ref.val
        yield from _cast_warnings(program, instr)
        for expr_ref in walk_instr_exprs(Walk.DOWN, instr):
            yield from _monus_warnings(program, instr, expr_ref.val, spell)


def language_warnings(program: Program, syntax: str = "ppy") -> List[LanguageWarning]:
    """The warnings of ``program`` (and its functions), in source order of their instructions.

    Call it on a program that type checks, before constants are substituted; expressions that cannot
    be typed are skipped."""
    found: List[LanguageWarning] = list(_warnings_of(program, program.instructions, syntax))
    for function in program.functions.values():
        local = Program.from_function(function, program, copy_functions=True)
        found.extend(_warnings_of(local, function.instructions, syntax))
    return found
