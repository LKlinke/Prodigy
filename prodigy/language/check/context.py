"""
What every rule of the type checker shares: the result of a failed check (:class:`CheckFail`), the context of a
check (:class:`Ctx`) and the two dispatch functions, :func:`type_of` for expressions and :func:`check_node` for
instructions. Each node class has its own small handler, registered with :func:`functools.singledispatch` in
:mod:`~prodigy.language.check.expressions` and :mod:`~prodigy.language.check.instructions`; adding a construct
adds one handler there (``tests/language/test_check_dispatch.py`` checks that every node class has one).
"""
# pylint: disable=broad-exception-raised  # an unknown node class is a bug of the checker, not of the program
from __future__ import annotations

import functools
from typing import Iterable, List, Mapping, Optional, TypeVar, Union, cast

import attr

from prodigy.language.ast import Expr, Instr, Node, Program, Type, Var, VarDecl

from .compatibility import is_compatible

_T = TypeVar('_T')


def spell_node(node: Node) -> str:
    """``node`` as ``.ppy`` text in a message."""
    # pylint: disable=import-outside-toplevel  # the printer imports the parser, which is built on import
    from prodigy.language.ppy_printer import spell
    return spell(cast(Union[Expr, Instr], node))


@attr.s
class CheckFail:
    """
    A check failed with an error message and an optional location.
    """

    location: Optional[Node] = attr.ib()
    message: str = attr.ib()

    @staticmethod
    def expected_type_got(node: Node, expected: Type,
                          got: Type) -> 'CheckFail':
        return CheckFail(node,
                         f'Expected value of type {expected}, got {got}.')

    @staticmethod
    def expected_numeric_got(node: Node, got: Type) -> 'CheckFail':
        return CheckFail(node, f'Expected numeric value, got {got}.')

    @staticmethod
    def fmap(
        source: Iterable[Union[_T,
                               'CheckFail']]) -> Union['CheckFail', List[_T]]:
        """
        Extract an error from an iterable.

        .. doctest::

            >>> CheckFail.fmap(['a'])
            ['a']
            >>> CheckFail.fmap([CheckFail(None, 'whoops!')])
            CheckFail(location=None, message='whoops!')
        """
        output = []
        for step in source:
            if isinstance(step, CheckFail):
                return step
            output.append(step)
        return output


CheckedType = Union[Type, CheckFail]
"""The type of an expression, or why it has none."""


@attr.s(frozen=True, auto_attribs=True)
class Ctx:
    """
    What a rule of the checker needs besides its node: the program (its variables, parameters, constants and
    functions), the signatures of the predefined distributions, and whether the operands of an expression are
    checked as well (``check``; ``False`` only asks for the type, see
    :func:`~prodigy.language.check.expressions.get_type`).
    """
    program: Program
    predefined_functions: Mapping[Var, List[VarDecl]]
    check: bool = True

    def checked(self) -> Ctx:
        """This context with ``check`` on."""
        return self if self.check else attr.evolve(self, check=True)

    def unchecked(self) -> Ctx:
        """This context with ``check`` off."""
        return attr.evolve(self, check=False) if self.check else self


@functools.singledispatch
def type_of(expr: Expr, ctx: Ctx) -> CheckedType:
    """The type of ``expr``: the handler of its class (:mod:`~prodigy.language.check.expressions`)."""
    raise Exception(f"unreachable: the type checker has no rule for {type(expr).__name__}")


@functools.singledispatch
def check_node(instr: Instr, ctx: Ctx) -> Optional[CheckFail]:
    """Check one instruction: the handler of its class (:mod:`~prodigy.language.check.instructions`)."""
    raise Exception(f"unreachable: the type checker has no rule for {type(instr).__name__}")


def expect(ctx: Ctx, expr: Expr, expected: Type) -> Optional[CheckFail]:
    """Why ``expr`` is not a value of type ``expected`` (or of a narrower type), or ``None`` if it is one."""
    typ = type_of(expr, ctx)
    if isinstance(typ, CheckFail):
        return typ
    if not is_compatible(expected, typ):
        return CheckFail.expected_type_got(expr, expected, typ)
    return None


def check_block(ctx: Ctx, *blocks: List[Instr]) -> Optional[CheckFail]:
    """The first failure of the instructions of ``blocks``, in order, or ``None``."""
    for block in blocks:
        for instr in block:
            failure = check_node(instr, ctx)
            if failure is not None:
                return failure
    return None
