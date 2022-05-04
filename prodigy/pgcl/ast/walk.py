"""
------------
Walking ASTs
------------

Utilities to traverse ("walk") an AST, i.e. visiting each node in the AST.
There are two directions (bottom-up and top-down) which you can select using :class:`Walk`.

We also have :data:`prodigy.util.ref.Ref` and :class:`prodigy.util.ref.Mut` to represent mutable references to values.
These allow a walk of an AST while modifying it.
E.g. constant folding can be implemented as a bottom-up walk where each callback simplifies the current node.

A simple example with :func:`walk_expr` is below.
The returned iterable contains ``Mut[Expr]`` objects.
Use :attr:`prodigy.util.ref.Mut.val` to access the current node.

    .. doctest::

        >>> from prodigy.util.ref import Mut
        >>> expr = Mut.alloc(UnopExpr(Unop.NEG, NatLitExpr(10)))

        >>> [ref.val for ref in walk_expr(Walk.DOWN, expr)]
        [UnopExpr(operator=Unop.NEG, expr=NatLitExpr(10)), NatLitExpr(10)]

        >>> [ref.val for ref in walk_expr(Walk.UP, expr)]
        [NatLitExpr(10), UnopExpr(operator=Unop.NEG, expr=NatLitExpr(10))]


.. rubric:: Why not manual recursion?

This framework allows easy and flexible traversal of ASTs while also changing direction along the way: Using :class:`prodigy.util.ref.Mut`, we can modify the AST and then continue traversal with the new AST.
This makes it hard to forget calling a recursive traversal call, which could easily happen if every single traversal was implemented manually.
"""

from enum import Enum, auto
from typing import Callable, Iterable, TypeVar, List

from .instructions import Instr, InstrClass
from .expressions import Expr, ExprClass
from prodigy.util.ref import Mut


T = TypeVar("T")


class Walk(Enum):
    """
    In which direction to go in a traversal: DOWN is a top-down traversal and UP is a bottom-up traversal.
    """
    DOWN = auto()
    UP = auto()

    def walk(self, get_children: Callable[[T], Iterable[T]],
             root: T) -> Iterable[T]:
        """
        Walk a tree given by ``get_children`` in the given direction.
        """
        if self == Walk.DOWN:
            yield root

        for child in get_children(root):
            yield from self.walk(get_children, child)

        if self == Walk.UP:
            yield root


def mut_expr_children(node_ref: Mut[Expr]) -> Iterable[Mut[Expr]]:
    """Get refs to all direct children of an expr."""
    node = node_ref.val
    for ref in Mut.dict_values(node.__dict__):
        if not isinstance(ref.val, ExprClass):
            continue

        yield ref


def walk_expr(walk: Walk, expr_ref: Mut[Expr]) -> Iterable[Mut[Expr]]:
    """Walk an expression."""
    assert isinstance(expr_ref.val, ExprClass)
    return walk.walk(mut_expr_children, expr_ref)


def _mut_instr_children(node_ref: Mut[Instr]) -> Iterable[Mut[Instr]]:
    """
    Get refs to all direct children of an instr, including those in attributes that are lists.

    .. doctest::

        >>> list(_mut_instr_children(Mut.alloc(ChoiceInstr(VarExpr('x'), [SkipInstr()], [SkipInstr()]))))
        [Mut(val=SkipInstr()), Mut(val=SkipInstr())]
    """
    node = node_ref.val
    for ref in Mut.dict_values(node.__dict__):
        for item_ref in Mut.iterate(ref):
            if not isinstance(item_ref.val, InstrClass):
                continue

            yield item_ref


def walk_instrs(walk: Walk, instrs: List[Instr]) -> Iterable[Mut[Instr]]:
    """Walk a list of instructions."""
    for instr in Mut.list(instrs):
        assert isinstance(instr.val, InstrClass)
        yield from walk.walk(_mut_instr_children, instr)


def instr_exprs(node: Instr) -> Iterable[Expr]:
    """
    Get the expressions that are direct children of this instruction as values.
    """
    for value in node.__dict__.values():
        if isinstance(value, ExprClass):
            yield value.cast()


def mut_instr_exprs(node: Instr) -> Iterable[Mut[Expr]]:
    """
    Get refs to the expressions of an instruction.
    """
    for ref in Mut.dict_values(node.__dict__):
        if not isinstance(ref.val, ExprClass):
            continue

        yield ref


def walk_instr_exprs(walk: Walk, instr: Instr) -> Iterable[Mut[Expr]]:
    """Walk all expressions in the instruction."""
    assert isinstance(instr, InstrClass)
    for expr_ref in mut_instr_exprs(instr):
        yield from walk_expr(walk, expr_ref)
