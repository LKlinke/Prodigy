from __future__ import annotations

from typing import Optional, Union

import attr

from .ast import Node
from .expressions import Expr


@attr.s
class Bounds:
    """
    Bounds for a natural number type.

    The bounds can contain constant expressions, therefore bounds have type :class:`Expr`.
    """
    lower: Expr = attr.ib()
    upper: Expr = attr.ib()

    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"


@attr.s
class TypeClass(Node):
    """Superclass for all types. See :obj:`Type`."""


@attr.s
class BoolType(TypeClass):
    """Boolean type."""


@attr.s
class NatType(TypeClass):
    """
    Natural number types with optional bounds.

    Bounds are only preserved for variables.
    Values of bounded types are considered as unbounded until they are assigned to a bounded variable.
    That is to say, bounds are lost in expressions such as in the example below:

    .. doctest::

        >>> from probably.pgcl.parser import parse_pgcl, parse_expr
        >>> from probably.pgcl.check import get_type
        >>> program = parse_pgcl("nat x [1,5]")
        >>> get_type(program, parse_expr("x + 5"))
        NatType(bounds=None)
    """

    bounds: Optional[Bounds] = attr.ib()


@attr.s
class RealType(TypeClass):
    """Real numbers, used for probabilities."""


Type = Union[BoolType, NatType, RealType]
"""Union type for all type objects. See :class:`TypeClass` for use with isinstance."""
