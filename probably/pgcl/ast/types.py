import attr
from .ast import Node
from typing import Optional, Union


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
    """

    bounds: Optional['Bounds'] = attr.ib()


@attr.s
class RealType(TypeClass):
    """
    Real number types.

    They are used both to represent probabilities and as values in the program if they are allowed (see :py:data:`ProgramConfig.allow_real_vars`).
    """

Type = Union[BoolType, NatType, RealType]
"""Union type for all type objects. See :class:`TypeClass` for use with isinstance."""