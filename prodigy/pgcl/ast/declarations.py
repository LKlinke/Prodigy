from __future__ import annotations

from abc import abstractmethod
from typing import Union

import attr

from .ast import Node, Var
from .expressions import Expr
from .types import BoolType, NatType, RealType, Type


class DeclClass(Node):
    """Superclass for all declarations. See :obj:`Decl`."""
    @abstractmethod
    def __str__(self) -> str:
        """
        Convert this declaration to corresponding source code in pGCL.

        .. doctest::

            >>> from .types import Bounds
            >>> str(VarDecl('x', NatType(Bounds(1, 10))))
            'nat x [1, 10];'
        """


@attr.s
class VarDecl(DeclClass):
    """A variable declaration with a name and a type."""
    var: Var = attr.ib()
    typ: Type = attr.ib()

    def __str__(self) -> str:
        if isinstance(self.typ, BoolType):
            return f"bool {self.var};"
        elif isinstance(self.typ, NatType):
            res = f"nat {self.var}"
            if self.typ.bounds is not None:
                res += " " + str(self.typ.bounds)
            return res + ";"
        elif isinstance(self.typ, RealType):
            return f"real {self.var};"
        raise ValueError(f"invalid type: {self.typ}")


@attr.s
class ConstDecl(DeclClass):
    """A constant declaration with a name and an expression."""
    var: Var = attr.ib()
    value: Expr = attr.ib()

    def __str__(self) -> str:
        return f"const {self.var} := {self.value};"


@attr.s
class ParameterDecl(DeclClass):
    """ A parameter declaration with a name and a type."""
    var: Var = attr.ib()
    typ: Type = attr.ib()

    def __str__(self) -> str:
        if isinstance(self.typ, BoolType):
            raise SyntaxError("A parameter cannot be of BoolType.")
        if isinstance(self.typ, NatType):
            res = f"nparam {self.var}"
            if self.typ.bounds is not None:
                res += " " + str(self.typ.bounds)
            return res + ";"
        elif isinstance(self.typ, RealType):
            return f"rparam {self.var};"
        raise ValueError(f"invalid type: {self.typ}")


Decl = Union[VarDecl, ConstDecl, ParameterDecl]
"""Union type for all declaration objects. See :class:`DeclClass` for use with isinstance."""
