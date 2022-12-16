from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Set, Union

import attr

from .ast import Node, Var
from .expressions import Expr
from .instructions import Instr
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


@attr.s
class FunctionDecl(DeclClass):
    """
    A function declaration with a name and a function corresponding to that
    name.
    """

    var: Var = attr.ib()
    """The function's name."""

    body: Function = attr.ib()

    def __str__(self) -> str:
        return f"fun {self.var} := {self.body};"


@attr.s
class Function(Node):
    """
    A function is similar to a :class:`~probably.pgcl.ast.Program`, but more
    limited and with a `return` statement at the end.
    """

    declarations: List[VarDecl] = attr.ib()
    variables: Set[Var] = attr.ib()
    instructions: List[Instr] = attr.ib()
    returns: Expr = attr.ib()

    @staticmethod
    def from_parse(declarations: List[VarDecl], instructions: List[Instr],
                   returns: Expr) -> Function:
        variables = set()

        for decl in declarations:
            assert isinstance(decl, VarDecl)
            variables.add(decl.var)

        return Function(declarations, variables, instructions, returns)

    def __str__(self):
        instrs: List[Any] = list(self.declarations)
        instrs.extend(self.instructions)
        res = "\n\t".join(map(str, instrs))
        return f"{{\n\t{res}\n\treturn {str(self.returns)};\n}}"


Decl = Union[VarDecl, ConstDecl, ParameterDecl, FunctionDecl]
"""Union type for all declaration objects. See :class:`DeclClass` for use with isinstance."""
