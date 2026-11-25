from __future__ import annotations

from typing import Dict, List, Set, Tuple, Union

import attr

from .ast import Node, Var
from .expressions import Expr, NatLitExpr
from .instructions import Instr
from .types import Type


class DeclClass(Node):
    """Superclass for all declarations. See :obj:`Decl`."""


@attr.s
class VarDecl(DeclClass):
    """A variable declaration with a name and a type."""
    var: Var = attr.ib()
    typ: Type = attr.ib()


@attr.s
class ConstDecl(DeclClass):
    """A constant declaration with a name and an expression."""
    var: Var = attr.ib()
    value: Expr = attr.ib()


@attr.s
class ParameterDecl(DeclClass):
    """ A parameter declaration with a name and a type."""
    var: Var = attr.ib()
    typ: Type = attr.ib()


@attr.s
class FunctionDecl(DeclClass):
    """
    A function declaration with a name and a function corresponding to that
    name.
    """

    var: Var = attr.ib()
    """The function's name."""

    body: Function = attr.ib()


@attr.s
class Function(Node):
    """
    A function is similar to a :class:`~prodigy.language.ast.Program`, but more
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

    def params_to_dict(
            self, params: Tuple[List[Expr], Dict[Var,
                                                 Expr]]) -> Dict[Var, Expr]:
        """
        Converts a tuple of positional and named parameters as given by a
        :class:`~prodigy.language.ast.FunctionCallExpr` to a dict that maps each
        variable name of this function to an initial value. Variables that are
        not contained in the given parameters are mapped to 0.

        In other words, converts the positional and named parameters to only
        named parameters.
        """
        res = params[1].copy()
        assert len(params[0]) <= len(
            self.declarations), "too many positional parameters"
        for index, expr in enumerate(params[0]):
            var = self.declarations[index].var
            assert var not in res, f"{var} is defined as a positional and as a named parameter"
            res[var] = expr
        res.update({var: NatLitExpr(0) for var in self.variables - res.keys()})
        return res


Decl = Union[VarDecl, ConstDecl, ParameterDecl, FunctionDecl]
"""Union type for all declaration objects. See :class:`DeclClass` for use with isinstance."""
