from __future__ import annotations

import copy
from typing import Any, Dict, List

import attr

from .ast import Var
from .declarations import (ConstDecl, Decl, Function, FunctionDecl,
                           ParameterDecl, VarDecl)
from .expressions import Expr
from .instructions import Instr
from .types import NatType, Type


@attr.s
class Program:
    """
    A pGCL program has a bunch of variables with types, constants with defining expressions, and a list of instructions.
    """

    declarations: List[Decl] = attr.ib(repr=False)
    """The original list of declarations."""

    variables: Dict[Var, Type] = attr.ib()
    """
    A dict of variables to their type.
    Only valid if the declarations are well-typed.
    """

    constants: Dict[Var, Expr] = attr.ib()
    """
    A dict of constant names to their defining expression.
    Only valid if the declarations are well-typed.
    """

    parameters: Dict[Var, Type] = attr.ib()
    """
        A dict of parameters to their type.
        Only valid if the declarations are well-typed.
    """

    functions: Dict[Var, Function] = attr.ib()

    instructions: List[Instr] = attr.ib()

    @staticmethod
    def from_parse(declarations: List[Decl],
                   instructions: List[Instr]) -> Program:
        """Create a program from the parser's output."""
        variables: Dict[Var, Type] = {}
        constants: Dict[Var, Expr] = {}
        parameters: Dict[Var, Type] = {}
        functions: Dict[Var, Function] = {}

        for decl in declarations:
            if isinstance(decl, VarDecl):
                variables[decl.var] = decl.typ
            elif isinstance(decl, ConstDecl):
                constants[decl.var] = decl.value
            elif isinstance(decl, ParameterDecl):
                parameters[decl.var] = decl.typ
            elif isinstance(decl, FunctionDecl):
                functions[decl.var] = decl.body

        return Program(declarations, variables, constants, parameters,
                       functions, instructions)

    @staticmethod
    def from_function(function: Function) -> Program:
        """
        Create a shallow copy of the function as a program. All variables are
        assigned the type NatType.
        """
        return Program(
            function.declarations.copy(),  # type: ignore
            {var: NatType(bounds=None)
             for var in function.variables},
            {},
            {},
            {},
            function.instructions.copy())

    def add_variable(self, var: Var, typ: Type):
        """
        Add a new variable declaration to the program's list of declarations and
        to the dict of variables.

        :raises AssertionError: if the variable is already declared
        """
        for decl in self.declarations:
            assert decl.var != var, f"name {var} is already declared in program"
        assert var not in self.variables, f"variable {var} is already declared in program"
        self.declarations.append(VarDecl(var, typ))
        self.variables[var] = typ

    def to_skeleton(self) -> Program:
        """
        Return a (shallow) copy of this program with just the declarations, but
        without any instructions.

        .. doctest::

            >>> from probably.pgcl.parser import parse_pgcl
            >>> program = parse_pgcl("nat x; nat y; while (x < 2) {}")
            >>> program.to_skeleton()
            Program(variables={'x': NatType(bounds=None), 'y': NatType(bounds=None)}, constants={}, parameters={}, functions={}, instructions=[])
        """
        return Program(declarations=copy.copy(self.declarations),
                       parameters=copy.copy(self.parameters),
                       variables=copy.copy(self.variables),
                       constants=copy.copy(self.constants),
                       functions=copy.copy(self.functions),
                       instructions=[])

    def __str__(self) -> str:
        """
        Convert this program to corresponding source code in pGCL.

        .. doctest::

            >>> from probably.pgcl.parser import parse_pgcl
            >>> program = parse_pgcl("nat x; nat y; while (x < 2) {}")
            >>> print(program)
            nat x;
            nat y;
            while (x < 2) { }
        """
        instrs: List[Any] = list(self.declarations)
        instrs.extend(self.instructions)
        return "\n".join(map(str, instrs))
