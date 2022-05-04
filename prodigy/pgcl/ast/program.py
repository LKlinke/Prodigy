import copy

import attr

from typing import List, Dict, Any
from . import Decl, Var, Type, Expr, Instr, VarDecl, ConstDecl


@attr.s(frozen=True)
class ProgramConfig:
    """
    Some compilation options for programs. Frozen after initialization (cannot
    be modified).

    At the moment, we only have a flag for the type checker on which types are
    allowed as program variables.
    """

    allow_real_vars: bool = attr.ib(default=True)
    """
    Whether real numbers are allowed as program values (in computations, or as
    variables).
    """


@attr.s
class Program:
    """
    A pGCL program has a bunch of variables with types, constants with defining expressions, and a list of instructions.
    """
    config: ProgramConfig = attr.ib(repr=False)

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

    instructions: List[Instr] = attr.ib()

    @staticmethod
    def from_parse(config: ProgramConfig, declarations: List[Decl], parameters: Dict[Var, Type],
                   instructions: List[Instr]) -> "Program":
        """Create a program from the parser's output."""
        variables: Dict[Var, Type] = dict()
        constants: Dict[Var, Expr] = dict()

        for decl in declarations:
            if isinstance(decl, VarDecl):
                variables[decl.var] = decl.typ
            elif isinstance(decl, ConstDecl):
                constants[decl.var] = decl.value

        return Program(config, declarations, variables, constants, parameters, instructions)

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

    def to_skeleton(self) -> 'Program':
        """
        Return a (shallow) copy of this program with just the declarations, but
        without any instructions.
        """
        return Program(config=self.config,
                       declarations=copy.copy(self.declarations),
                       parameters=copy.copy(self.parameters),
                       variables=copy.copy(self.variables),
                       constants=copy.copy(self.constants),
                       instructions=[])

    def __str__(self) -> str:
        """
        Convert this program to corresponding source code in pGCL.
        """
        instrs: List[Any] = list(self.declarations)
        instrs.extend(self.instructions)
        return "\n".join(map(str, instrs))
