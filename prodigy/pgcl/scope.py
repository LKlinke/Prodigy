"""
-------------------------------
Block-scoped declarations
-------------------------------

A declaration inside a block (``.ppy`` only, :class:`~prodigy.pgcl.ast.LocalDeclInstr`) makes a
variable visible from the declaration to the end of the enclosing block. It starts with its initial
value (0 or ``false`` without one) every time the declaration is executed, e.g. in every iteration of
a loop body, and it is marginalised out of the generating function at the end of the block.

:func:`lower_block_scope` translates such declarations into the plain AST that the type checker and
the analysis already understand:

- the variable becomes an ordinary variable of the program (or of the function whose body contains
  the block);
- the declaration becomes the assignment ``x := value`` (``x := 0`` without a value);
- the end of the block gets the assignment ``x := 0``.

Setting ``x`` to 0 is the marginalisation: it maps a generating function ``G(x, …)`` to
``G(1, …)``, the marginal of the other variables, and outside its block ``x`` is 0 in every state,
so it adds no factor to the generating function. A variable declared in a loop body is therefore
0 whenever the guard is evaluated, and invariants of the loop need not mention it.

Rules (checked here, so that ``.ppy`` reports them with the position of the declaration):

- a variable declared in a block may only be used after its declaration and inside that block
  (including nested blocks);
- a declaration in a block may not hide another name: a variable, constant or parameter of the
  program (of the function, inside a function body) or a variable of an enclosing block;
- the same name may be declared in several blocks that do not contain each other, with the same
  type; the declarations share one variable of the lowered program;
- only ``nat`` and ``bool`` variables can be declared in a block.

.. doctest::

    >>> from prodigy.pgcl.ppy_parser import parse_ppy
    >>> program = lower_block_scope(parse_ppy("n: nat = 3; while (n > 0) { c: nat = n % 2; n = n - 1 - c; }"))
    >>> print(program)
    nat n;
    nat c;
    n := 3;
    while (n > 0) {
        c := n % 2;
        n := (n - 1) - c;
        c := 0;
    }
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import attr

from prodigy.pgcl.ast import (AsgnInstr, BoolLitExpr, BoolType, ChoiceInstr,
                              Expr, Function, FunctionDecl, IfInstr, Instr,
                              LocalDeclInstr, LoopInstr, NatLitExpr, NatType,
                              OptimizationQuery, Program, QueryInstr,
                              SampleExpr, Type, VarDecl, VarExpr, WhileInstr)
from prodigy.pgcl.positions import copy_span
from prodigy.pgcl.ast.walk import Walk, instr_exprs, walk_expr, walk_instrs
from prodigy.pgcl.util.ref import Mut


class ScopeError(Exception):
    """
    A violation of the scope rules. ``declaration`` is the block declaration the error refers to
    (the offending one, or the one of a variable that is used outside its block), if any.
    """

    def __init__(self, message: str, declaration: Optional[LocalDeclInstr] = None):
        super().__init__(message)
        self.message = message
        self.declaration = declaration


def has_block_declarations(instructions: Iterable[Instr]) -> bool:
    """Whether a declaration inside a block occurs in ``instructions`` (at any depth)."""
    return any(isinstance(ref.val, LocalDeclInstr) for ref in walk_instrs(Walk.DOWN, list(instructions)))


def _zero(typ: Type) -> Expr:
    if isinstance(typ, BoolType):
        return BoolLitExpr(False)
    return NatLitExpr(0)


def _used_names(instr: Instr) -> Set[str]:
    """The names an instruction reads or writes itself (not in nested blocks)."""
    names: Set[str] = set()
    if isinstance(instr, AsgnInstr):
        names.add(instr.lhs)
    for expr in instr_exprs(instr):
        names |= _expr_names(expr)
    return names


def _expr_names(expr: Expr) -> Set[str]:
    names: Set[str] = set()
    for sub in walk_expr(Walk.DOWN, Mut.alloc(expr)):
        if isinstance(sub.val, VarExpr):
            names.add(sub.val.var)
        elif isinstance(sub.val, SampleExpr):
            names.add(sub.val.dist)
    return names


class _Lowering:
    """Lowers the blocks of one program body or function body."""

    def __init__(self, outer: Set[str], instructions: List[Instr]):
        self.outer = outer
        self.types: Dict[str, Type] = {}
        self.first: Dict[str, LocalDeclInstr] = {}
        for ref in walk_instrs(Walk.DOWN, instructions):
            if isinstance(ref.val, LocalDeclInstr):
                self.first.setdefault(ref.val.decl.var, ref.val)

    def check_uses(self, names: Set[str], visible: Set[str]) -> None:
        for name in sorted(names):
            if name in self.first and name not in visible and name not in self.outer:
                raise ScopeError(
                    f"'{name}' is used outside the block that declares it (or before its "
                    f"declaration).", self.first[name])

    def body(self, instructions: List[Instr], visible: Set[str]) -> List[Instr]:
        """The top level of a program or function: no declarations to forget at its end."""
        return self._instrs(instructions, set(visible), [])

    def block(self, instructions: List[Instr], visible: Set[str]) -> List[Instr]:
        declared: List[VarDecl] = []
        result = self._instrs(instructions, set(visible), declared)
        for decl in reversed(declared):
            forget = AsgnInstr(decl.var, _zero(decl.typ))
            if not result or result[-1] != forget:
                result.append(forget)
        return result

    def _instrs(self, instructions: List[Instr], visible: Set[str],
                declared: List[VarDecl]) -> List[Instr]:
        result: List[Instr] = []
        for instr in instructions:
            if isinstance(instr, LocalDeclInstr):
                result.append(self._declare(instr, visible))
                declared.append(instr.decl)
                visible.add(instr.decl.var)
                continue
            names = _used_names(instr)
            if isinstance(instr, OptimizationQuery):
                names.discard(instr.parameter)
            self.check_uses(names, visible)
            result.append(self._compound(instr, visible))
        return result

    def _declare(self, instr: LocalDeclInstr, visible: Set[str]) -> Instr:
        name, typ = instr.decl.var, instr.decl.typ
        if name in self.outer or name in visible:
            raise ScopeError(
                f"'{name}' is already declared; a declaration inside a block cannot hide "
                f"another name. Choose a different name.", instr)
        if not isinstance(typ, (NatType, BoolType)):
            raise ScopeError(
                f"Only nat and bool variables can be declared inside a block (not implemented "
                f"for '{name}': {typ}).", instr)
        known = self.types.setdefault(name, typ)
        if known != typ:
            raise ScopeError(
                f"'{name}' is declared in several blocks with different types; use different "
                f"names.", instr)
        if instr.value is not None:
            self.check_uses(_expr_names(instr.value), visible)
        return copy_span(instr, AsgnInstr(name, _zero(typ) if instr.value is None else instr.value))

    def _compound(self, instr: Instr, visible: Set[str]) -> Instr:
        return copy_span(instr, self._lower_compound(instr, visible))

    def _lower_compound(self, instr: Instr, visible: Set[str]) -> Instr:
        if isinstance(instr, WhileInstr):
            return attr.evolve(instr, body=self.block(instr.body, visible))
        if isinstance(instr, LoopInstr):
            return attr.evolve(instr, body=self.block(instr.body, visible))
        if isinstance(instr, IfInstr):
            return attr.evolve(instr, true=self.block(instr.true or [], visible),
                               false=self.block(instr.false or [], visible))
        if isinstance(instr, ChoiceInstr):
            return attr.evolve(instr, lhs=self.block(instr.lhs, visible),
                               rhs=self.block(instr.rhs, visible))
        if isinstance(instr, QueryInstr):
            return attr.evolve(instr, instrs=self.block(instr.instrs, visible))
        return instr


def _lower_function(function: Function, program: Program) -> Function:
    if not has_block_declarations(function.instructions):
        return function
    outer = set(function.variables) | set(program.constants) | set(program.parameters)
    lowering = _Lowering(outer, function.instructions)
    instructions = lowering.body(function.instructions, set())
    lowering.check_uses(_expr_names(function.returns), set())
    declarations = list(function.declarations)
    declarations.extend(VarDecl(name, typ) for name, typ in lowering.types.items())
    return Function(declarations, set(function.variables) | set(lowering.types), instructions,
                    function.returns)


def lower_block_scope(program: Program) -> Program:
    """
    Translate the declarations inside blocks of ``program`` and of its functions into plain
    assignments (see the module documentation). Returns ``program`` itself if it has none, else a
    new program; ``program`` is not changed.

    :raises ScopeError: if the program violates the scope rules.
    """
    functions = {name: _lower_function(function, program)
                 for name, function in program.functions.items()}
    changed_functions = any(functions[name] is not function
                            for name, function in program.functions.items())
    if not changed_functions and not has_block_declarations(program.instructions):
        return program

    outer = set(program.variables) | set(program.constants) | set(program.parameters)
    lowering = _Lowering(outer, program.instructions)
    instructions = lowering.body(program.instructions, set())

    declarations = []
    for decl in program.declarations:
        if isinstance(decl, FunctionDecl):
            decl = FunctionDecl(decl.var, functions[decl.var])
        declarations.append(decl)
    variables = dict(program.variables)
    for name, typ in lowering.types.items():
        declarations.append(VarDecl(name, typ))
        variables[name] = typ
    return Program(declarations, variables, dict(program.constants), dict(program.parameters),
                   functions, instructions,
                   block_variables=set(program.block_variables) | set(lowering.types))
