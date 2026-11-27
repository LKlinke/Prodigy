"""
The check of a program (:func:`check_program`) and of a function (:func:`check_function`): declarations,
constants and instructions; and the guards that are no ``bool`` (:func:`non_bool_guard`), which the parsers
reject with a position.
"""
# pylint: disable=line-too-long  # style as taken over from probably
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

from prodigy.language.ast import (Binop, BinopExpr, BoolType, Decl, Expr, Function, FunctionDecl, IfInstr, Instr,
                                  NatType, ObserveInstr, Program, Type, Var, VarDecl, VarExpr, Walk, WhileInstr,
                                  DistributionType, name_uses, walk_expr, walk_instrs)
from prodigy.language.util.ref import Mut

from .calls import sample_predefined_functions
from .compatibility import is_compatible
from .context import CheckFail, Ctx, check_block, type_of


def _check_constant_declarations(program: Program) -> Optional[CheckFail]:
    """
    Check that constants are defined before they are used.
    Also check that constants only contain other constants.

    .. doctest::

        >>> from prodigy.language.parser import parse_program
        >>> program = parse_program("const x = y; const y = x;")
        >>> _check_constant_declarations(program)
        CheckFail(location=VarExpr('y'), message='Constant is not yet defined')

        >>> program = parse_program("x: nat; const y = x;")
        >>> _check_constant_declarations(program)
        CheckFail(location=VarExpr('x'), message='Variable used in constant definition is not a constant')
    """
    declared = set()
    for name, value in program.constants.items():
        for use in name_uses(value):
            if isinstance(use, VarExpr):
                if use.var not in program.constants:
                    return CheckFail(use, "Variable used in constant definition is not a constant")
                if use.var not in declared:
                    return CheckFail(use, "Constant is not yet defined")
        declared.add(name)

    return None


def _check_declaration_list(
        program: Program,
        predefined_functions: Mapping[Var, List[VarDecl]]) -> Optional[CheckFail]:
    """
    Check that all variables/constants are defined at most once.

    .. doctest::

        >>> from prodigy.language.parser import parse_program
        >>> program = parse_program("const x: nat = 1; const x: nat = 1;")
        >>> _check_declaration_list(program, {})
        CheckFail(location=..., message='Already declared variable/constant before.')
    """
    declared: Dict[Var, Decl] = {}
    for decl in program.declarations:
        if declared.get(decl.var) is not None:
            return CheckFail(decl,
                             "Already declared variable/constant before.")
        declared[decl.var] = decl
        if isinstance(decl, VarDecl) and isinstance(decl.typ, NatType) \
                and decl.typ.bounds is not None:
            # bounds are not type-checked or evaluated; keep '//' out of them
            for bound in (decl.typ.bounds.lower, decl.typ.bounds.upper):
                for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(bound)):
                    if isinstance(sub_expr.val, BinopExpr) \
                            and sub_expr.val.operator == Binop.FLOOR_DIVIDE:
                        return CheckFail(
                            sub_expr.val,
                            "Floor division '//' is not implemented in bounds.")
        if isinstance(decl, FunctionDecl):
            res = check_function(decl.body, program, predefined_functions)
            if isinstance(res, CheckFail):
                return res
    return None


def check_program(
    program: Program,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """Check a program for type-safety. Messages write expressions in ``.ppy`` (#425, #489)."""
    return _check_program(program, predefined_functions)


def _check_program(program: Program,
                   predefined_functions: Mapping[Var, List[VarDecl]]) -> Optional[CheckFail]:
    check_result = _check_constant_declarations(program)
    if check_result is not None:
        return check_result
    check_result = _check_declaration_list(program, predefined_functions)
    if check_result is not None:
        return check_result
    return check_block(Ctx(program, predefined_functions), program.instructions)


def check_function(
    function: Function,
    context: Program,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """
    Check a function for type-safety. Its parameters and variables are ``nat`` or ``dist``, and it returns a
    ``nat``.
    """

    for decl in function.declarations:
        if decl.typ != NatType(bounds=None) and not isinstance(decl.typ, DistributionType):
            return CheckFail(
                location=decl,
                message=
                "Only variables of type nat or dist are allowed in functions")

    as_prog = Program.from_function(function, context, copy_functions=True)
    res = check_program(as_prog, predefined_functions)
    if isinstance(res, CheckFail):
        return res

    typ = type_of(function.returns, Ctx(as_prog, predefined_functions))
    if isinstance(typ, CheckFail):
        return typ
    if not isinstance(typ, NatType):
        return CheckFail(location=function.returns,
                         message="Functions may only return integers")

    return None


def _guards(instructions: List[Instr]) -> Iterable[tuple[Instr, Expr, str]]:
    for ref in walk_instrs(Walk.DOWN, instructions):
        instr = ref.val
        if isinstance(instr, IfInstr):
            yield instr, instr.cond, "the guard of if"
        elif isinstance(instr, WhileInstr):
            yield instr, instr.cond, "the guard of while"
        elif isinstance(instr, ObserveInstr):
            yield instr, instr.cond, "the condition of observe"


def non_bool_guard(program: Program) -> Optional[tuple[Instr, Expr, str, Type]]:
    """
    The first guard of ``if``/``while`` or condition of ``observe`` (in the program or in one of
    its functions) whose type is known and is not ``bool``, as (instruction, guard, what it is,
    its type); ``None`` if there is none. A guard needs a truth value; a number such as ``x`` or
    ``bernoulli(1/2)`` has none, so such a program is rejected (the parsers raise
    :class:`~prodigy.language.syntax_errors.ProgramTypeError`). Guards whose type cannot be determined
    (e.g. an undeclared name) are left to :func:`check_program`.

    .. doctest::

        >>> from prodigy.language.check import type_name
        >>> from prodigy.language.parser import parse_program
        >>> _, guard, what, typ = non_bool_guard(parse_program("x: nat; if (x) {     x = 1; } else {     skip; }", check_guards=False))
        >>> guard, what, type_name(typ)
        (VarExpr('x'), 'the guard of if', 'nat')
        >>> non_bool_guard(parse_program("x: nat; while (x > 0) {     x = x - 1; }")) is None
        True
    """
    views = [(program, program.instructions)]
    views.extend((Program.from_function(function, program), function.instructions)
                 for function in program.functions.values())
    for view, instructions in views:
        for instr, guard, what in _guards(instructions):
            typ = type_of(guard, Ctx(view, sample_predefined_functions))
            if not isinstance(typ, CheckFail) and not is_compatible(BoolType(), typ):
                return instr, guard, what, typ
    return None
