# pylint: disable=line-too-long  # style as taken over from probably
"""
-----------------
Compiler Frontend
-----------------

This module provides an API that combines parsing and type-checking all at once.
It is just a thin wrapper around the parser, type-checker, and constant substitution.

Refer to :mod:`prodigy.pgcl.parser` and :mod:`prodigy.pgcl.check` for the parser and type-checker respectively.
Variable/constant substituion is implemented in :mod:`prodigy.pgcl.substitute`.
"""

import os
from typing import List, Mapping, Optional, Union, cast

from prodigy.pgcl import substitute
from prodigy.pgcl.util.ref import Mut

from .ast import Expr, Instr, Node, Program, Var, VarDecl
from .check import (CheckFail, check_expectation, check_expression,
                    check_program, sample_predefined_functions)
from .parser import parse_pgcl
from .ppy_parser import parse_ppy_expr


def _ppy_spelling(node: Node) -> str:
    """``node`` in ``.ppy`` syntax (the pGCL spelling if it has no ``.ppy`` form), for messages."""
    # pylint: disable=import-outside-toplevel
    from .ppy_printer import PpyPrintError, expr_to_ppy, instr_to_ppy
    try:
        if type(node).__name__.endswith("Expr"):
            return expr_to_ppy(cast(Expr, node))
        return instr_to_ppy(cast(Instr, node))
    except PpyPrintError:
        return str(node)


def compile_pgcl(
    code: str,
    *,
    syntax: Optional[str] = None,
    path: Union[str, os.PathLike, None] = None,
    substitute_constants: bool = True,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Union[Program, CheckFail]:
    """
    Parse and type-check a pGCL program.

    .. doctest::

        >>> compile_pgcl("x: nat; y: nat; x = y;")
        Program(variables={'x': NatType(bounds=None), 'y': NatType(bounds=None)}, constants={}, parameters={}, functions={}, instructions=[AsgnInstr(lhs='x', rhs=VarExpr('y'))])

        >>> compile_pgcl("x = y;")
        CheckFail(location=..., message='x is not a variable.')

    Args:
        code:
        syntax: ``"ppy"`` (``"pgcl"`` raises ``ValueError``, see :func:`prodigy.pgcl.parser.parse_pgcl`).
        path: The file the code comes from, only checked (a ``.pgcl`` file raises ``ValueError``).
        substitute_constants: Whether constant substitution is done on the program, defaults to `True`.
    """
    program = parse_pgcl(code, syntax, path=path)
    if substitute_constants:
        substitute.substitute_constants(program)
    check_result = check_program(program, predefined_functions, spell=_ppy_spelling)
    if check_result is not None:
        return check_result
    return program


def compile_expr(
    program: Program,
    code: str,
    *,
    substitute_constants: bool = True,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Union[Expr, CheckFail]:
    """
    Parse (``.ppy`` syntax) and type-check an expression.

    Args:
        program:
        code:
        substitute_constants: Whether constant substitution is done on the expression, defaults to `True`.
    """
    expr = parse_ppy_expr(code)
    check_result = check_expression(program, expr, predefined_functions)
    if check_result is not None:
        return check_result
    if substitute_constants:
        expr_ref = Mut.alloc(expr)
        substitute.substitute_constants_expr(program, expr_ref)
        return expr_ref.val
    return expr


def compile_expectation(
    program: Program,
    code: str,
    *,
    substitute_constants: bool = True,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Union[Expr, CheckFail]:
    """
    Parse (``.ppy`` syntax) and type-check an expectation.

    Args:
        program:
        code:
        substitute_constants: Whether constant substitution is done on the expectation, defaults to `True`.
    """
    expr = parse_ppy_expr(code)
    check_result = check_expectation(program, expr, predefined_functions)
    if check_result is not None:
        return check_result
    if substitute_constants:
        expr_ref = Mut.alloc(expr)
        substitute.substitute_constants_expr(program, expr_ref)
        return expr_ref.val
    return expr
