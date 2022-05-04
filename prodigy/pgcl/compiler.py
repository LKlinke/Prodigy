"""
-----------------
Compiler Frontend
-----------------

This module provides an API that combines parsing and type-checking all at once.
It is just a thin wrapper around the parser, type-checker, and constant substitution.

Refer to :mod:`prodigy.pgcl.parser` and :mod:`prodigy.pgcl.check` for the parser and type-checker respectively.
Variable/constant substituion is implemented in :mod:`prodigy.pgcl.substitute`.
If you want them, linearity checks live in :mod:`prodigy.pgcl.syntax`.
"""

from typing import Union

from prodigy.pgcl import substitute
from prodigy.util.ref import Mut

from .ast import Expr, Program
from prodigy.pgcl.typechecker.check import (CheckFail, check_expectation, check_expression,
                                            check_program)
from prodigy.pgcl.parser import parse_expectation, parse_expr, parse_pgcl


def compile_pgcl(code: str,
                 *,
                 substitute_constants=True) -> Union[Program, CheckFail]:
    """
    Parse and type-check a pGCL program.

    .. doctest::

        >>> compile_pgcl("nat x; nat y; x := y")
        Program(variables={'x': NatType(bounds=None), 'y': NatType(bounds=None)}, constants={}, instructions=[AsgnInstr(lhs='x', rhs=VarExpr('y'))])

        >>> compile_pgcl("x := y")
        CheckFail(location=..., message='x is not a variable.')

    Args:
        code:
        substitute_constants: Whether constant substitution is done on the program, defaults to `True`.
    """
    program = parse_pgcl(code)
    check_result = check_program(program)
    if check_result is not None:
        return check_result
    if substitute_constants:
        substitute.substitute_constants(program)
    return program


def compile_expr(program: Program,
                 code: str,
                 *,
                 substitute_constants=True) -> Union[Expr, CheckFail]:
    """
    Parse and type-check an expression.

    Args:
        program:
        code:
        substitute_constants: Whether constant substitution is done on the expression, defaults to `True`.
    """
    expr = parse_expr(code)
    check_result = check_expression(program, expr)
    if check_result is not None:
        return check_result
    if substitute_constants:
        expr_ref = Mut.alloc(expr)
        substitute.substitute_constants_expr(program, expr_ref)
        return expr_ref.val
    return expr


def compile_expectation(program: Program,
                        code: str,
                        *,
                        substitute_constants=True) -> Union[Expr, CheckFail]:
    """
    Parse and type-check an expectation.

    Args:
        program:
        code:
        substitute_constants: Whether constant substitution is done on the expectation, defaults to `True`.
    """
    expr = parse_expectation(code)
    check_result = check_expectation(program, expr)
    if check_result is not None:
        return check_result
    if substitute_constants:
        expr_ref = Mut.alloc(expr)
        substitute.substitute_constants_expr(program, expr_ref)
        return expr_ref.val
    return expr
