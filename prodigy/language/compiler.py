# pylint: disable=line-too-long  # style as taken over from probably
"""
-----------------
Compiler Frontend
-----------------

This module provides an API that combines parsing and type-checking all at once.
It is just a thin wrapper around the parser, type-checker, and constant substitution.

Refer to :mod:`prodigy.language.parser` and :mod:`prodigy.language.check` for the parser and type-checker respectively.
Variable/constant substituion is implemented in :mod:`prodigy.language.substitute`.
"""

import os
from typing import List, Mapping, Optional, Union

from prodigy.language import substitute

from .ast import Program, Var, VarDecl
from .check import CheckFail, check_program, sample_predefined_functions
from .parser import parse_program



def compile_program(
    code: str,
    *,
    syntax: Optional[str] = None,
    path: Union[str, os.PathLike, None] = None,
    substitute_constants: bool = True,
    predefined_functions: Mapping[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Union[Program, CheckFail]:
    """
    Parse and type-check a program (``.ppy`` syntax); called ``compile_pgcl`` until v0.7.0 (#446).

    .. doctest::

        >>> compile_program("x: nat; y: nat; x = y;")
        Program(variables={'x': NatType(bounds=None), 'y': NatType(bounds=None)}, constants={}, parameters={}, functions={}, instructions=[AsgnInstr(lhs='x', rhs=VarExpr('y'))])

        >>> compile_program("x = y;")
        CheckFail(location=..., message='x is not a variable.')

    Args:
        code:
        syntax: ``"ppy"`` (``"pgcl"`` raises ``ValueError``, see :func:`prodigy.language.parser.parse_program`).
        path: The file the code comes from, only checked (a ``.pgcl`` file raises ``ValueError``).
        substitute_constants: Whether constant substitution is done on the program, defaults to `True`.
    """
    program = parse_program(code, syntax, path=path)
    if substitute_constants:
        substitute.substitute_constants(program)
    check_result = check_program(program, predefined_functions)
    if check_result is not None:
        return check_result
    return program
