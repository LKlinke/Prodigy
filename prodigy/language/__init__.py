# pylint: disable=line-too-long  # style as taken over from probably
"""
====================
``prodigy.language``
====================

The language of Prodigy: the ``.ppy`` parser and printer, the abstract syntax tree, the type checker,
block scopes, source positions and the transformations of programs (constant substitution, draws in
expressions). The package was called ``prodigy.pgcl`` until v0.7.0 (#446); the old name still imports, with a
``DeprecationWarning``, until v0.8.0.

The most important modules:

1. :mod:`prodigy.language.compiler`: parsing and type checking at once (:func:`compile_program`).
2. :mod:`prodigy.language.ast`: the abstract syntax tree (AST) and its walks (:mod:`prodigy.language.ast.walk`).

.. doctest::

    >>> from prodigy.language.compiler import compile_program
    >>> code = '''# this is a comment
    ... # Every variable must be declared, with its type (nat, bool, real, dist).
    ... f: bool;
    ... c: nat;  # optional bounds: c: nat[0, 100];
    ...
    ... # Logical operators: and, or, not; comparisons: <=, <, ==
    ... skip;
    ...
    ... while (c < 10 and f) {
    ...     { c = unif(1, 10); } [0.8] { f = true; }
    ... }'''
    >>> compile_program(code)
    Program(variables={'f': BoolType(), 'c': NatType(bounds=None)}, constants={}, parameters={}, functions={}, instructions=[SkipInstr(), WhileInstr(cond=BinopExpr(operator=Binop.AND, lhs=BinopExpr(operator=Binop.LT, lhs=VarExpr('c'), rhs=NatLitExpr(10)), rhs=VarExpr('f')), body=[ChoiceInstr(prob=RealLitExpr("0.8"), lhs=[AsgnInstr(lhs='c', rhs=FunctionCallExpr(function='unif', params=([NatLitExpr(1), NatLitExpr(10)], {})))], rhs=[AsgnInstr(lhs='f', rhs=BoolLitExpr(True))])])])

The syntax is described in :ref:`ppy_reference`.
"""

# the AST and its walks are part of this package's API; ast.__all__ lists them, so the star import is exact
from .ast import *  # pylint: disable=wildcard-import,unused-wildcard-import
from . import ast
from .check import (CheckFail, sample_predefined_functions, get_type, mentions_dist, is_compatible, check_instr,
                    check_program, check_function, type_name, non_bool_guard)
from .parser import parse_program
from .ast.ast import install_printer as _install_printer
from .ppy_printer import node_to_ppy as _node_to_ppy

# str() of every node is its .ppy form; the AST imports no printer (tests/test_layering.py)
_install_printer(_node_to_ppy)

# the API of the package (#379, step 1): the AST and its walks, the type checker, the parser
__all__ = ["CheckFail", "sample_predefined_functions", "get_type", "mentions_dist", "is_compatible", "check_instr",
           "check_program", "check_function", "type_name", "non_bool_guard", "parse_program"]
__all__ += ast.__all__
