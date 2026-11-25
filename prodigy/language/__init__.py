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

from .ast import *
from .ast.walk import *
from .check import *
from .parser import parse_program
