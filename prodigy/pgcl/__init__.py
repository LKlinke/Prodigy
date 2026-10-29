# pylint: disable=line-too-long  # style as taken over from probably
"""
================
``prodigy.pgcl``
================

This module implements pGCL parsing, type-checking, and transformations of pGCL programs.
The most important modules are the two below:

1. :mod:`prodigy.pgcl.compiler`. Functions that do parsing and type-checking all at once.
2. :mod:`prodigy.pgcl.ast`. The abstract syntax tree (AST) with all its types that define it.

.. doctest::

    >>> from prodigy.pgcl.compiler import compile_pgcl
    >>> code = '''# this is a comment
    ... # A program starts with the variable declarations.
    ... # Every variable must be declared.
    ... # A variable is either of type nat or of type bool.
    ... bool f;
    ... nat c;  # optional: provide bounds for the variables. The declaration then looks as follows: nat c [0,100]
    ...
    ... # Logical operators: & (and),  || (or), not (negation)
    ... # Operator precedences: not > & > ||
    ... # Comparing arithmetic expressions: <=, <, ==
    ...
    ... skip;
    ...
    ... while (c < 10 & f) {
    ...     {c := unif(1,10)} [0.8] {f:=true}
    ... }'''
    >>> compile_pgcl(code)
    Program(variables={'f': BoolType(), 'c': NatType(bounds=None)}, constants={}, parameters={}, functions={}, instructions=[SkipInstr(), WhileInstr(cond=BinopExpr(operator=Binop.AND, lhs=BinopExpr(operator=Binop.LT, lhs=VarExpr('c'), rhs=NatLitExpr(10)), rhs=VarExpr('f')), body=[ChoiceInstr(prob=RealLitExpr("0.8"), lhs=[AsgnInstr(lhs='c', rhs=FunctionCallExpr(function='unif', params=([NatLitExpr(1), NatLitExpr(10)], {})))], rhs=[AsgnInstr(lhs='f', rhs=BoolLitExpr(True))])])])

For more details on what syntax is accepted for pGCL programs, you can view the :ref:`formal grammar used for the pGCL parser <pgcl_grammar>`.

.. automodule:: prodigy.pgcl.compiler
.. automodule:: prodigy.pgcl.ast
    :no-members:
.. automodule:: prodigy.pgcl.parser
.. automodule:: prodigy.pgcl.walk
.. automodule:: prodigy.pgcl.check
.. automodule:: prodigy.pgcl.substitute
"""

from .ast import *
from .ast.walk import *
from .check import *
from .parser import parse_pgcl
