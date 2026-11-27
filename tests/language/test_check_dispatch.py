"""The type checker has one handler per node class (#513, review L2): a new construct of the AST fails here until
its rule exists in `prodigy/language/check/expressions.py` or `instructions.py`."""
from typing import Set, get_args

import pytest

from prodigy.language.ast import Expr, ExprClass, Instr, InstrClass, Program, SubstExpr, VarExpr
from prodigy.language.check import get_type
from prodigy.language.check.context import check_node, type_of

NOT_IN_PROGRAMS = {SubstExpr}
"""Expressions that program transformations build, never the parser; they have no type rule."""


def _subclasses(base: type) -> Set[type]:
    """The node classes of the AST below ``base`` (not the private helpers of program transformations, such as
    the bound expressions of :mod:`prodigy.language.substitute`)."""
    found = set()
    for cls in base.__subclasses__():
        found |= _subclasses(cls)
        if cls.__module__.startswith("prodigy.language.ast."):
            found.add(cls)
    return found


def test_the_unions_list_every_node_class() -> None:
    assert set(get_args(Expr)) == _subclasses(ExprClass)
    assert set(get_args(Instr)) == _subclasses(InstrClass)


@pytest.mark.parametrize("cls", sorted(_subclasses(ExprClass) - NOT_IN_PROGRAMS, key=lambda cls: cls.__name__))
def test_every_expression_has_a_type_rule(cls: type) -> None:
    assert cls in type_of.registry, f"no type rule for {cls.__name__}"


@pytest.mark.parametrize("cls", sorted(_subclasses(InstrClass), key=lambda cls: cls.__name__))
def test_every_instruction_has_a_check(cls: type) -> None:
    assert cls in check_node.registry, f"no check for {cls.__name__}"


def test_a_node_without_a_rule_is_a_bug_of_the_checker() -> None:
    program = Program(list(), dict(), dict(), dict(), dict(), list())
    with pytest.raises(Exception, match="no rule for SubstExpr"):
        get_type(program, SubstExpr({}, VarExpr("x")))
