r"""
----------------------
Lark Expression Parser
----------------------

Generate Lark grammars for expressions from operator tables.
The algorithm is pretty simple, but easy to mess up when doing it by hand.

.. doctest::

    >>> table = [
    ...    [infixl("plus", "+"), infixl("minus", "-")],
    ...    [prefix("neg", "-"), atom("parens", '"(" test ")"'), atom("int", "/[0-9]+/")]
    ... ]
    >>> print(build_expr_parser(table, "test"))
    ?test: test_0
    ?test_0: test_1
        | test_0 "+" test_1 -> plus
        | test_0 "-" test_1 -> minus
    ?test_1: "-" test_1 -> neg
        | "(" test ")" -> parens
        | /[0-9]+/ -> int
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Union

import attr


class Assoc(Enum):
    LEFT = auto()
    RIGHT = auto()


RuleName = str


@attr.s
class _OperatorClass(ABC):
    name: str = attr.ib()
    op: str = attr.ib()

    @abstractmethod
    def grammar(self, name: RuleName, next_name: RuleName) -> str:
        pass


@attr.s
class _InfixOperator(_OperatorClass):
    assoc: Assoc = attr.ib()

    def grammar(self, name: RuleName, next_name: RuleName) -> str:
        (lhs, rhs) = (name,
                      next_name) if self.assoc == Assoc.LEFT else (next_name,
                                                                   name)
        return f'{lhs} "{self.op}" {rhs} -> {self.name}'


@attr.s
class _PrefixOperator(_OperatorClass):
    def grammar(self, name: RuleName, next_name: RuleName) -> str:
        return f'"{self.op}" {name} -> {self.name}'


@attr.s
class _PostfixOperator(_OperatorClass):
    def grammar(self, name: RuleName, next_name: RuleName) -> str:
        return f'{name} "{self.op}" -> {self.name}'


@attr.s
class _AtomOperator(_OperatorClass):
    def grammar(self, name: RuleName, next_name: RuleName) -> str:
        return f'{self.op} -> {self.name}'


Operator = Union[_InfixOperator, _PrefixOperator, _PostfixOperator,
                 _AtomOperator]


def infixl(name: RuleName, op: str) -> Operator:
    return _InfixOperator(name, op, Assoc.LEFT)


def infixr(name: RuleName, op: str) -> Operator:
    return _InfixOperator(name, op, Assoc.RIGHT)


def prefix(name: RuleName, op: str) -> Operator:
    return _PrefixOperator(name, op)


def postfix(name: RuleName, op: str) -> Operator:
    return _PostfixOperator(name, op)


def atom(name: RuleName, rule_str: str) -> Operator:
    return _AtomOperator(name, rule_str)


OperatorTable = List[List[Operator]]
"""
An OperatorTable contains a list of precedence levels in ascending order.
"""


def build_expr_parser(table: OperatorTable, rule_name: RuleName) -> str:
    """Generate Lark grammar from the operator table."""

    # I've found the following example very instructive for the implementation:
    # https://github.com/lark-parser/lark/blob/84a876724958ac582c40737581d938cded37fe02/examples/calc.py

    def level_name(level: int) -> RuleName:
        return f'{rule_name}_{level}'

    grammar = [f'?{rule_name}: {level_name(0)}']

    for i, level in enumerate(table):
        name = level_name(i)
        next_name = level_name((i + 1) % len(table))

        # Either match an operator from this level...
        op_grammars = [operator.grammar(name, next_name) for operator in level]

        # ... or match one from the next (higher) level.
        if i < len(table) - 1:
            op_grammars.insert(0, next_name)

        rule = "\n    | ".join(op_grammars)
        grammar.append(f'?{name}: {rule}')

    return "\n".join(grammar)
