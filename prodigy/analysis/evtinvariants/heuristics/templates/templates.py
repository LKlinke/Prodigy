"""
Templates of invariants: generating functions whose coefficients are unknowns (parameters) to be solved for.

A template is built as an expression of the language (:class:`~prodigy.language.Expr`) and handed to the engine with
``factory.from_expr``, which writes it as documented engine text (:mod:`prodigy.language.engine_text`); no template
text is assembled and parsed again (review A11 of #221).
"""
from abc import ABC, abstractmethod
from typing import Collection, Iterator, Sequence

from prodigy.distribution import Distribution
from prodigy.language import Binop, BinopExpr, Expr, NatLitExpr, VarExpr

__all__ = ["TemplateHeuristic", "monomial", "balanced"]


class TemplateHeuristic(ABC):
    @abstractmethod
    def generate(self) -> Iterator[Distribution]:
        ...

    def __str__(self) -> str:
        return self.__class__.__name__


def balanced(operator: Binop, operands: Sequence[Expr]) -> Expr:
    """``operands`` joined by the associative ``operator`` as a balanced tree, so that the depth of the expression
    (and of the recursion that writes it) grows with the logarithm of the number of operands; at least one."""
    if len(operands) == 1:
        return operands[0]
    middle = len(operands) // 2
    return BinopExpr(operator, balanced(operator, operands[:middle]), balanced(operator, operands[middle:]))


def monomial(coefficient: str, variables: Collection[str], powers: Collection[int]) -> Expr:
    """The term ``coefficient * x^a * y^b * …`` with the unknown ``coefficient`` (factors with power 0 omitted).

    >>> from prodigy.language.engine_text import to_engine_text
    >>> to_engine_text(monomial("n_3", ["x", "y"], [1, 2]))
    'n_3 * (x * ((y) ^ (2)))'
    """
    factors: list[Expr] = [VarExpr(coefficient)]
    for variable, power in zip(variables, powers, strict=True):
        if power == 1:
            factors.append(VarExpr(variable))
        elif power > 1:
            factors.append(BinopExpr(Binop.POWER, VarExpr(variable), NatLitExpr(power)))
    return balanced(Binop.TIMES, factors)
