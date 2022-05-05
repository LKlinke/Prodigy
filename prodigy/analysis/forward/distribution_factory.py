from abc import ABC, abstractmethod
from typing import Union

from prodigy.analysis.forward.distribution import Param, Distribution
from prodigy.pgcl import VarExpr, Expr


class CommonDistributionsFactory(ABC):
    """ Abstract Factory Class implementing a Factory for common distributions."""

    @staticmethod
    @abstractmethod
    def geometric(var: Union[str, VarExpr], p: Param) -> Distribution:
        """ A geometric distribution with parameter `p`."""
        pass

    @staticmethod
    @abstractmethod
    def uniform(var: Union[str, VarExpr], a: Param, b: Param) -> Distribution:
        """ A uniform distribution with bounds [`a`,`b`]."""
        pass

    @staticmethod
    @abstractmethod
    def bernoulli(var: Union[str, VarExpr], p: Param) -> Distribution:
        """ A bernoulli distribution with parameter `p`."""
        pass

    @staticmethod
    @abstractmethod
    def poisson(var: Union[str, VarExpr], lam: Param) -> Distribution:
        """ A poisson distribution with parameter `lamb`da."""
        pass

    @staticmethod
    @abstractmethod
    def log(var: Union[str, VarExpr], p: Param) -> Distribution:
        """ A logarithmic distribution with parameter `p`."""
        pass

    @staticmethod
    @abstractmethod
    def binomial(var: Union[str, VarExpr], n: Param, p: Param) -> Distribution:
        """ A binomial distribution with parameters `n` and `p`."""
        pass

    @staticmethod
    @abstractmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where actually no information about the states is given."""
        pass

    @staticmethod
    @abstractmethod
    def one(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where all variables are initialized with 0."""
        pass

    @staticmethod
    @abstractmethod
    def from_expr(expression: Union[str, Expr], *variables, **kwargs) -> Distribution:
        """ A distribution represented by the expression."""
        pass