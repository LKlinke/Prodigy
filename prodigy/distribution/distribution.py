from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Generator, Iterator, Set, Tuple, Union

from probably.pgcl import Expr, IidSampleExpr, VarExpr  # type: ignore


class MarginalType(Enum):
    """ Specifies the type of marginalization. """
    INCLUDE = auto()
    EXCLUDE = auto()


DistributionParam = Union[str, Expr]


@dataclass
class State:
    valuations: Dict[str, int] = field(default_factory=dict)

    def __iter__(self):
        return self.valuations.__iter__()

    def __getitem__(self, item):
        return self.valuations.__getitem__(item)

    def __setitem__(self, key, value):
        return self.valuations.__setitem__(key, value)

    def __str__(self):
        return self.valuations.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            return self.valuations == other
        if isinstance(other, State):
            return self.valuations == other.valuations
        return False

    def items(self):
        return self.valuations.items()

    def to_monomial(self) -> str:
        if self.valuations:
            result = "*"
            addends = []
            for variable, value in self.valuations.items():
                addends.append(f"{variable}^{value}")
            return result.join(addends)
        return ""

    def copy(self):
        return State(valuations=self.valuations.copy())


class Distribution(ABC):
    """ Abstract class that models different representations of probability distributions. """
    @abstractmethod
    def __add__(self, other) -> Distribution:
        """ The addition of two distributions. """

    @abstractmethod
    def __sub__(self, other) -> Distribution:
        """ The subtraction of two distributions. """

    @abstractmethod
    def __mul__(self, other) -> Distribution:
        """ The multiplication of two distributions. """

    @abstractmethod
    def __truediv__(self, other) -> Distribution:
        """ The division of two distributions. """

    @abstractmethod
    def __eq__(self, other) -> bool:
        """ Checks whether two distributions are equal. """

    @abstractmethod
    def __le__(self, other) -> bool:
        """ Checks whether `self` is less or equal than `other`."""

    def __lt__(self, other) -> bool:
        return self <= other and not self == other

    @abstractmethod
    def __str__(self) -> str:
        """ The string representation of a distribution. """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, State]]:
        """ Returns an iterator that iterates over the (probability, state) pairs of the distribution."""

    @abstractmethod
    def copy(self, deep: bool = True) -> Distribution:
        """ Returns a full copy of itself."""

    @abstractmethod
    def get_probability_of(self, condition: Union[Expr, str]):
        """
        Returns the probability of a given `condition` or variable.
        :param condition: The condition.
        :return: The probability that the condition is satisfied.
        """

    @abstractmethod
    def get_probability_mass(self) -> Union[Expr, str]:
        """ Returns the probability mass of the distribution. """

    @abstractmethod
    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        """ Returns the expected value of the expression `expression` evaluated in the distribution. """

    @abstractmethod
    def normalize(self) -> Distribution:
        """ Normalizes the probability mass of the distribution."""

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Returns the free variables of the distribution. """

    @abstractmethod
    def get_parameters(self) -> Set[str]:
        """ Returns the parameters of the distribution. """

    @abstractmethod
    def filter(self, condition: Union[Expr, str]) -> Distribution:
        """ Filters the distribution such that only the parts which satisfy the `condition` are left."""

    @abstractmethod
    def is_zero_dist(self) -> bool:
        """ Returns whether the distribution encodes the 0 distribution."""

    @abstractmethod
    def is_finite(self) -> bool:
        """ Returns whether the distribution has finite support."""

    @abstractmethod
    def update(self, expression: Expr) -> Distribution:
        """ Updates the distribution by the result of the expression. """

    @abstractmethod
    def update_iid(self, sampling_exp: IidSampleExpr,
                   variable: Union[str, VarExpr]) -> 'Distribution':
        """ Updates the distribution by the the iid-sampling rules. """

    @abstractmethod
    def marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> Distribution:
        """ 
        Computes the marginal distribution for the given variables (MarginalType.Include),
        or for all but the given variables (MarginalType.Exclude).
        :param method: The method of marginalization.
        :param variables: A list of variables for which the marginal distribution should be computed.
        If this list is empty or contains symbols that are not known variables of this distribution,
        this function will raise an exception.
        :return: The marginal distribution.
        """

    @abstractmethod
    def set_variables(self, *variables: str) -> Distribution:
        """
        Sets the free variables in a distribution.
        :param variables: The variables.
        :return:  The distribution with free variables `variables`
        """

    @abstractmethod
    def set_parameters(self, *parameters: str) -> Distribution:
        """
        Sets the parameters in a distribution.
        :param parameters: The parameters.
        :return: The Distribution with parameters `parameters`
        """

    @abstractmethod
    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        """
        Computes the approximation until the given threshold is reached. (Might not terminate)
        :param threshold: The threshold either as a maximum number of states, or a certain probability mass.
        :return: The approximated (truncated) probability distribution.
        """


class CommonDistributionsFactory(ABC):
    """ Abstract Factory Class implementing a Factory for common distributions."""
    @staticmethod
    @abstractmethod
    def geometric(var: Union[str, VarExpr],
                  p: DistributionParam) -> Distribution:
        """ A geometric distribution with parameter `p`."""

    @staticmethod
    @abstractmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> Distribution:
        """ A uniform distribution with bounds [`lower`,`upper`]."""

    @staticmethod
    @abstractmethod
    def bernoulli(var: Union[str, VarExpr],
                  p: DistributionParam) -> Distribution:
        """ A bernoulli distribution with parameter `p`."""

    @staticmethod
    @abstractmethod
    def poisson(var: Union[str, VarExpr],
                lam: DistributionParam) -> Distribution:
        """ A poisson distribution with parameter `lamb`da."""

    @staticmethod
    @abstractmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        """ A logarithmic distribution with parameter `p`."""

    @staticmethod
    @abstractmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam,
                 p: DistributionParam) -> Distribution:
        """ A binomial distribution with parameters `n` and `p`."""

    @staticmethod
    @abstractmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where actually no information about the states is given."""

    @staticmethod
    @abstractmethod
    def one(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where all variables are initialized with 0."""

    @staticmethod
    @abstractmethod
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> Distribution:
        """ A distribution represented by the expression."""
