from abc import ABC, abstractmethod
from typing import Union, List

import attr

from prodigy.distribution.distribution import Distribution
from probably.pgcl import Expr, VarExpr, OptimizationType


@attr.s
class Optimizer(ABC):
    """
    The `Optimizer` optimizes target functions and distributions.
    """
    @staticmethod
    @abstractmethod
    def optimize(
        condition: Union[str, Expr],
        dist: Distribution,
        *parameters: Union[str, VarExpr],
        method: OptimizationType = OptimizationType.MINIMIZE
    ) -> List[Union[str, Expr]]:
        """
        This method optimizes the given distribution parameters wrt to the optimization method and condition.
        :param parameters: The parameter(s) which is to be optimized.
        :param dist: The distribution for which
        :param condition: The condition for which the parameters shall be optimized.
        :param config: The Analysis Configuration
        :param method: The optimization method.
        :return: The parameter values, that optimize the function.
        """
        pass
