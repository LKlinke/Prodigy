import itertools
from abc import ABC, abstractmethod
from typing import Iterator, Collection

import sympy

from prodigy.distribution import Distribution, CommonDistributionsFactory
from prodigy.util import make_poly_clause


class TemplateHeuristic(ABC):
    @abstractmethod
    def generate(self) -> Iterator[Distribution]:
        ...

    def __str__(self):
        return self.__class__.__name__


class RationalFunctionMaxDeg(TemplateHeuristic):
    """
        This heuristic enumerates rational functions with increasing total degree.
        It does so my increasing the denominator degree first and then tries all numerator polynomials
        upto the current denominator degree before increasing the numerator degree.
    """

    def __init__(self, max_deg: int, variables: Collection[str], dist_factory: CommonDistributionsFactory):
        self.max_deg = max_deg
        self.variables = variables
        self.dist_fact = dist_factory

    def generate(self) -> Iterator[Distribution]:
        for max_powers_denom in sympy.utilities.iterables.iproduct(*[range(self.max_deg + 1) for _ in self.variables]):
            degrees = [range(max_powers_denom[i] + 1) for i, _ in enumerate(self.variables)]
            numerator = " + ".join((make_poly_clause(f"n_{c}", self.variables, pows)
                                    for c, pows in enumerate(itertools.product(*degrees))))
            for max_powers_num in sympy.utilities.iterables.iproduct(
                    *[range(sum(max_powers_denom) + 1) for _ in self.variables]):
                if sum(max_powers_num) > sum(max_powers_denom):
                    continue
                denom_degrees = [range(max_powers_num[i] + 1) for i, _ in
                                 enumerate(self.variables)]
                denominator = " + ".join((make_poly_clause(f"d_{c}", self.variables, pows) for c, pows in
                                          enumerate(itertools.product(*denom_degrees))))
                yield self.dist_fact.from_expr(f"({numerator}) / ({denominator})", *self.variables)


class TemplateHeuristicsFactory:

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"Static class {str(cls)} cannot be instantiated.")

    @classmethod
    def rational_functions_total_deg(cls,
                                     max_deg: int,
                                     variables: Collection[str],
                                     dist_fact: CommonDistributionsFactory
                                     ) -> RationalFunctionMaxDeg:
        return RationalFunctionMaxDeg(max_deg, variables, dist_fact)
