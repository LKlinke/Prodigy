import itertools
from typing import Collection, Iterator

import sympy

from prodigy.analysis.evtinvariants.heuristics.templates.templates import \
    TemplateHeuristic
from prodigy.distribution.distribution import (CommonDistributionsFactory,
                                               Distribution)
from prodigy.util import make_poly_clause


class PolynomialEnumeration(TemplateHeuristic):
    """
        This heuristic enumerates polynomials up to a maximum total degree to find a suitable invaraint.
        If this maximum degree is set to -1, it enumerates forever.
        This might be working best for finite state models.
    """
    def __init__(self,
                 variables: Collection[str],
                 dist_factory: CommonDistributionsFactory,
                 max_deg: int = -1):
        self.max_deg = max_deg
        self.variables = variables
        self.dist_fact = dist_factory

    def generate(self) -> Iterator[Distribution]:

        iterators = sympy.S.Naturals0 if self.max_deg == -1 else range(
            self.max_deg + 1)

        for max_powers in sympy.utilities.iterables.iproduct(
                *[iterators for _ in self.variables]):
            degrees = [
                range(max_powers[i] + 1) for i, _ in enumerate(self.variables)
            ]
            numerator = " + ".join(
                (make_poly_clause(f"n_{c}", self.variables, pows)
                 for c, pows in enumerate(itertools.product(*degrees))))
            yield self.dist_fact.from_expr(numerator, *self.variables)
