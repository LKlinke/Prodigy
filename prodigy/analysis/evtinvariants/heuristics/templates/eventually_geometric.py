import itertools
from typing import Collection, Iterator

import sympy

from prodigy.analysis.evtinvariants.heuristics.templates.templates import TemplateHeuristic
from prodigy.distribution import CommonDistributionsFactory, Distribution
from prodigy.util import make_poly_clause


class EventuallyGeometricDistributionEnumeration(TemplateHeuristic):
    """
        Eventually Geometric Distributions as defined in Definition 6.4.1 of
            > Fabian Zaiser. 2024b. Towards Formal Verification of Bayesian Inference in Probabilistic Programming
            > via Guaranteed Bounds. Ph. D. Dissertation. University of Oxford. 
        This heuristic produces PGFs in the form of Lemma C.1.2 by enumerating eventually geometric distribution
        functions with increasing total degree of the initial block.
        If the maximum degree is set to -1, it enumerates forever.
    """

    def __init__(self,
                 variables: Collection[str],
                 dist_factory: CommonDistributionsFactory,
                 max_deg: int = -1):
        self.max_deg = max_deg
        self.variables = variables
        self.dist_fact = dist_factory

    def make_egd_summand(self, coef: str, powers: Collection[int], max_powers: Collection[int]) -> str:
        """
            Generates an egd summand coef * vars^powers / ((1 - a_x * X) * (1 - a_z * Z) * ...) 
            where v=X, Z, ... are variables with powers[v] == max_powers[v]
            Importantly: the parameters a_v are the same in all egd summands
        """
        numerator = make_poly_clause(coef, self.variables, powers)
        denominator = " * ".join(f"(1 - a_{i} * {v})" for i, v in enumerate(self.variables) if powers[i] == max_powers[i])
        return f"({numerator}) / ({denominator})" if denominator else numerator

    def generate(self) -> Iterator[Distribution]:
        iterators = sympy.S.Naturals0 if self.max_deg == -1 else range(
            self.max_deg + 1)

        for max_powers in sympy.utilities.iterables.iproduct(*[iterators for _ in self.variables]):
            degrees = [
                range(max_powers[i] + 1) for i, _ in enumerate(self.variables)
            ]
            numerator = " + ".join((self.make_egd_summand(f"q_{c}", pows, max_powers)
                                    for c, pows in enumerate(itertools.product(*degrees))))
            yield self.dist_fact.from_expr(numerator, *self.variables)
