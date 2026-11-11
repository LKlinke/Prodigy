import itertools
from typing import Collection, Iterator

import sympy

from prodigy.analysis.evtinvariants.heuristics.templates.templates import TemplateHeuristic
from prodigy.distribution import CommonDistributionsFactory, Distribution
from prodigy.util import make_poly_clause


class RationalFunctionMaxDeg(TemplateHeuristic):
    """
        This heuristic enumerates rational functions N / D with increasing degree: for every bound on the
        degrees of the numerator N in each variable (at most ``max_deg``), all denominators D whose degree
        bounds sum up to at most that of N.

        The constant coefficient of D is fixed to 1. This loses no invariant: a rational function with a
        power series expansion has, in lowest terms, a denominator with a non-zero constant term, which can be
        scaled to 1. Without it, every solution N / D comes with the whole family (c * N) / (c * D), and
        solving the template's system for such a positive-dimensional set (a Gröbner basis) is far more
        expensive.
    """

    def __init__(self, max_deg: int, variables: Collection[str], dist_factory: CommonDistributionsFactory):
        self.max_deg = max_deg
        self.variables = variables
        self.dist_fact = dist_factory

    def generate(self) -> Iterator[Distribution]:
        for max_powers_num in sympy.utilities.iterables.iproduct(*[range(self.max_deg + 1) for _ in self.variables]):
            num_degrees = [range(max_powers_num[i] + 1) for i, _ in enumerate(self.variables)]
            numerator = " + ".join((make_poly_clause(f"n_{c}", self.variables, pows)
                                    for c, pows in enumerate(itertools.product(*num_degrees))))
            for max_powers_denom in sympy.utilities.iterables.iproduct(
                    *[range(sum(max_powers_num) + 1) for _ in self.variables]):
                if sum(max_powers_denom) > sum(max_powers_num):
                    continue
                denom_degrees = [range(max_powers_denom[i] + 1) for i, _ in enumerate(self.variables)]
                # the first monomial is the constant one (all powers 0), its coefficient is 1
                denominator = " + ".join(["1"] + [make_poly_clause(f"d_{c}", self.variables, pows) for c, pows in
                                                  enumerate(itertools.product(*denom_degrees)) if c > 0])
                yield self.dist_fact.from_expr(f"({numerator}) / ({denominator})", *self.variables)
