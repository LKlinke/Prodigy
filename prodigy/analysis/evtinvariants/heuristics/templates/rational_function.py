import itertools
from typing import Collection, Iterator, List, Tuple

from prodigy.analysis.evtinvariants.heuristics.templates.templates import TemplateHeuristic
from prodigy.distribution import CommonDistributionsFactory, Distribution
from prodigy.util import make_poly_clause


def _ordered_factorisations(size: int, count: int) -> Iterator[Tuple[int, ...]]:
    """The degree bounds ``e`` of ``count`` variables with ``prod(e_i + 1) == size``."""
    if count == 1:
        yield (size - 1,)
        return
    for factor in range(1, size + 1):
        if size % factor == 0:
            for rest in _ordered_factorisations(size // factor, count - 1):
                yield (factor - 1, *rest)


def _max_monomials(count: int, total: int) -> int:
    """The largest ``prod(e_i + 1)`` of ``count`` degree bounds with ``sum(e) <= total``: the product of
    factors with a given sum is largest when they are balanced."""
    base, extra = divmod(total + count, count)
    return base ** (count - extra) * (base + 1) ** extra


MAX_NUMERATOR_COEFFICIENTS = 500
"""Templates whose numerator has more coefficients are not enumerated (with many variables the full
numerator of a level has ``(L + 1)^k`` of them; building such a template already exhausts SymPy)."""

DENOMINATOR_WEIGHT = 8
"""Cost of an unknown coefficient of D relative to one of N in the order of the templates."""


class RationalFunctionMaxDeg(TemplateHeuristic):
    """
        This heuristic enumerates rational functions N / D: for every level ``L = 0, ..., max_deg``, the
        numerator N has degree at most L in each variable and the denominator D runs through all degree bounds
        per variable that sum up to at most that of N (``L`` times the number of variables). The templates with
        fewer unknown coefficients come first, those of D weighted more: the system is linear in the
        coefficients of N and cheap to solve, those of D make it non-linear. Levels whose numerator has more
        than :data:`MAX_NUMERATOR_COEFFICIENTS` coefficients are left out (``effective_max_deg``).

        Only the full numerator of each level is enumerated: the synthesis solves the coefficients of N by
        linear algebra (:mod:`prodigy.analysis.evtinvariants.template_solver`), so a numerator with smaller
        degree bounds is the special case with some coefficients 0 and needs no template of its own. This
        covers the same rational functions as all pairs of degree bounds, with far fewer templates (95
        instead of 425 for two variables and ``max_deg`` 4).

        The constant coefficient of D is fixed to 1. This loses no invariant: a rational function with a
        power series expansion has, in lowest terms, a denominator with a non-zero constant term, which can be
        scaled to 1. Without it, every solution N / D comes with the whole family (c * N) / (c * D).
    """

    def __init__(self, max_deg: int, variables: Collection[str], dist_factory: CommonDistributionsFactory):
        self.max_deg = max_deg
        self.variables = variables
        self.dist_fact = dist_factory
        count = len(variables)
        self.levels = [level for level in range(max_deg + 1) if (level + 1) ** count <= MAX_NUMERATOR_COEFFICIENTS]
        self.effective_max_deg = self.levels[-1] if self.levels else -1
        """The largest level enumerated: below ``max_deg`` if a full numerator of a higher level has more than
        :data:`MAX_NUMERATOR_COEFFICIENTS` coefficients."""

    def templates(self) -> Iterator[Tuple[int, Tuple[int, ...]]]:
        """The pairs (level, degree bounds of D) in the order of :meth:`generate`: by the number of unknown
        coefficients, those of D weighted with :data:`DENOMINATOR_WEIGHT` (they make the system non-linear and
        expensive), then the lower level. Lazy: with many variables there are far more pairs than a search ever
        tries."""
        count = len(self.variables)
        if count == 0:
            yield from ((level, ()) for level in self.levels)
            return
        # the numerator of a level has (L + 1)^k coefficients, D with the bounds e has prod(e_i + 1) - 1
        costs = {level: (level + 1) ** count for level in self.levels}
        last = max((costs[level] + DENOMINATOR_WEIGHT * (_max_monomials(count, count * level) - 1)
                    for level in self.levels), default=0)
        for cost in range(1, last + 1):
            pairs: List[Tuple[int, Tuple[int, ...]]] = []
            for level in self.levels:
                rest = cost - costs[level]
                if rest < 0 or rest % DENOMINATOR_WEIGHT:
                    continue
                pairs.extend((level, bounds)
                             for bounds in _ordered_factorisations(rest // DENOMINATOR_WEIGHT + 1, count)
                             if sum(bounds) <= count * level)
            yield from sorted(pairs, key=lambda pair: (pair[0], sum(pair[1]), pair[1]))

    def generate(self) -> Iterator[Distribution]:
        for level, max_powers_denom in self.templates():
            num_degrees = [range(level + 1) for _ in self.variables]
            numerator = " + ".join((make_poly_clause(f"n_{c}", self.variables, pows)
                                    for c, pows in enumerate(itertools.product(*num_degrees))))
            denom_degrees = [range(max_powers_denom[i] + 1) for i, _ in enumerate(self.variables)]
            # the first monomial is the constant one (all powers 0), its coefficient is 1
            denominator = " + ".join(["1"] + [make_poly_clause(f"d_{c}", self.variables, pows) for c, pows in
                                              enumerate(itertools.product(*denom_degrees)) if c > 0])
            yield self.dist_fact.from_expr(f"({numerator}) / ({denominator})", *self.variables)
