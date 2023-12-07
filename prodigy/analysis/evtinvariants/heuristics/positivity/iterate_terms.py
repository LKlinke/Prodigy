from typing import Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.distribution import CommonDistributionsFactory


class IterateTermsHeuristic(PositivityHeuristic):
    """
        This heuristic generates the first _iterations_ terms (default: 10) and checks whether there is
        a negative coefficient, hence witnessing negativity of the formal power series. For the term generation,
        the heuristics it relies on the distribution functionality to generate terms.

        ..returns:
            - False: The given distribution is not non-negative (i.e., there is at least one negative coefficient)
            - None: The first _iterations_ may coefficients were >= 0.
    """

    def __init__(self,
                 dist_fact: CommonDistributionsFactory,
                 iterations: int = 10,
                 sub_heuristic: PositivityHeuristic = None,
                 *variables: str):
        super().__init__(sub_heuristic)
        self._factory = dist_fact
        self._max_iters = iterations
        self._vars = variables

    def _is_positive(self, f: str) -> Optional[bool]:
        dist = self._factory.from_expr(f, *self._vars)
        for i, (prob, state) in enumerate(dist):
            if i >= self._max_iters:
                return None
            if sympy.S(prob) < 0:
                return False
