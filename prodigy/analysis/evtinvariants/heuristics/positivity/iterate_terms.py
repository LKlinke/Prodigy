from typing import Optional

import sympy

from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.distribution import CommonDistributionsFactory


class IterateTermsHeuristic(PositivityHeuristic):

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
        dist = self._factory.from_expr(f, self._vars)
        for i, (prob, state) in enumerate(dist):
            if i >= self._max_iters:
                return None
            if sympy.S(prob) < 0:
                return False
