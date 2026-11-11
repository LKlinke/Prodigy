from typing import Collection, Optional

from prodigy.analysis.evtinvariants.heuristics.positivity.coefficients import series_form, sign
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.distribution import CommonDistributionsFactory


class IterateTermsHeuristic(PositivityHeuristic):
    """
        This heuristic generates the first _iterations_ terms (default: 10) of the power series in ``variables``
        and checks whether there is a negative coefficient, hence witnessing negativity of the formal power series.
        For the term generation, the heuristics it relies on the distribution functionality to generate terms.

        ..returns:
            - False: There is a coefficient that is negative (for every real value of the parameters)
            - True: The series is finite and all its coefficients are non-negative
            - None: The first _iterations_ many coefficients were not shown to be negative
    """

    def __init__(self,
                 dist_fact: CommonDistributionsFactory,
                 iterations: int = 10,
                 sub_heuristic: Optional[PositivityHeuristic] = None,
                 *, variables: Collection[str]):
        super().__init__(sub_heuristic)
        self._factory = dist_fact
        self._max_iters = iterations
        self._vars = tuple(sorted(map(str, variables)))

    def _is_positive(self, f) -> Optional[bool]:
        dist = self._factory.from_expr(str(f).replace("**", "^"), *self._vars)
        verdict: Optional[bool] = True
        for i, (prob, _) in enumerate(dist):
            if i >= self._max_iters:
                return None
            coefficient, _ = series_form(prob, self._vars)
            result = sign(coefficient)
            if result is False:
                return False
            if result is None:
                verdict = None
        return verdict
