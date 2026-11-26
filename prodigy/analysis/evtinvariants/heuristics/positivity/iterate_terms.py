from typing import Collection, Iterator, Optional

from prodigy.distribution.coefficient_signs import series_form, sign
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic, PositivityInput
from prodigy.distribution import CommonDistributionsFactory, Distribution
from prodigy.distribution.equality import engine_text
from prodigy.util.three_valued import all_three_valued


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

    def _is_positive(self, f: PositivityInput) -> Optional[bool]:
        dist = self._factory.from_expr(engine_text(f).replace("**", "^"), *self._vars)
        return all_three_valued(self._signs(dist))

    def _signs(self, dist: Distribution) -> Iterator[Optional[bool]]:
        """The signs of the first coefficients, and ``None`` once more terms exist than are inspected."""
        for i, (prob, _) in enumerate(dist):
            if i >= self._max_iters:
                yield None
                return
            coefficient, _ = series_form(prob, self._vars)
            yield sign(coefficient)
