from enum import Enum

from prodigy.analysis.evtinvariants.heuristics.positivity.individual_sum_terms import SumPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.iterate_terms import IterateTermsHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.logical import OrHeuristic, AndHeuristic, NotHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.polynomial_positivity import FinitePolynomialPositivity
from prodigy.analysis.evtinvariants.heuristics.positivity.positivity import PositivityHeuristic
from prodigy.analysis.evtinvariants.heuristics.positivity.rational_function import SingleRationalFunction, \
    RationalFunctionDenomSign


class PositivityHeuristics(Enum):
    INDIVIDUAL_SUM = SumPositivity
    ITERATE_TERMS = IterateTermsHeuristic
    OR = OrHeuristic
    AND = AndHeuristic
    NOT = NotHeuristic
    POLYNOMIAL = FinitePolynomialPositivity
    TRUE_RAT_FUNC = SingleRationalFunction
    MIXED_RAT_FUNC = RationalFunctionDenomSign
    DEFAULT = RationalFunctionDenomSign

    @staticmethod
    def create(heuristic_type: 'PositivityHeuristics', *args, **kwargs) -> PositivityHeuristic:
        return heuristic_type.value(*args, **kwargs)
