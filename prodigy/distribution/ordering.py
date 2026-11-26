"""Re-exports :mod:`prodigy.distribution.pgf.rules.ordering` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.ordering import (
    Coefficient,
    OrderVerdict,
    SEARCH_COEFFICIENTS,
    UndecidedComparisonError,
    coefficient_verdict,
    dominates_finite,
    nonneg_verdict,
)

__all__ = ["Coefficient", "OrderVerdict", "SEARCH_COEFFICIENTS", "UndecidedComparisonError", "coefficient_verdict",
           "dominates_finite", "nonneg_verdict"]
