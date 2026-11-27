"""Re-exports :mod:`prodigy.distribution.pgf.rules.independent_comparison` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.independent_comparison import (
    Coefficients,
    ComparisonNotImplemented,
    Coordinate,
    DEPENDENT,
    Element,
    NOT_A_DIFFERENCE,
    NOT_RATIONAL,
    filter_difference,
)

__all__ = ["Coefficients", "ComparisonNotImplemented", "Coordinate", "DEPENDENT", "Element", "NOT_A_DIFFERENCE",
           "NOT_RATIONAL", "filter_difference"]
