"""Re-exports :mod:`prodigy.distribution.pgf.rules.linear_guards` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.linear_guards import (
    DifferenceComparison,
    LinearForm,
    difference_comparison,
    normalize_guard,
)

__all__ = ["DifferenceComparison", "LinearForm", "difference_comparison", "normalize_guard"]
