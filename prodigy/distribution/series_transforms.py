"""Re-exports :mod:`prodigy.distribution.pgf.rules.series_transforms` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.series_transforms import (
    MonomialSubstitution,
    Shift,
    low_coefficients_vanish,
    monomial_substitution,
    shift,
)

__all__ = ["MonomialSubstitution", "Shift", "low_coefficients_vanish", "monomial_substitution", "shift"]
