"""Re-exports :mod:`prodigy.distribution.pgf.rules.dominant_pole` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.dominant_pole import (
    MAX_EXACT_COEFFICIENTS,
    PRECISIONS,
    PoleVerdict,
    Witness,
    dominant_pole_verdict,
    rational_verdict,
)

__all__ = ["MAX_EXACT_COEFFICIENTS", "PRECISIONS", "PoleVerdict", "Witness", "dominant_pole_verdict",
           "rational_verdict"]
