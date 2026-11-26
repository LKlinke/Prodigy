"""Re-exports :mod:`prodigy.distribution.pgf.rules.rational_series` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.rational_series import (
    GroundElement,
    rational_taylor_coefficients,
    rational_taylor_stream,
)

__all__ = ["GroundElement", "rational_taylor_coefficients", "rational_taylor_stream"]
