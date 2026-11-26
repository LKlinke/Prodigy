"""Re-exports :mod:`prodigy.distribution.pgf.rules.exponential_series` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.exponential_series import (
    KernelTerm,
    Window,
    exp_polynomial_columns,
    exp_polynomial_taylor_stream,
)

__all__ = ["KernelTerm", "Window", "exp_polynomial_columns", "exp_polynomial_taylor_stream"]
