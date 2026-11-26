"""Re-exports :mod:`prodigy.distribution.pgf.coefficient_stream` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.coefficient_stream import (
    Clock,
    Coefficient,
    FIRST_ORDER,
    STEP_BUDGET,
    Waiting,
    doubling,
    marginal_stream,
    series_prefix,
    sympy_form,
    taylor_stream,
    taylor_window,
)

__all__ = ["Clock", "Coefficient", "FIRST_ORDER", "STEP_BUDGET", "Waiting", "doubling", "marginal_stream",
           "series_prefix", "sympy_form", "taylor_stream", "taylor_window"]
