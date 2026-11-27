"""Re-exports :mod:`prodigy.distribution.pgf.rules.finite_measure` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.finite_measure import (
    Verdict,
    check_input_expression,
    check_probability_generating_function,
    check_series,
)

__all__ = ["Verdict", "check_input_expression", "check_probability_generating_function", "check_series"]
