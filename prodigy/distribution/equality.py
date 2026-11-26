"""Re-exports :mod:`prodigy.distribution.pgf.rules.equality` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.equality import (
    decide_equal,
    decide_equal_strings,
    engine_text,
    is_rational_over_q,
    parse_engine_text,
    sympify_distribution,
    sympify_distribution_string,
    sympify_engine_string,
    sympify_symbolic_string,
    sympy_text,
)

__all__ = ["decide_equal", "decide_equal_strings", "engine_text", "is_rational_over_q", "parse_engine_text",
           "sympify_distribution", "sympify_distribution_string", "sympify_engine_string", "sympify_symbolic_string",
           "sympy_text"]
