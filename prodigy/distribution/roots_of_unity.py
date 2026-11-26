"""Re-exports :mod:`prodigy.distribution.pgf.rules.roots_of_unity` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.roots_of_unity import (
    NotSingleValuedError,
    is_single_valued_in,
    require_single_valued,
    root_of_unity,
)

__all__ = ["NotSingleValuedError", "is_single_valued_in", "require_single_valued", "root_of_unity"]
