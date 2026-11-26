"""Re-exports :mod:`prodigy.distribution.pgf.rules.convergence` (moved in #511) until the end of v0.7.0."""

from prodigy.distribution.pgf.rules.convergence import (
    Obstacle,
    marginal_obstacle,
)

__all__ = ["Obstacle", "marginal_obstacle"]
