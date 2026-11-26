"""
Re-exports the GiNaC engine :mod:`prodigy.distribution.pgf.ginac.engine` (moved in #511) until the end of v0.7.0.
``FPS`` and ``ProdigyPGF`` are the deprecated names of :class:`~prodigy.distribution.pgf.ginac.engine.GinacPGF`,
which is the engine and its factory at once.
"""

from prodigy.distribution.pgf.ginac.engine import (
    FPS,
    GinacPGF,
    ProdigyPGF,
)

__all__ = ["FPS", "GinacPGF", "ProdigyPGF"]
