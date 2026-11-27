"""
Re-exports the SymPy engine :mod:`prodigy.distribution.pgf.sympy.engine` (moved in #511) until the end of v0.7.0.
``GeneratingFunction`` is the deprecated name of :class:`~prodigy.distribution.pgf.sympy.engine.SympyPGF`, which is
the engine and its factory at once.
"""

from prodigy.distribution.pgf.sympy.engine import (
    GeneratingFunction,
    MAX_FACTORED_PARAMETERS,
    SympyPGF,
)
from prodigy.distribution.pgf.sympy.settings import GeneratingFunctionSettings
from prodigy.distribution.pgf.rules.value_at_one import (
    _continuous_at_one,
    _substitute_one,
)

__all__ = ["GeneratingFunction", "GeneratingFunctionSettings", "MAX_FACTORED_PARAMETERS", "SympyPGF",
           "_continuous_at_one", "_substitute_one"]
