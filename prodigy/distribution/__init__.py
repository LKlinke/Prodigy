"""
Distributions: the semantic interface, the generating-function logic and the engines (#511).

* :mod:`prodigy.distribution.distribution`: the :class:`Distribution` interface and the
  :class:`CommonDistributionsFactory`; :mod:`~prodigy.distribution.values`, :mod:`~prodigy.distribution.errors`
  and :mod:`~prodigy.distribution.terms` hold the value types, errors and term helpers it uses.
* :mod:`prodigy.distribution.pgf`: the logic on probability generating functions
  (:class:`~prodigy.distribution.pgf.pgf.PGF`, in layers), the proof rules on closed forms
  (:mod:`prodigy.distribution.pgf.rules`) and the engines
  :class:`~prodigy.distribution.pgf.sympy.engine.SympyPGF` (default) and
  :class:`~prodigy.distribution.pgf.ginac.engine.GinacPGF`, which implement only the CAS hooks.
* :mod:`prodigy.distribution.kernel`: exact arithmetic on rational functions (FLINT).
* :mod:`prodigy.distribution.display` and :mod:`prodigy.distribution.engine_text`: how closed forms are spelled
  and read.

It owns the meaning of a program state, not which engine is chosen, loops or results (``prodigy.analysis``).
The modules at the old paths (``generating_function``, ``fast_generating_function``, ``equality``, …) only
re-export their names until the end of v0.7.0.
"""

from .distribution import CommonDistributionsFactory, Distribution
from .errors import NonFiniteMarginalError, UndecidedComparisonError, ZeroDivisorError, known_convergent_series
from .values import DistributionParam, MarginalType, State

# the API of the package (#379, step 1)
__all__ = ["CommonDistributionsFactory", "Distribution", "DistributionParam", "MarginalType", "NonFiniteMarginalError",
           "ZeroDivisorError", "State", "known_convergent_series", "UndecidedComparisonError"]
