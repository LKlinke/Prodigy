"""
Exact kernels below the :class:`~prodigy.distribution.distribution.Distribution` logic (Prodigy#417).

A kernel is not an engine: it has no case analysis (support splits, guards, zero mass), only CAS operations
that the shared ``Distribution`` logic or an engine may call as hooks. Nothing in the analysis uses a kernel yet.

* :mod:`prodigy.distribution.kernel.rational`: rational generating functions over ℚ on FLINT
  (:class:`~prodigy.distribution.kernel.rational.RationalGF`).
"""
from prodigy.distribution.kernel.rational import NotRepresentableError, PoleAtZeroError, RationalGF

__all__ = ["NotRepresentableError", "PoleAtZeroError", "RationalGF"]
