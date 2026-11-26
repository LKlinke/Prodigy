"""
The shared logic of generating functions (#511, ``docs/source/architecture.md``):
:class:`~prodigy.distribution.pgf.pgf.PGF` implements the :class:`~prodigy.distribution.distribution.Distribution`
interface on probability generating functions, with the case analysis of every operation and the proof rules
(:mod:`prodigy.distribution.pgf.rules`). The engines (:mod:`prodigy.distribution.pgf.sympy`,
:mod:`prodigy.distribution.pgf.ginac`) implement only the CAS hooks of
:class:`~prodigy.distribution.pgf.hooks.PGFHooks`.

The logic is split into layers, each a module: :mod:`~prodigy.distribution.pgf.base` (names, comparisons,
marginals and convergence proofs), :mod:`~prodigy.distribution.pgf.filtering` (filters and the queries computed with
them) and :mod:`~prodigy.distribution.pgf.updating` (updates and their case splits).
"""
