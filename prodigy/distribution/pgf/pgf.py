"""The class :class:`PGF`: the shared logic of all engines of generating functions (#511)."""
from prodigy.distribution.pgf.updating import PGFUpdating


class PGF(PGFUpdating):  # pylint: disable=abstract-method  # the engines implement the hooks
    """A probability generating function: the shared logic of all engines; an engine adds the CAS hooks."""


__all__ = ["PGF"]
