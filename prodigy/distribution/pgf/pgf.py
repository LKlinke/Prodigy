"""The class :class:`PGF`: the shared logic of all engines of generating functions (#511)."""
from __future__ import annotations

from typing import Sequence, Union

from prodigy.distribution.values import DistributionParam
from prodigy.distribution.pgf.sampling import sampling_pgf
from prodigy.distribution.pgf.updating import PGFUpdating
from prodigy.language import VarExpr
from prodigy.language.engine_text import engine_text_of


class PGF(PGFUpdating):  # pylint: disable=abstract-method  # the engines implement the hooks
    """
    A probability generating function: the shared logic of all engines; an engine adds the CAS hooks.

    The factory of the common distributions (:class:`~prodigy.distribution.distribution.CommonDistributionsFactory`)
    is the engine class itself (decided in #511, e.g. ``SympyPGF.geometric("x", "1/2")``): each distribution is
    built from the shared table :func:`~prodigy.distribution.pgf.sampling.sampling_pgf` through the engine's
    :meth:`from_expr`, so all engines validate the parameters alike.
    """

    @classmethod
    def _sampling(cls, function: str, var: Union[str, VarExpr], parameters: Sequence[DistributionParam]) -> PGF:
        sample = sampling_pgf(function, parameters, engine_text_of(var))
        return cls.from_expr(sample.text, engine_text_of(var), closed=True, finite=sample.finite)

    @classmethod
    def geometric(cls, var: Union[str, VarExpr], p: DistributionParam) -> PGF:
        return cls._sampling("geometric", var, [p])

    @classmethod
    def uniform(cls, var: Union[str, VarExpr], lower: DistributionParam, upper: DistributionParam) -> PGF:
        return cls._sampling("unif", var, [lower, upper])

    @classmethod
    def bernoulli(cls, var: Union[str, VarExpr], p: DistributionParam) -> PGF:
        return cls._sampling("bernoulli", var, [p])

    @classmethod
    def poisson(cls, var: Union[str, VarExpr], lam: DistributionParam) -> PGF:
        return cls._sampling("poisson", var, [lam])

    @classmethod
    def log(cls, var: Union[str, VarExpr], p: DistributionParam) -> PGF:
        return cls._sampling("logdist", var, [p])

    @classmethod
    def binomial(cls, var: Union[str, VarExpr], n: DistributionParam, p: DistributionParam) -> PGF:
        return cls._sampling("binomial", var, [n, p])

    @classmethod
    def negative_binomial(cls, var: Union[str, VarExpr], r: DistributionParam, p: DistributionParam) -> PGF:
        return cls._sampling("negative_binomial", var, [r, p])


__all__ = ["PGF"]
