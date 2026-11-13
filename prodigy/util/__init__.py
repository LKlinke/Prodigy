"""
=================
``prodigy.util``
=================

.. autofunction:: prodigy.util.make_poly_clause
"""
from typing import Collection


def make_poly_clause(coef: str, variables: Collection[str], powers: Collection[int]) -> str:
    """
        Generates a polynomial clause _alpha_ * X^a * Y^b * Z^c ...
    """
    c = [coef]
    vp = (['', '{}', '({}^{})'][min(p, 2)].format(v, p) for v, p in zip(variables, powers))
    return '*'.join(c + [s for s in vp if s])
