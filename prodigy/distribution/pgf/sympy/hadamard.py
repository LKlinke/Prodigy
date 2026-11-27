"""The Hadamard product of two univariate rational generating functions on SymPy (#511), split from
:meth:`~prodigy.distribution.pgf.sympy.engine.SympyPGF.hadamard_product`."""
from __future__ import annotations

import logging

import sympy

from prodigy.distribution.pgf.sympy.expressions import _parse_to_sympy
from prodigy.language.engine_text import engine_text_of

logger = logging.getLogger("generating_function")

__all__ = ["rational_hadamard_product"]


def rational_hadamard_product(f: sympy.Expr, g: sympy.Expr, var: sympy.Symbol) -> sympy.Expr:
    """The function whose coefficients in ``var`` are the products of those of the rational functions ``f`` and
    ``g``: termwise for a polynomial, otherwise by the Kronecker product of the companion matrices of the
    denominators (the denominator of the result) and the series of the first ``deg f * deg g`` terms."""
    s_f, s_g = f, g

    if not (s_f.is_rational_function() and s_g.is_rational_function()):
        raise ValueError(f"Distributions need to be rational functions. Got {s_f} and {s_g}")

    if s_f.is_polynomial(var):
        s_f, s_g = s_g, s_f
    if s_g.is_polynomial(var):
        if s_g.is_constant(var):
            return s_g * s_f.subs(var, 0)
        return _coefficient_product(s_f, s_g, var, s_g.as_poly(var).degree() + 1).as_expr()

    _, s_f_denom = s_f.as_numer_denom()
    _, s_g_denom = s_g.as_numer_denom()
    f_degree = s_f_denom.as_poly(var).degree()
    g_degree = s_g_denom.as_poly(var).degree()
    logger.debug("Degree f: %i\nDegree g: %i", f_degree, g_degree)

    s_compm_f = sympy.Matrix(sympy.matrices.expressions.CompanionMatrix(
        (var ** f_degree * s_f_denom.subs(var, 1 / var)).as_poly(var).monic()))
    s_compm_g = sympy.Matrix(sympy.matrices.expressions.CompanionMatrix(
        (var ** g_degree * s_g_denom.subs(var, 1 / var)).as_poly(var).monic()))

    s_result_denom = sympy.det(
        sympy.eye(f_degree * g_degree) - (sympy.matrices.kronecker_product(s_compm_f, s_compm_g)) * var)
    logger.debug("Denominator: %s", s_result_denom)
    logger.debug("Comparing %i terms", f_degree * g_degree)

    result = _coefficient_product(s_f, s_g, var, f_degree * g_degree) * s_result_denom
    s_num = result.as_expr().series(var, n=f_degree * g_degree).removeO()

    logger.debug("Resulting function: %s", s_num / s_result_denom)
    # read back from text, as the engine reads every closed form
    return _parse_to_sympy(engine_text_of(s_num / s_result_denom))


def _coefficient_product(f: sympy.Expr, g: sympy.Expr, var: sympy.Symbol, count: int) -> sympy.Poly:
    """The polynomial whose coefficients of ``var^0``, …, ``var^(count-1)`` are the products of those of ``f`` and
    ``g``."""
    f_coefficients = reversed(f.series(var, n=count).removeO().as_poly(var).all_coeffs())
    g_coefficients = reversed(g.series(var, n=count).removeO().as_poly(var).all_coeffs())
    # coefficients from degree 0 up; zip stops at the shorter list, whose missing coefficients are 0
    return sympy.polys.Poly(reversed([cf * cg for cf, cg in zip(f_coefficients, g_coefficients, strict=False)]), var)
