"""How the SymPy engine prints (#511): :class:`GeneratingFunctionSettings`, set per analysis with
:meth:`~prodigy.distribution.pgf.sympy.engine.SympyPGF.configured`."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["GeneratingFunctionSettings"]


@dataclass(frozen=True)
class GeneratingFunctionSettings:
    """How the SymPy engine prints; set per analysis, see :meth:`SympyPGF.configured`.

    No setting changes the computation: closed forms get their numerator and denominator factored, polynomials
    are expanded (see ``_arithmetic``). Simplifying
    the shown result is a matter of display (:mod:`prodigy.distribution.display`, #404)."""

    latex_output: bool = False
    """``str()`` gives LaTeX instead of a SymPy expression."""


_SETTINGS: ContextVar[GeneratingFunctionSettings] = ContextVar(
    "generating_function_settings", default=GeneratingFunctionSettings())
