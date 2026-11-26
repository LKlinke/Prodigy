"""
-----------
Engine text
-----------

*Engine text* is how expressions travel between Prodigy and the computer algebra of its engines as strings: the
expressions of a program on their way into an engine (written by :func:`prodigy.language.engine_text.to_engine_text`),
and the generating functions, masses and coefficients an engine gives back (written by the engine, or by
:func:`sympy_text` for SymPy expressions). Its grammar is in ``docs/source/language-design.md`` ("Engine text").

:func:`parse_engine_text` is the one parser of engine text into SymPy; the engines, the analysis and the front
ends use it (directly or through the shorthands in :mod:`prodigy.distribution.pgf.rules.equality`), and nothing
else parses engine text. It lives in the distribution layer because it produces SymPy objects; the language layer only
writes engine text and knows nothing of SymPy.

Engine text must come from Prodigy or an engine, never from a user or a language model: SymPy's parser evaluates
Python code. Untrusted text is validated by the ``.ppy`` parser or an allowlist first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Union

import sympy
from sympy.parsing.sympy_parser import convert_xor, parse_expr, rationalize, standard_transformations

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.distribution import Distribution

__all__ = ["parse_engine_text", "sympy_text", "engine_text"]


def _function(name: str) -> Callable[..., Any]:
    """A called name in engine text: a SymPy function (``exp``, ``log``, ``Min``, …) or ``sum``."""
    if name == "sum":
        return sum
    return getattr(sympy, name)


def parse_engine_text(text: str, names: Optional[Iterable[str]] = None, *, rational: bool = False,
                      symbol: Callable[[str], sympy.Symbol] = sympy.Symbol) -> sympy.Expr:
    """
    The SymPy expression of the engine text ``text`` (``^`` or ``**`` for powers).

    How names are read depends on ``names``:

    * ``names`` given (the variables and parameters): exactly these are symbols, so names of Python builtins or
      SymPy objects (a variable ``sum``, ``max``, ``E`` or ``S``) stay symbols (#321); every other name has its
      SymPy meaning (``E``, ``pi``, ``oo``, ``exp``, …).
    * ``names`` None: every name that is not called is a symbol, also ``E``, ``pi`` or ``I`` (#354); a called name
      is a SymPy function (``exp(1)`` is Euler's number) or ``sum``.

    ``rational`` reads decimal literals as exact rationals (``0.1`` is ``1/10``); otherwise they stay
    floating-point numbers, which the exact checks never call different. ``symbol`` makes the symbols (e.g.
    real ones, as the SymPy engine uses).

    .. doctest::

        >>> parse_engine_text("x^2/(2 - y)", ["x", "y"])
        x**2/(2 - y)
        >>> parse_engine_text("E + exp(1)")
        E + E
        >>> parse_engine_text("0.5*x", rational=True)
        x/2
    """
    transformations = standard_transformations + (convert_xor,) + ((rationalize,) if rational else ())
    if names is not None:
        local_dict: Dict[str, Any] = {name: symbol(name) for name in names}
        return parse_expr(text, local_dict=local_dict, transformations=transformations)
    global_dict: Dict[str, Any] = {"Symbol": symbol, "Function": _function, "Integer": sympy.Integer,
                                   "Float": sympy.Float, "Rational": sympy.Rational}
    return parse_expr(text, global_dict=global_dict, transformations=transformations)


class _EngineStrPrinter(sympy.printing.str.StrPrinter):
    """SymPy's string form, but Euler's number as ``exp(1)``: Prodigy reads ``E`` back as a symbol (a variable or
    parameter may be called E), so ``E - 2`` would lose its value."""

    def _print_Exp1(self, expr: sympy.Basic) -> str:  # pylint: disable=invalid-name,unused-argument
        return "exp(1)"


def sympy_text(expr: sympy.Basic) -> str:
    """The engine text of a SymPy expression (:class:`_EngineStrPrinter`)."""
    return _EngineStrPrinter().doprint(expr)


def engine_text(value: Union[str, sympy.Basic, "Distribution"]) -> str:
    """
    The engine text of a value: a string stays, a SymPy expression goes through :func:`sympy_text`, a
    distribution gives its engine's own text (``Distribution._engine_text``, independent of the printing flags:
    ``str`` of a SymPy generating function is LaTeX while ``latex_output`` is active, #445, #465). Use it instead
    of ``str(value)`` where the text is parsed again.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, sympy.Basic):
        return sympy_text(value)
    return value._engine_text()  # pylint: disable=protected-access
