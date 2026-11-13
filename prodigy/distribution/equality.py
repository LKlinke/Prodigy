"""
Three-valued equality of SymPy expressions: proven equal, proven different, or unknown.

An engine's ``==`` must never claim equality without proof (#117), so for non-rational expressions it
answers False when it merely cannot prove equality. Checks such as the EVT fixed-point equation need to
tell "provably different" from "not proven" apart (#123); :func:`decide_equal` gives that answer.
"""
from typing import Any, Iterable, Optional

import sympy


def is_rational_over_q(expr: sympy.Expr) -> bool:
    """
    Whether ``expr`` is built from symbols and rational numbers with ``+``, ``*`` and integer powers only,
    i.e. it is a rational function with rational coefficients. Constants such as ``sqrt(2)`` or ``pi`` and
    floating-point numbers are excluded: for them, exact cancellation is not a decision procedure.
    """
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, (sympy.Symbol, sympy.Rational, sympy.Add, sympy.Mul)):
            continue
        if isinstance(node, sympy.Pow) and isinstance(node.exp, sympy.Integer):
            continue
        return False
    return True


def decide_equal(lhs: sympy.Expr, rhs: sympy.Expr, use_simplify: bool = True) -> Optional[bool]:
    """
    Decides ``lhs == rhs`` as an identity in all symbols.

    :returns: True if equality is proven, False if inequality is proven, None if neither.

    * Floating-point numbers make arithmetic inexact (``1e20 + 1 - 1e20`` is 0), so expressions containing
      them are only equal if they are identical; otherwise the answer is None.
    * The difference is put over a common denominator and cancelled. This is exact polynomial arithmetic
      over the rationals in which every non-rational subterm (e.g. ``exp(p*x)``) is an indeterminate, so a
      zero result proves the difference zero. If the difference, or its cancelled form, is a rational
      function over the rationals (:func:`is_rational_over_q`), a nonzero result proves it nonzero (cancelling
      is an identity, so ``exp(x) + 1 - exp(x)`` is proven different from 0).
    * Otherwise (only if ``use_simplify``) SymPy's ``Expr.equals`` may prove equality: it returns True only
      if ``simplify`` turns the difference into 0 or, for constant differences, by an algebraic proof
      (minimal polynomials). Its False comes from evaluation at random points and is no proof, so it counts
      as None.
    """
    if lhs == rhs:
        return True
    if lhs.has(sympy.Float) or rhs.has(sympy.Float):
        return None
    difference = lhs - rhs
    reduced = None
    try:
        reduced = sympy.cancel(sympy.together(difference))
        if reduced == 0:
            return True
    except sympy.PolynomialError:
        pass
    if is_rational_over_q(difference) or (reduced is not None and not reduced.has(sympy.Float)
                                          and is_rational_over_q(reduced)):
        return False
    if use_simplify and difference.equals(0) is True:
        return True
    return None


class _EngineStrPrinter(sympy.printing.str.StrPrinter):
    """SymPy's string form, but Euler's number as ``exp(1)``: Prodigy reads ``E`` back as a symbol (a variable or
    parameter may be called E), so ``E - 2`` would lose its value."""

    def _print_Exp1(self, expr):  # pylint: disable=invalid-name,unused-argument
        return "exp(1)"


def sympy_text(expr: Any) -> str:
    """The string form of a SymPy expression as Prodigy reads it back (:class:`_EngineStrPrinter`)."""
    return _EngineStrPrinter().doprint(expr)


def sympify_engine_string(text: str, names: Iterable[str]) -> sympy.Expr:
    """
    The SymPy expression of an engine's string form ``text`` (``^`` or ``**`` for powers). ``names`` are the
    variables and parameters; they are parsed as plain symbols, so names of Python builtins or SymPy objects
    (e.g. a variable ``sum``, ``max`` or ``S``) stay symbols instead of becoming functions (#321).

    The string must come from an engine, not from the user: SymPy's parser evaluates Python code.
    """
    local_dict = {name: sympy.Symbol(name) for name in names}
    return sympy.sympify(text.replace("^", "**"), locals=local_dict)


def sympify_distribution(dist: Any) -> sympy.Expr:
    """The SymPy expression of a distribution of any engine, see :func:`sympify_engine_string`."""
    return sympify_engine_string(str(dist), dist.get_variables() | dist.get_parameters())


def decide_equal_strings(lhs: str, rhs: str, names: Iterable[str]) -> Optional[bool]:
    """
    Like :func:`decide_equal` (without ``simplify``) for the string forms of two engine expressions, e.g. of
    SymEngine or GiNaC. ``names`` are the variables and parameters; they are parsed as plain symbols, so names
    such as ``E`` or ``S`` keep their meaning. Decimal literals stay floating-point numbers, so expressions
    containing them are never decided different. Unparsable strings give None.

    The strings must come from an engine, not from the user: SymPy's parser evaluates Python code.
    """
    local_dict = {name: sympy.Symbol(name) for name in names}
    transformations = sympy.parsing.sympy_parser.standard_transformations \
        + (sympy.parsing.sympy_parser.convert_xor,)
    try:
        lhs_expr = sympy.parse_expr(lhs, local_dict=local_dict, transformations=transformations)
        rhs_expr = sympy.parse_expr(rhs, local_dict=local_dict, transformations=transformations)
    except (SyntaxError, TypeError, ValueError, sympy.SympifyError):
        return None
    if not isinstance(lhs_expr, sympy.Expr) or not isinstance(rhs_expr, sympy.Expr):
        return None
    return decide_equal(lhs_expr, rhs_expr, use_simplify=False)
