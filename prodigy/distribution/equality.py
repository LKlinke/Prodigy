"""
Three-valued equality of SymPy expressions: proven equal, proven different, or unknown.

An engine's ``==`` must never claim equality without proof (#117), so for non-rational expressions it
answers False when it merely cannot prove equality. Checks such as the EVT fixed-point equation need to
tell "provably different" from "not proven" apart (#123); :func:`decide_equal` gives that answer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Union

import re
import sympy

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.distribution import Distribution


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


_SAMPLE_POINTS = (sympy.Rational(1, 3), sympy.Rational(2, 5), sympy.Rational(3, 7))


def _numerically_nonzero(expression: sympy.Expr) -> bool:
    """
    Whether ``expression`` is clearly nonzero at some point (high-precision evaluation, magnitude far above
    the error): then it is not identically 0 and ``Expr.equals`` cannot prove it so. A cheap pre-check, since
    ``equals`` may take minutes on sums of exponentials with complex roots of unity (the residue-class filter
    ``y % 3 == 1`` of a Poisson draw). False means only "not shown nonzero".
    """
    symbols = sorted(expression.free_symbols, key=str)
    for shift in range(len(_SAMPLE_POINTS)):
        point = {symbol: _SAMPLE_POINTS[(index + shift) % len(_SAMPLE_POINTS)] for index, symbol in enumerate(symbols)}
        try:
            value = sympy.N(expression.subs(point), 40)
            if value.is_number and value.is_finite and abs(value) > sympy.Float("1e-15", 40):
                return True
        except (ArithmeticError, ValueError, TypeError, AttributeError):
            continue
    return False


def _constants_in_normal_form(expression: sympy.Expr) -> sympy.Expr:
    """
    ``expression`` with every constant subterm that is not a rational number written as ``a + b*I`` (an exact
    rewrite, ``expand_complex``): ``exp(2*I*pi/3)`` becomes ``-1/2 + sqrt(3)*I/2``. Two engines write the same roots
    of unity differently, and ``cancel`` of the difference with both forms works over an algebraic extension
    that takes minutes (the residue classes ``y % 3`` of a Poisson draw), while after the rewrite the terms
    cancel at once.
    """
    try:
        return expression.replace(lambda node: node.is_number and not node.is_Rational and not node.is_Atom,
                                  sympy.expand_complex)
    except (ArithmeticError, ValueError, TypeError, sympy.PolynomialError):
        return expression


def _has_irrational_constants(expression: sympy.Expr) -> bool:
    """Whether ``expression`` has a constant subterm that is not a rational number (``I``, ``sqrt(3)``, ``exp(1)``)."""
    return any(node.is_number and not node.is_Rational for node in sympy.preorder_traversal(expression))


NORMAL_FORM_LIMIT = 3000
"""Operations of the difference in normal form (:func:`_constants_in_normal_form`) up to which ``cancel`` is
tried over the algebraic extension; for larger ones only the cheap proof (:func:`_cancels_with_frozen_constants`)."""


def _cancels_with_frozen_constants(expression: sympy.Expr) -> bool:
    """
    Whether ``expression`` cancels to 0 with every constant subterm that is not a rational number replaced by an
    indeterminate. That is a polynomial identity over the rationals, so it holds for the actual values as well:
    True proves ``expression == 0``; False means only "not shown".
    """
    frozen = expression.replace(lambda node: node.is_number and not node.is_Rational and not node.is_Atom,
                                lambda node: sympy.Dummy("c"))
    try:
        return sympy.cancel(sympy.together(frozen)) == 0
    except sympy.PolynomialError:
        return False


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
        if _has_irrational_constants(difference) and _numerically_nonzero(difference):
            # nonzero, so no proof of 0; and not rational over Q (the constants stay in it), so no proof of
            # inequality: cancelling over the algebraic extension (seconds) cannot decide anything
            return None
        normal_form = _constants_in_normal_form(difference)
        if normal_form.count_ops() <= NORMAL_FORM_LIMIT:
            reduced = sympy.cancel(sympy.together(normal_form))
            if reduced == 0:
                return True
        elif _cancels_with_frozen_constants(difference):
            return True
    except sympy.PolynomialError:
        pass
    if is_rational_over_q(difference) or (reduced is not None and not reduced.has(sympy.Float)
                                          and is_rational_over_q(reduced)):
        return False
    if use_simplify and not _numerically_nonzero(difference) and difference.equals(0) is True:
        return True
    return None


class _EngineStrPrinter(sympy.printing.str.StrPrinter):
    """SymPy's string form, but Euler's number as ``exp(1)``: Prodigy reads ``E`` back as a symbol (a variable or
    parameter may be called E), so ``E - 2`` would lose its value."""

    def _print_Exp1(self, expr: sympy.Basic) -> str:  # pylint: disable=invalid-name,unused-argument
        return "exp(1)"


def sympy_text(expr: sympy.Basic) -> str:
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


_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b(?!\s*\()")


def sympify_symbolic_string(text: Union[str, int, float, sympy.Basic]) -> sympy.Expr:
    """
    The SymPy expression of an expression string (anything else is sympified as it is) whose variable and
    parameter names are not known (``^`` or ``**`` for powers). Every identifier that is not a function call
    (not followed by ``(``) is a plain symbol, so variables named like builtins or constants (``pow``, ``pi``,
    ``I``, ``S``) stay symbols (#354); real functions (``exp``, ``log``, ``sqrt``, ...) keep their meaning.

    The string must come from Prodigy, not from the user: SymPy's parser evaluates Python code.
    """
    if not isinstance(text, str):
        return sympy.sympify(text)
    return sympify_engine_string(text, set(_IDENTIFIER.findall(text)))


def sympify_distribution(dist: Distribution) -> sympy.Expr:
    """
    The SymPy expression of a distribution of any engine: the engine's own expression
    (``Distribution._series_expression``), so it does not depend on the printing flags (``str`` of a SymPy
    generating function is LaTeX while ``latex_output`` is active, #445).
    """
    expression = dist._series_expression()  # pylint: disable=protected-access
    # as parsed from the string form before: plain symbols, without the assumptions of the engine's symbols
    return expression.xreplace({symbol: sympy.Symbol(symbol.name) for symbol in expression.free_symbols})


def sympify_distribution_string(dist: Distribution) -> sympy.Expr:
    """
    The SymPy expression of a distribution whose engine has no SymPy expression, parsed from its string form,
    see :func:`sympify_engine_string`. Only for engines whose ``str`` is a plain expression (#445).
    """
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
