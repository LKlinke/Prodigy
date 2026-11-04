"""
Soundness check for substitutions with roots of unity.

The residue classes of a variable x modulo q are extracted by evaluating the generating function f
at w x for the q-th roots of unity w (modulo guards `x % q = r`, `Distribution._arithmetic_progression`
and `x % q` in an assignment with an infinite marginal of x). The engines evaluate f(w x) with the
principal branches of roots and logarithms, which need not agree with the power series of f: for
example, `sqrt(x^2*y^2)` denotes the series x*y, but at -x it evaluates to `sqrt(x^2*y^2)` again
instead of -x*y.

The substitution is exact if the closed form of f is single-valued in x, i.e. built from x and
subexpressions without x by `+`, `*`, integer powers and entire functions (`exp`, `sin`, `cos`,
`sinh`, `cosh`); this covers all rational generating functions. Such an f is holomorphic in x on a
punctured disk around 0 except for isolated singularities. It agrees with its power series on a
positive interval, hence by the identity theorem on that whole domain, so f(w x) and the series at
w x agree there as well and have the same power series. All other generating functions (roots,
logarithms, absolute values, or anything else) are rejected with a :class:`NotSingleValuedError`.

The check is engine independent and works on SymPy expressions.
"""
from __future__ import annotations

import sympy

_ENTIRE = (sympy.exp, sympy.sin, sympy.cos, sympy.sinh, sympy.cosh)


class NotSingleValuedError(ValueError):
    """A substitution with roots of unity was requested on a generating function that is not
    provably single-valued in the substituted variable."""


def _single_valued(expression: sympy.Basic, variable: sympy.Symbol) -> bool:
    if not expression.has(variable) or expression == variable:
        return True
    if isinstance(expression, (sympy.Add, sympy.Mul)):
        return all(_single_valued(argument, variable) for argument in expression.args)
    if isinstance(expression, sympy.Pow):
        return isinstance(expression.exp, sympy.Integer) and _single_valued(expression.base, variable)
    if isinstance(expression, _ENTIRE):
        return _single_valued(expression.args[0], variable)
    return False


def is_single_valued_in(function: sympy.Expr, variable: str) -> bool:
    """
    Whether `function` is provably single-valued in the symbol called `variable` (see the module
    documentation). Other symbols may occur in arbitrary form, e.g. under a root, as long as these
    subexpressions do not contain `variable`.
    """
    symbols = [s for s in function.free_symbols if isinstance(s, sympy.Symbol) and s.name == variable]
    return all(_single_valued(function, symbol) for symbol in symbols)


def require_single_valued(function: sympy.Expr, variable: str, operation: str) -> None:
    """
    Raises a :class:`NotSingleValuedError` unless `function` is provably single-valued in
    `variable`. `operation` names the instruction for the error message, e.g. `x % 2`.
    """
    if not is_single_valued_in(function, variable):
        raise NotSingleValuedError(
            f"Cannot compute '{operation}' exactly: the generating function is not provably "
            f"single-valued in '{variable}' (built from rational functions and exp). The residue "
            f"classes modulo a number are extracted with complex roots of unity, which can pick the "
            f"wrong branch of roots or logarithms.")
