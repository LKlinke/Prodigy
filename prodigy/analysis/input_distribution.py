"""
Validation of input distributions given in closed form (#296).

An input distribution is a generating function typed in by a person: the CLI argument of
``prodigy analyze`` (and ``main``, ``invariant_synthesis``, ``evt_lower_bound``), the field of the
GUI's Setup, the ``input_distribution`` of an :class:`~prodigy.analysis.session.AnalysisSession`,
or the distribution passed to :func:`~prodigy.analysis.analyzer.compute_discrete_distribution`.
The analysis treats it as a (sub-)probability distribution: residual masses and lower bounds
assume non-negative coefficients, and masses are read off the closed form at 1. So an input is
accepted only when Prodigy *proves* that it is

* a power series at 0 in the program variables,
* with non-negative coefficients,
* and finite mass (the sum of all coefficients).

An input for which one of these is refuted, or cannot be proven, is refused with
:class:`InputDistributionError`, naming the input and the reason. Refusing is preferred over
running the analysis with every guarantee downgraded to ``unknown``: a result without any
guarantee is of little use, and the message tells how to state the input so that it can be
checked.

The proofs, all exact:

* polynomials: every coefficient is checked; with parameters, SymPy must prove its sign from the
  parameter's type (``nparam`` is a natural number, ``rparam`` has no known sign);
* rational functions: the reduced denominator does not vanish at 0 (power series); the
  coefficients are non-negative by the exact finite positivity check for product denominators
  (:func:`~prodigy.analysis.lowerbounds.subinvariant.to_product_form`, which also refutes) or by
  the construction ``P / prod (c - R)``
  (:func:`~prodigy.analysis.lowerbounds.subinvariant.nonneg_by_construction`); the mass is finite
  iff the diagonal has no pole in ``(0, 1]`` (Pringsheim,
  :func:`~prodigy.analysis.lowerbounds.subinvariant.rational_mass`);
* other closed forms are taken apart: sums, products and positive integer powers of certified
  series are certified, and so is ``exp(G)`` when ``G - G(0)`` is (``exp(G(0)) > 0``, and
  ``exp(H) = sum H^n / n!``); e.g. the Poisson ``exp(3*(x - 1))``. Anything else is not proven.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sympy

from prodigy.analysis.exceptions import ForwardAnalysisError


class InputDistributionError(ForwardAnalysisError):
    """The input distribution is refused: it is not proven to be a power series at 0 with
    non-negative coefficients and finite mass. ``details["proof"]`` says whether this was refuted
    (``refuted``) or could not be proven (``not_proven``). It is not called ``verdict``, which the
    result views explain as the verdict on a loop invariant."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


# A verdict: True (proven), False (refuted) or None (not proven), with the reason if not proven.
_Verdict = Tuple[Optional[bool], str]
_PROVEN: _Verdict = (True, "")

_CONSTANTS = frozenset({"E", "pi", "I"})

REQUIREMENT = ("Prodigy accepts an input distribution only when it can prove that it is a power series at 0 "
               "in the program variables with non-negative coefficients and finite mass.")


def input_distribution_from_text(config, text: str, program, syntax: str = "pgcl"):
    """The distribution of the closed form ``text`` for ``program``, after checking that ``text``
    is a generating function that is safe to give to the engines (SymPy evaluates Python code)
    and that it is a distribution (:func:`validate_input_distribution`)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import MATH_FUNCTIONS, CandidateError, admissible_expression
    names = [str(v) for v in program.variables] + [str(p) for p in program.parameters]
    try:
        spelled = admissible_expression(text, names=names, syntax=syntax, functions=MATH_FUNCTIONS)
    except CandidateError as exc:
        raise InputDistributionError(f"The input distribution is refused: {exc}",
                                     details={"type": "input_distribution", "proof": "refuted",
                                              "input": text}) from exc
    dist = config.factory.from_expr(spelled, *program.variables.keys())
    validate_input_distribution(dist, program)
    return dist


def validate_input_distribution(dist, program) -> None:
    """Raises :class:`InputDistributionError` unless ``dist`` is proven to be a power series at 0
    in its variables (the program's and its own) with non-negative coefficients and finite mass.
    Every symbol that is not a variable is a parameter; ``program`` gives their types."""
    variables = sorted({str(v) for v in program.variables} | {str(v) for v in dist.get_variables()})
    parameter_types = {str(p): t for p, t in program.parameters.items()}
    text = _plain_text(dist)
    verdict, reason = check_input_expression(text, variables, parameter_types)
    if verdict is True:
        return
    if verdict is False:
        message = f"The input distribution {text} is not a distribution: {reason}."
    else:
        message = f"The input distribution {text} is refused: {reason}. {REQUIREMENT}"
    raise InputDistributionError(message, details={"type": "input_distribution", "input": text,
                                                   "proof": "refuted" if verdict is False else "not_proven",
                                                   "reason": reason})


def _plain_text(dist) -> str:
    """The closed form of ``dist`` as plain text (the SymPy engine prints LaTeX under ``use_latex``)."""
    function = getattr(dist, "_function", None)
    return str(function) if isinstance(function, sympy.Basic) else str(dist)


def check_input_expression(text: str, variables: Sequence[str],
                           parameter_types: Optional[Mapping[str, Any]] = None) -> _Verdict:
    """Whether the closed form ``text`` (an engine's printout) is a power series at 0 in
    ``variables`` with non-negative coefficients and finite mass: ``(True, "")`` if proven,
    ``(False, reason)`` if refuted, ``(None, reason)`` if this could not be proven."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import MATH_FUNCTIONS, CandidateError, admissible_expression
    parameter_types = dict(parameter_types or {})
    try:
        admissible_expression(text, syntax="pgcl", functions=MATH_FUNCTIONS)
    except CandidateError as exc:
        return None, f"it could not be read back for the check ({exc})"
    local: Dict[str, Any] = {name: sympy.Symbol(name) for name in variables}
    for name in set(_identifiers(text)) - set(variables) - set(MATH_FUNCTIONS):
        if name in _CONSTANTS and name not in parameter_types:
            continue  # printed by an engine, e.g. SymEngine writes exp(1) as E
        local[name] = _parameter_symbol(name, parameter_types.get(name))
    try:
        expr = sympy.parse_expr(text.replace("^", "**"), local_dict=local)
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"it could not be read back for the check ({exc})"
    syms = [local[name] for name in variables]
    return _check(sympy.sympify(expr), syms)


def _identifiers(text: str) -> List[str]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import _IDENTIFIER  # pylint: disable=protected-access
    return _IDENTIFIER.findall(text)


def _parameter_symbol(name: str, typ: Any) -> sympy.Symbol:
    """A parameter as symbol with what its type proves: ``nparam`` is a natural number."""
    # pylint: disable=import-outside-toplevel
    from prodigy.pgcl.ast.types import NatType
    if isinstance(typ, NatType):
        return sympy.Symbol(name, integer=True, nonnegative=True)
    return sympy.Symbol(name)


def _names(syms: Iterable[sympy.Symbol]) -> List[str]:
    return [s.name for s in syms]


def _check(expr: sympy.Expr, syms: List[sympy.Symbol]) -> _Verdict:
    if expr.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return False, "it contains an infinite or undefined value"
    if not syms or expr.is_polynomial(*syms):
        return _check_polynomial(expr, syms)
    if expr.is_rational_function(*syms):
        return _check_rational(expr, syms)
    return _check_structure(expr, syms)


def _parameter_hint(value: sympy.Expr, syms: List[sympy.Symbol]) -> str:
    parameters = sorted(s.name for s in value.free_symbols if s not in syms)
    if not parameters:
        return ""
    return (f" (it depends on the parameter{'s' if len(parameters) > 1 else ''} {', '.join(parameters)}, "
            "whose sign is not known; only nparam parameters are known to be non-negative)")


def _monomial_text(monomial: Sequence[int], syms: List[sympy.Symbol]) -> str:
    term = sympy.Mul(*(s ** k for s, k in zip(syms, monomial)))
    return str(term).replace("**", "^")


def _check_polynomial(expr: sympy.Expr, syms: List[sympy.Symbol]) -> _Verdict:
    if not syms:
        terms: List[Tuple[Tuple[int, ...], sympy.Expr]] = [((), sympy.expand(expr))]
    else:
        terms = sympy.Poly(sympy.expand(expr), *syms).terms()
    for monomial, coefficient in terms:
        if coefficient.is_negative:
            return False, f"the coefficient of {_monomial_text(monomial, syms)} is {coefficient}, which is negative"
        if not coefficient.is_nonnegative:
            return None, (f"the coefficient {coefficient} of {_monomial_text(monomial, syms)} is not proven to be "
                          f"non-negative{_parameter_hint(coefficient, syms)}")
    return _PROVEN


def _check_rational(expr: sympy.Expr, syms: List[sympy.Symbol]) -> _Verdict:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.lowerbounds.egcalc import GFCalculus
    from prodigy.analysis.lowerbounds.subinvariant import nonneg_by_construction, rational_mass, to_product_form
    variables = _names(syms)
    reduced = sympy.cancel(sympy.together(expr))
    _, denominator = sympy.fraction(reduced)
    at_origin = sympy.expand(denominator).subs({s: 0 for s in syms})
    if at_origin.is_zero:
        return False, "it has a pole at 0, so it is not a power series at 0"
    if not at_origin.is_nonzero:
        return None, (f"its denominator {denominator} is not proven to be non-zero at 0"
                      f"{_parameter_hint(at_origin, syms)}")
    form = None
    if reduced.free_symbols <= set(syms):
        try:
            form = to_product_form(reduced, variables)
        except (TypeError, ValueError, sympy.PolynomialError):
            form = None
    if form is not None:
        numerator, product = form
        negative = GFCalculus(variables, product).first_negative(numerator)
        if negative is not None:
            state = ", ".join(f"{v} = {k}" for v, k in zip(variables, negative[0]))
            return False, f"the coefficient at {state} is {negative[1]}, which is negative"
    elif not nonneg_by_construction(reduced, variables):
        hint = _parameter_hint(reduced, syms)
        return None, ("the non-negativity of its coefficients could not be proven" + (hint or
                      " (Prodigy proves it for P / ((c1 - R1) * ... * (ck - Rk)) with constants ci > 0 and "
                      "polynomials P, Ri with non-negative coefficients, and exactly for denominators that are "
                      "products of univariate factors (1 - r*x^p))"))
    try:
        mass = rational_mass(str(reduced), variables)
    except Exception:  # pylint: disable=broad-except
        mass = None
    if mass is None:
        return None, f"its mass could not be determined{_parameter_hint(reduced, syms)}"
    if mass == math.inf:
        return False, ("its mass (the sum of its coefficients) is infinite: the series diverges at 1, it has a pole "
                       "in (0, 1] on the diagonal")
    return _PROVEN


def _check_structure(expr: sympy.Expr, syms: List[sympy.Symbol]) -> _Verdict:
    """Sufficient rules for closed forms that are not rational (see the module documentation)."""
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        rational = [a for a in expr.args if a.is_rational_function(*syms)]
        others = [a for a in expr.args if not a.is_rational_function(*syms)]
        parts = ([expr.func(*rational)] if rational else []) + others
        for part in parts:
            verdict, reason = _check(part, syms)
            if verdict is not True:
                # a refuted summand may cancel against another one
                return None, (reason if verdict is None else
                              f"{_text(part)} is not a distribution ({reason}), so Prodigy cannot prove the sum is")
        return _PROVEN
    if isinstance(expr, sympy.Pow) and expr.exp.is_Integer and expr.exp > 0:
        return _check(expr.base, syms)
    if isinstance(expr, sympy.exp):
        argument = expr.args[0]
        constant = argument.subs({s: 0 for s in syms})
        if not (constant.is_real and constant.is_finite):
            return None, f"{_text(argument)} is not proven to be a power series at 0 with a real constant term"
        verdict, reason = _check(sympy.expand(argument - constant), syms)
        if verdict is not True:
            return None, (f"{_text(expr)} is certified only when {_text(argument)} minus its constant term is a "
                          f"series with non-negative coefficients and finite mass, and here {reason}")
        return _PROVEN
    return None, (f"{_text(expr)} is not proven to be a power series at 0 with non-negative coefficients "
                  "(Prodigy certifies polynomials, rational functions, exp, and their sums and products)")


def _text(expr: sympy.Expr) -> str:
    return str(expr).replace("**", "^")
