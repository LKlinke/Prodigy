"""
Proofs that a closed form is a finite measure: a power series at 0 with non-negative coefficients and finite mass.

The check of input distributions (:mod:`prodigy.analysis.input_distribution`, #296), the check of the arguments of
``sample_pgf`` and ``iid`` (#424, also in the type checker) and the convergence of marginals of distributions
(:meth:`~prodigy.distribution.distribution.Distribution.marginal`) share these rules. A verdict is ``(True, "")``
if proven, ``(False, reason)`` if refuted and ``(None, reason)`` if not proven. The proofs, all exact:

* polynomials: every coefficient is checked; with parameters, SymPy must prove its sign from the
  parameter's type (``nparam`` is a natural number, ``rparam`` has no known sign);
* rational functions: the reduced denominator does not vanish at 0 (power series); the
  coefficients are non-negative by the exact finite positivity check for product denominators
  (:func:`~prodigy.distribution.product_form.to_product_form`, which also refutes) or by
  the construction ``P / prod (c - R)``
  (:func:`~prodigy.distribution.coefficient_signs.nonneg_by_construction`); the mass is finite
  iff the diagonal has no pole in ``(0, 1]`` (Pringsheim,
  :func:`~prodigy.distribution.product_form.rational_mass`); a univariate rational function with rational
  coefficients and a unique dominant simple pole is decided exactly
  (:mod:`~prodigy.distribution.dominant_pole`, #431);
* other closed forms are taken apart: sums, products and positive integer powers of certified
  series are certified, and so is ``exp(G)`` when ``G - G(0)`` is (``exp(G(0)) > 0``, and
  ``exp(H) = sum H^n / n!``); e.g. the Poisson ``exp(3*(x - 1))``;
* if none of these proves or refutes it, two changes that keep the coefficients are tried
  (:mod:`~prodigy.distribution.series_transforms`, #431): ``f = G / x^k`` where the coefficients of ``G``
  below degree ``k`` vanish, and ``f = G(x^m)`` for a variable that occurs only in powers ``x^(m*j)``;
  ``f`` is then certified (or refuted) with ``G``. Anything else is not proven.

The text of a closed form is checked with the allowlist of
:func:`~prodigy.pgcl.expression_allowlist.admissible_expression` before SymPy reads it.
"""
from __future__ import annotations

import functools
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sympy

from prodigy.distribution.coefficient_signs import nonneg_by_construction
from prodigy.distribution.dominant_pole import dominant_pole_verdict
from prodigy.distribution.product_form import ProductFormSeries, rational_mass, to_product_form
from prodigy.distribution.series_transforms import low_coefficients_vanish, monomial_substitution, shift
from prodigy.pgcl.ast.types import NatType, Type
from prodigy.pgcl.expression_allowlist import IDENTIFIER, MATH_FUNCTIONS, InadmissibleExpression, admissible_expression

Verdict = Tuple[Optional[bool], str]
"""True (proven), False (refuted) or None (not proven), with the reason if not proven."""
_PROVEN: Verdict = (True, "")

_CONSTANTS = frozenset({"E", "pi", "I"})


def check_input_expression(text: str, variables: Sequence[str],
                           parameter_types: Optional[Mapping[str, Optional[Type]]] = None, *,
                           probability: bool = False) -> Verdict:
    """Whether the closed form ``text`` (an engine's printout) is a power series at 0 in
    ``variables`` with non-negative coefficients and finite mass: ``(True, "")`` if proven,
    ``(False, reason)`` if refuted, ``(None, reason)`` if this could not be proven. With
    ``probability``, the mass must also be 1 (see :func:`check_probability_generating_function`)."""
    parameter_types = dict(parameter_types or {})
    try:
        admissible_expression(text, functions=MATH_FUNCTIONS)
    except InadmissibleExpression as exc:
        return None, f"it could not be read back for the check ({exc})"
    local: Dict[str, sympy.Symbol] = {name: sympy.Symbol(name) for name in variables}
    for name in set(_identifiers(text)) - set(variables) - set(MATH_FUNCTIONS):
        if name in _CONSTANTS and name not in parameter_types:
            continue  # printed by an engine, e.g. exp(1) as E
        local[name] = _parameter_symbol(name, parameter_types.get(name))
    try:
        expr = sympy.parse_expr(text.replace("^", "**"), local_dict=local)
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"it could not be read back for the check ({exc})"
    syms = [local[name] for name in variables]
    expr = sympy.sympify(expr)
    verdict = _check(expr, syms)
    if not probability:
        return verdict
    return _check_probability_mass(expr, syms, verdict)


@functools.lru_cache(maxsize=512)
def check_probability_generating_function(text: str, variables: Tuple[str, ...],
                                          natural_parameters: Tuple[str, ...] = ()) -> Verdict:
    """Whether the closed form ``text`` is a probability generating function in ``variables``: a power
    series at 0 with non-negative coefficients (as for :func:`check_input_expression`) and mass exactly 1.
    This is the condition on the distribution argument ``G`` of ``sample_pgf(G, x)`` and ``iid(G, n)``
    (#424). A mass different from 1 refutes it even if the non-negativity is not proven: the value at 1
    of a probability generating function is 1 (Abel's theorem). Every other symbol is a parameter;
    ``natural_parameters`` are those of type ``nat``. Cached, since a sampling instruction may be executed
    many times (e.g. in an unrolled loop)."""
    types = {name: NatType(None) for name in natural_parameters}
    return check_input_expression(text, variables, types, probability=True)


def _check_probability_mass(expr: sympy.Expr, syms: List[sympy.Symbol], verdict: Verdict) -> Verdict:
    """``verdict`` on the coefficients of ``expr``, combined with the condition that its mass (its value at 1,
    the limit from below on the diagonal) is 1."""
    if verdict[0] is False:
        return verdict
    mass = _mass(expr, syms)
    if mass is None:
        return None, verdict[1] or "its mass (its value at 1) could not be determined"
    if mass.is_number and not mass.is_finite:
        return False, "its mass (its value at 1) is infinite, not 1"
    try:
        difference = sympy.simplify(mass - 1)
    except Exception:  # pylint: disable=broad-except
        difference = None
    if difference is not None and difference.is_zero:
        return verdict
    if mass.is_number and difference is not None and difference.is_zero is False:
        return False, f"its mass (its value at 1) is {_text(mass)}, not 1"
    if verdict[1]:
        return None, f"{verdict[1]}; its mass (its value at 1) is {_text(mass)}, which is not proven to be 1"
    return None, f"its mass (its value at 1) is {_text(mass)}, which is not proven to be 1{_parameter_hint(mass, syms)}"


def _mass(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Optional[sympy.Expr]:
    """The value of ``expr`` at 1 in ``syms``, as the limit from below on the diagonal if the value itself is
    not defined (e.g. a removable singularity); ``None`` if it cannot be computed."""
    if not syms:
        return expr
    value = expr.subs({s: 1 for s in syms})
    if not value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return value
    t = sympy.Dummy("t")
    try:
        return sympy.limit(expr.subs({s: t for s in syms}), t, 1, "-")
    except Exception:  # pylint: disable=broad-except
        return None


def _identifiers(text: str) -> List[str]:
    return IDENTIFIER.findall(text)


def _parameter_symbol(name: str, typ: Optional[Type]) -> sympy.Symbol:
    """A parameter as symbol with what its type proves: ``nparam`` is a natural number."""
    if isinstance(typ, NatType):
        return sympy.Symbol(name, integer=True, nonnegative=True)
    return sympy.Symbol(name)


def _names(syms: Iterable[sympy.Symbol]) -> List[str]:
    return [s.name for s in syms]


def check_series(expr: sympy.Expr, variables: Sequence[str]) -> Verdict:
    """The verdict of :func:`check_input_expression` (non-negative coefficients and finite mass) for an
    expression that is already a SymPy expression; its other symbols are parameters with the assumptions they
    carry. Used by the order of :mod:`~prodigy.distribution.ordering`, which only uses ``True``."""
    by_name = {s.name: s for s in sympy.S(expr).free_symbols}
    syms = [by_name.get(name, sympy.Symbol(name)) for name in variables]
    return _check(sympy.S(expr), syms)


def _check(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Verdict:
    if expr.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return False, "it contains an infinite or undefined value"
    if not syms or expr.is_polynomial(*syms):
        return _check_polynomial(expr, syms)
    if expr.is_rational_function(*syms):
        verdict = _check_rational(expr, syms)
    else:
        verdict = _check_structure(expr, syms)
    if verdict[0] is not None:
        return verdict
    for rule in (_check_shift, _check_monomial_substitution):
        transformed = rule(expr, syms)
        if transformed is not None and transformed[0] is not None:
            return transformed
    return verdict


def _check_shift(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Optional[Verdict]:
    """``expr = G / x^k`` whose ``G`` has no coefficients below degree ``k``: the verdict of ``G``
    (:func:`~prodigy.distribution.series_transforms.shift`). None if the rule does not apply."""
    for symbol in syms:
        shifted = shift(expr, symbol)
        if shifted is None:
            continue
        inner, k = shifted
        if not low_coefficients_vanish(inner, symbol, k):
            continue
        verdict, reason = _check(inner, syms)
        if verdict is True:
            return _PROVEN
        if verdict is False:
            # the coefficients of expr are those of `inner` from degree k on, shifted down by k
            return False, f"{reason} (for {_text(inner)}, which is {_text(expr)} times {symbol}^{k})"
    return None


def _check_monomial_substitution(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Optional[Verdict]:
    """``expr = G(x^m)``: the verdict of ``G``, whose coefficients are those of ``expr``
    (:func:`~prodigy.distribution.series_transforms.monomial_substitution`). None if the rule does not
    apply."""
    inner = expr
    substitutions = []
    for symbol in syms:
        substituted = monomial_substitution(inner, symbol)
        if substituted is not None:
            inner = substituted[0]
            substitutions.append(f"{symbol} for {symbol}^{substituted[1]}")
    if not substitutions:
        return None
    verdict, reason = _check(inner, syms)
    if verdict is True:
        return _PROVEN
    if verdict is False:
        return False, f"{reason} (in {_text(inner)}, which is {_text(expr)} with {', '.join(substitutions)})"
    return None


def _parameter_hint(value: sympy.Expr, syms: List[sympy.Symbol]) -> str:
    parameters = sorted(s.name for s in value.free_symbols if s not in syms)
    if not parameters:
        return ""
    return (f" (it depends on the parameter{'s' if len(parameters) > 1 else ''} {', '.join(parameters)}, "
            "whose sign is not known; only nparam parameters are known to be non-negative)")


def _monomial_text(monomial: Sequence[int], syms: List[sympy.Symbol]) -> str:
    term = sympy.Mul(*(s ** k for s, k in zip(syms, monomial)))
    return str(term).replace("**", "^")


def _check_polynomial(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Verdict:
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


def _check_rational(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Verdict:
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
        negative = ProductFormSeries(variables, product).first_negative(numerator)
        if negative is not None:
            state = ", ".join(f"{v} = {k}" for v, k in zip(variables, negative[0]))
            return False, f"the coefficient at {state} is {negative[1]}, which is negative"
    elif not nonneg_by_construction(reduced, variables):
        pole = _check_dominant_pole(reduced, syms)
        if pole[0] is False:
            return pole
        if pole[0] is None:
            hint = _parameter_hint(reduced, syms)
            return None, ("the non-negativity of its coefficients could not be proven" + (hint or
                          " (Prodigy proves it for P / ((c1 - R1) * ... * (ck - Rk)) with constants ci > 0 and "
                          "polynomials P, Ri with non-negative coefficients, exactly for denominators that are "
                          "products of univariate factors (1 - r*x^p), and exactly for univariate functions "
                          "with a unique dominant simple pole)"))
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


def _check_dominant_pole(reduced: sympy.Expr, syms: List[sympy.Symbol]) -> Verdict:
    """The decision of :mod:`~prodigy.distribution.dominant_pole` on the coefficients of a univariate rational
    function with rational coefficients (its mass is checked separately); ``(None, reason)`` outside this
    class."""
    used = [s for s in syms if reduced.has(s)]
    if len(used) != 1 or reduced.free_symbols != set(used):
        return None, "it is not a rational function in one variable with rational coefficients"
    verdict = dominant_pole_verdict(reduced, used[0])
    if verdict.holds is True:
        return _PROVEN
    if verdict.holds is False and verdict.witness is not None:
        index, value = verdict.witness
        position = [0] * len(syms)
        position[syms.index(used[0])] = index
        return False, f"the coefficient of {_monomial_text(position, syms)} is {value}, which is negative"
    return None, verdict.reason


def _check_structure(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Verdict:
    """Sufficient rules for closed forms that are not rational (see the module documentation)."""
    power_verdict = _check_fractional_power(expr, syms)
    if power_verdict is not None:
        return power_verdict
    if isinstance(expr, sympy.Mul):
        expr = _normalise_factor_signs(expr, syms)
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


def _normalise_factor_signs(expr: sympy.Mul, syms: List[sympy.Symbol]) -> sympy.Expr:
    """
    ``expr``, a product, with the signs of its factors normalised (an equal expression). An engine may print
    ``(1 - sqrt(1 - x)) / (2 - y)`` as ``(-2 + y)^(-1) * (-1 + sqrt(1 - x))``: the factors are checked one by
    one, and ``1 / (y - 2)`` has negative coefficients, although two negated factors give the same product
    (#428). A rational factor whose value at 0 is negative is negated, and the sign is paired with a negation
    of the first factor that is not rational; if no such factor exists, ``expr`` stays as it is.
    """
    origin = {s: 0 for s in syms}
    sign = 1
    factors: List[sympy.Expr] = []
    for factor in expr.args:
        base, exponent = factor.as_base_exp()
        value = base.subs(origin) if base.is_rational_function(*syms) else None
        if value is not None and value.is_number and value.is_negative and exponent.is_Integer:
            base = sympy.expand(-base)
            sign *= (-1) ** int(exponent)
            factor = base ** exponent
        factors.append(factor)
    if sign == 1:
        return sympy.Mul(*factors)
    for index, factor in enumerate(factors):
        if not factor.is_rational_function(*syms):
            factors[index] = sympy.expand(-factor)
            return sympy.Mul(*factors)
    return expr


def _check_fractional_power(expr: sympy.Expr, syms: List[sympy.Symbol]) -> Optional[Verdict]:
    """``A + B * (1 - r*x)^a`` with numbers A, B, a rational in (0, 1) and r in (0, 1]: the series of
    ``(1 - r*x)^a`` is ``1 - sum_{n>=1} |binom(a, n)| r^n x^n``, so the constant coefficient of the sum is
    ``A + B`` (the series value at 0, not that of a single summand) and the others are ``-B |binom(a, n)| r^n``.
    The coefficients are non-negative iff ``B <= 0`` and ``A + B >= 0``; the mass is ``A + B (1 - r)^a``."""
    if not isinstance(expr, sympy.Add) or len(expr.args) != 2:
        return None
    constants = [a for a in expr.args if a.is_number]
    powers = [a for a in expr.args if not a.is_number]
    if len(constants) != 1 or len(powers) != 1:
        return None
    offset = constants[0]
    coefficient, power = powers[0].as_independent(*syms, as_Add=False)
    if not (isinstance(power, sympy.Pow) and power.exp.is_Rational and 0 < power.exp < 1 and coefficient.is_number):
        return None
    base = sympy.expand(power.base)
    used = [s for s in syms if base.has(s)]
    if len(used) != 1 or not base.is_polynomial(used[0]) or sympy.degree(base, used[0]) != 1:
        return None
    ratio = -base.coeff(used[0], 1)
    if base.subs(used[0], 0) != 1 or not (ratio.is_number and ratio.is_positive and (ratio - 1).is_nonpositive):
        return None
    if coefficient.is_positive:
        return False, (f"the coefficient of {used[0]} is {sympy.simplify(-coefficient * power.exp * ratio)}, "
                       "which is negative")
    if not (coefficient.is_nonpositive and (offset + coefficient).is_nonnegative):
        return False, f"the coefficient of 1 is {sympy.simplify(offset + coefficient)}, which is negative"
    return _PROVEN


def _text(expr: sympy.Expr) -> str:
    return str(expr).replace("**", "^")
