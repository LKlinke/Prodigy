"""
SymPy expressions of generating functions for the SymPy engine (#511): parsing, cancellation, values and limits at
1, Taylor coefficients and the terms of polynomials.
"""
from __future__ import annotations

import functools
from fractions import Fraction
from typing import Callable, Dict, FrozenSet, Iterator, List, Tuple, Union, get_args

import sympy

from prodigy.distribution.engine_text import parse_engine_text
from prodigy.language import Binop, BinopExpr, Expr, FunctionCallExpr, NatLitExpr, RealLitExpr, VarExpr
from prodigy.language.engine_text import engine_text_of
from prodigy.distribution.pgf.rules.equality import decide_equal
from prodigy.distribution.pgf.rules.rational_series import rational_taylor_coefficients
from prodigy.distribution.pgf.rules.value_at_one import _UNDEFINED_VALUES, _cancel, _continuous_at_one


def _has_radical(function: sympy.Expr) -> bool:
    """Whether ``function`` has a power with a non-integer exponent, e.g. a square root, or an exponential
    (``exp(7*x - 7)`` of a Poisson distribution)."""
    return any(not power.exp.is_Integer for power in function.atoms(sympy.Pow)) or function.has(sympy.exp)


@functools.lru_cache(maxsize=1024)
def _is_finite_function(function: sympy.Expr, variables: FrozenSet[sympy.Symbol]) -> bool:
    """The finiteness heuristic of the constructor: ``function`` is a polynomial in ``variables``
    once cancelled. Cancelling is expensive on large closed forms, and the analysis builds many
    distributions with the same function (every ``copy`` repeats the heuristic for a flag that is False),
    so the result is cached; it depends only on the arguments.

    The function is cancelled to ``f/g`` in lowest terms (:func:`_cancel`). If ``g`` contains a variable,
    ``g`` does not divide ``f``, so the function is no polynomial; else it is ``f/g`` with a constant ``g``,
    a polynomial iff ``f`` is one (this is what ratsimp, used before, decided after a second, expensive
    cancellation).

    The subexpressions that are not rational (``I``, ``sqrt(3)``, ``exp(1/2)``, the constants of a
    roots-of-unity filter) are replaced by fresh symbols first, so that the cancellation works over the
    rationals instead of an algebraic or transcendental extension, which can take minutes (Prodigy#472).
    A polynomial in the variables with symbolic constants stays one when the constants are
    substituted back, so the answer True is exact; the relations between the constants (``I**2 = -1``) are
    not used, so a function that is a polynomial only because of them may be reported as not finite, the
    conservative answer of this heuristic.

    A function with an exponential of a variable (Poisson distributions and their residue classes) is not
    cancelled: cancelling would treat each exponential as a further generator, slowly (many of them after a
    roots-of-unity filter). Its exponentials are split into products (``exp(5*x - 5)`` into
    ``exp(5*x)*exp(-5)``), so that equal factors cancel in products and quotients (``x**3*exp(5*x - 5)/exp(5*x -
    5)``); it is finite if that leaves a polynomial. Otherwise the answer is the conservative one (exponentials
    that would cancel only in sums, e.g. ``exp(x)*(1 + y) - exp(x)*y``, are not found).
    """
    if function.is_polynomial(*variables):
        return True
    if any(power.args[0].free_symbols & variables for power in function.atoms(sympy.exp)):
        # exp(5*x - 5) is split into exp(5*x)*exp(-5), so that equal exponentials cancel in products and quotients
        split = sympy.expand_power_exp(function)
        return bool(split.is_polynomial(*variables))
    numerator, denominator = _cancel(function).as_numer_denom()
    if denominator.free_symbols & variables:
        return False
    return bool(numerator.is_polynomial(*variables))


@functools.lru_cache(maxsize=1024)
def _limit_at_one(function: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr:
    """The limit of ``function`` as ``variable`` tends to 1 from below, i.e. the variable marginalised
    out of a closed form. The analysis marginalises the same function several times, so the result is
    cached; it depends only on the arguments.

    SymPy's ``limit`` returns small expressions, and later steps rely on that (substituting into the
    uncancelled function let them grow, #272). But it may hang on unsimplified forms with a removable
    singularity at 1, e.g. after ``unif`` (#274), also when another variable is marginalised. For such
    forms (:func:`_denominator_vanishes_at_ones`) the function is cancelled. If that removed every such
    denominator (the singularity was removable) and the cancelled form is provably continuous at 1
    (:func:`_continuous_at_one`) and finite there, its value is the limit. Otherwise (e.g. the genuine
    pole of ``1/(1 - c)``, whose form the later positivity checks expect) ``limit`` is used as before.
    """
    if variable not in function.free_symbols:
        return function
    if not _denominator_vanishes_at_ones(function):
        return function.limit(variable, 1, "-")
    if _regular_in_at_one(function, variable) and _continuous_at_one(function, variable):
        value = function.subs(variable, 1)
        if not value.has(*_UNDEFINED_VALUES):
            return value
    cancelled = _cancel(function)
    if not _denominator_vanishes_at_ones(cancelled) and _continuous_at_one(cancelled, variable):
        value = cancelled.subs(variable, 1)
        if not value.has(*_UNDEFINED_VALUES):
            return value
    return function.limit(variable, 1, "-")


def _regular_in_at_one(function: sympy.Expr, variable: sympy.Symbol) -> bool:
    """True only if every denominator of `function` that depends on `variable` is a product whose factors
    with `variable` are provably nonzero when every symbol is 1.

    The other factors do not depend on `variable`: they divide the function for every value of `variable`, so
    they cannot make it singular in `variable` (where they are zero, the function is undefined for every
    value of `variable`). This is the class of closed forms that are singular at 1 only in other variables,
    e.g. a normalised `unif` or observation in another variable, multiplied by exponentials of the
    marginalised one. For them the limit in `variable` is the substitution (with
    :func:`_continuous_at_one`), without cancelling the whole function first, which takes minutes on sums of
    many exponential constants (#486)."""
    ones: Dict[sympy.Symbol, int] = {}
    for power in function.atoms(sympy.Pow):
        base, exponent = power.args
        if not exponent.is_negative or variable not in base.free_symbols:
            continue
        for factor in sympy.Mul.make_args(sympy.factor_terms(base)):
            if variable not in factor.free_symbols:
                continue
            for symbol in factor.free_symbols:
                ones.setdefault(symbol, 1)
            if factor.subs(ones).is_zero is not False:
                return False
    return True


def _denominator_vanishes_at_ones(function: sympy.Expr) -> bool:
    """Whether some negative power in `function` has a base that is 0 when every symbol is 1, like the
    `(z - 1)` of an unsimplified `unif`. Only then does `_limit_at_one` cancel first; this is a choice of the
    method, not a soundness condition."""
    for power in function.atoms(sympy.Pow):
        base, exponent = power.args
        if exponent.is_negative and base.free_symbols \
                and base.subs({symbol: 1 for symbol in base.free_symbols}) == 0:
            return True
    return False


Term = Tuple[sympy.Expr, sympy.Expr]
"""A term of a power series: its coefficient and its monomial."""

SympyInput = Union[str, int, float, Fraction, Expr, sympy.Basic]
"""What :func:`_parse_to_sympy` reads: text, a number, a program expression or a SymPy object (returned as it is)."""


def _term_generator(function: sympy.Poly) -> Iterator[Term]:
    assert isinstance(function,
                      sympy.Poly), "Terms can only be generated for finite GF"
    # `terms()` lists the monomials in decreasing lex order; the states come in increasing order. One pass,
    # instead of repeatedly subtracting the smallest term from the polynomial (quadratic in the number of terms).
    for monomial, coefficient in reversed(function.terms()):
        if coefficient == 0:
            continue
        yield coefficient, sympy.Mul(*(gen ** power for gen, power in zip(function.gens, monomial, strict=True)))


def _parse_to_sympy(expr: SympyInput, real: bool = True, **kwargs: bool) -> sympy.Expr:
    """
    Parses something (`float`s, `bool`s, `Fraction`s, program `Expr`s, anything that can be converted to a valid
    string) to a sympy expression. Can convert a program expression faster than passing it as a string to sympy.
    Also will correctly determine if something is a symbol or a function (e.g., if we have a variable called `exp` or
    `log`). All symbols are given the assumptions that they are real unless specified otherwise.
    """

    # first we cover some cases that we can handle faster than by parsing a string
    if getattr(expr, '__sympy__', None):
        return expr
    if isinstance(expr, (float, bool, Fraction)):
        return sympy.S(expr, rational=True, **kwargs)

    s: sympy.Expr

    def get_function(name: str) -> Callable[..., sympy.Expr]:
        functions = {'sum': sum}
        if name in functions:
            return functions[name]
        return getattr(sympy, name)

    if isinstance(expr, get_args(Expr)):

        def probably_to_sympy(prob_expr: Expr) -> sympy.Expr:
            if isinstance(prob_expr, VarExpr):
                return _sympy_symbol(prob_expr.var, real, **kwargs)
            if isinstance(prob_expr, NatLitExpr):
                return sympy.Integer(prob_expr.value)
            if isinstance(prob_expr, RealLitExpr):
                frac = prob_expr.to_fraction()
                return sympy.Rational(frac.numerator, frac.denominator)
            if isinstance(prob_expr, FunctionCallExpr):
                if len(prob_expr.params[1]) != 0:
                    raise ValueError(
                        f'Cannot parse function with named parameters: {prob_expr}'
                    )
                return get_function(prob_expr.function)(
                    *map(probably_to_sympy, prob_expr.params[0]))
            if isinstance(prob_expr, BinopExpr):
                op = prob_expr.operator
                if op == Binop.DIVIDE:
                    return probably_to_sympy(
                        prob_expr.lhs) / probably_to_sympy(prob_expr.rhs)
                if op == Binop.MINUS:
                    return probably_to_sympy(
                        prob_expr.lhs) - probably_to_sympy(prob_expr.rhs)
                if op == Binop.TIMES:
                    return probably_to_sympy(
                        prob_expr.lhs) * probably_to_sympy(prob_expr.rhs)
                if op == Binop.POWER:
                    return probably_to_sympy(prob_expr.lhs) ** probably_to_sympy(
                        prob_expr.rhs)
                if op == Binop.MODULO:
                    return probably_to_sympy(
                        prob_expr.lhs) % probably_to_sympy(prob_expr.rhs)
                if op == Binop.PLUS:
                    return probably_to_sympy(
                        prob_expr.lhs) + probably_to_sympy(prob_expr.rhs)
                raise ValueError(f'Unsupported operand: {op}')
            raise ValueError(f'Unsupported expression type: {prob_expr}')

        s = probably_to_sympy(expr)

    else:
        s = parse_engine_text(engine_text_of(expr), rational=True,
                              symbol=lambda name: _sympy_symbol(name, real, **kwargs))
    return s


def _sympy_symbol(name: str, real: bool = True, **kwargs: bool) -> sympy.Symbol:
    """
    Creates a real sympy Symbol.
    
    Note that `sympy.Symbol('x') == sympy.Symbol('x', real=True)` evaluates to false, as variables
    with the same name but different domains are not considered equal by sympy.
    """
    return sympy.Symbol(name, real=real, **kwargs)


def _polynomial_coefficients(function: sympy.Expr, variable: sympy.Symbol, count: int) -> List[sympy.Expr]:
    """
    The coefficients of `variable`^0, ..., `variable`^(count-1) of the polynomial `function` (other symbols are
    constants), read off its expanded terms. Unlike a `Poly`, this never builds the dense list of all
    coefficients, which is huge for sparse polynomials of high degree such as the x**(2**k) of a doubled
    variable (#224).
    """
    coefficients = [sympy.Integer(0)] * count
    for term in sympy.Add.make_args(sympy.expand(function)):
        coefficient, power = term.as_independent(variable, as_Add=False)
        exponent = power.as_coeff_exponent(variable)[1] if power != 1 else 0
        if exponent < count:
            coefficients[int(exponent)] += coefficient
    return coefficients


def _taylor_coefficient_by_limit(function: sympy.Expr, variable: sympy.Symbol, index: int) -> sympy.Expr:
    """The coefficient of `variable`^`index` in the power series of `function`, via derivatives and a limit."""
    return (function.diff(variable, index) / sympy.factorial(index)).limit(variable, 0, '-')


def _taylor_coefficient(function: sympy.Expr, variable: sympy.Symbol, index: int) -> sympy.Expr:
    """
    The coefficient of `variable`^`index` in the power series of `function`: exactly from the recursion for
    rational functions (see `rational_series`), otherwise via derivatives and a limit.
    """
    coefficients = rational_taylor_coefficients(function, variable, index + 1)
    if coefficients is None:
        return _taylor_coefficient_by_limit(function, variable, index)
    return coefficients[index] if index < len(coefficients) else sympy.Integer(0)


def provably_equal(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
    """Decides ``lhs == rhs`` such that True is backed by a symbolic proof (see :func:`decide_equal`)."""
    return decide_equal(lhs, rhs) is True
