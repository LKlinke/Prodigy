"""
The table of sampling distributions (#511, P4b): the probability generating function of each distribution of the
language (``geometric(p)``, ``unif(a, b)``, …) as engine text, with the validation of its parameters.

Every engine builds these distributions from this one table, through
:meth:`~prodigy.distribution.pgf.hooks.PGFHooks.from_expr`: the factory methods
(``SympyPGF.geometric("x", "1/2")``) and the samplings of ``iid``. A parameter without names is checked
against the range of its distribution (e.g. ``0 < p <= 1`` for ``geometric``); a parameter with names (a
program parameter) is not, and the result is a formula that holds for the values in the range, as everywhere
in the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import sympy

from prodigy.distribution.engine_text import parse_engine_text
from prodigy.distribution.pgf.rules.equality import decide_equal
from prodigy.distribution.values import DistributionParam
from prodigy.language.engine_text import engine_format, engine_text_of

__all__ = ["SAMPLING_DISTRIBUTIONS", "SamplingPGF", "is_sampling_distribution", "refutes_unit_mass", "sampling_pgf"]

Values = Dict[str, Optional[sympy.Expr]]
"""The exact values of the parameters by name; None for a parameter with names."""


@dataclass(frozen=True)
class _Family:
    """A distribution of the language: the template of its generating function in ``{var}``, the names of its
    parameters, the check of their values (the message of the first violated range, or None) and whether the
    generating function is a polynomial."""
    template: str
    parameters: Tuple[str, ...]
    check: Callable[[Values], Optional[str]]
    finite: Callable[[Values], bool]


def _first(*conditions: Tuple[Optional[sympy.Expr], Callable[[sympy.Expr], bool], str]) -> Optional[str]:
    """The message of the first condition whose value is known and violates its range."""
    for value, holds, message in conditions:
        if value is not None and not holds(value):
            return message
    return None


def _in_unit_interval(value: sympy.Expr) -> bool:
    return bool(0 <= value <= 1)


def _in_open_unit_interval(value: sympy.Expr) -> bool:
    return bool(0 < value <= 1)


def _non_negative(value: sympy.Expr) -> bool:
    return bool(value >= 0)


def _uniform(values: Values) -> Optional[str]:
    lower, upper = values["a"], values["b"]
    if lower is not None and (lower < 0 or (upper is not None and lower > upper)):
        return "Distribution parameters must satisfy 0 <= a < b < oo"
    return None


def _negative_binomial(values: Values) -> Optional[str]:
    r = values["r"]
    if r is None or not (r.is_integer and r >= 0):
        return "Parameter r of Negative Binomial Distribution must be a natural number, but was {r}"
    return _first((values["p"], _in_open_unit_interval,
                   "Parameter p of Negative Binomial Distribution must be in (0,1], but was {p}"))


# The templates put every parameter in parentheses (`engine_format`), so that it keeps its value.
_FAMILIES: Dict[str, _Family] = {
    "geometric": _Family(
        "({p}) / (1 - (1 - ({p})) * {var})", ("p",),
        lambda v: _first((v["p"], _in_open_unit_interval, "parameter of geom distr must be 0 < p <= 1, was {p}")),
        lambda v: False),
    "unif": _Family(
        "1 / (({b}) - ({a}) + 1) * {var}^({a}) * ({var}^(({b}) - ({a}) + 1) - 1) / ({var} - 1)", ("a", "b"),
        _uniform, lambda v: True),
    "bernoulli": _Family(
        "1 - ({p}) + ({p}) * {var}", ("p",),
        lambda v: _first((v["p"], _in_unit_interval,
                          "Parameter of Bernoulli Distribution must be in [0,1], but was {p}")),
        lambda v: True),
    "poisson": _Family(
        "exp(({p}) * ({var} - 1))", ("p",),
        lambda v: _first((v["p"], _non_negative, "Parameter of Poisson Distribution must be in [0, oo), but was {p}")),
        lambda v: False),
    "logdist": _Family(
        "log(1 - ({p}) * {var}) / log(1 - ({p}))", ("p",),
        lambda v: _first((v["p"], _in_unit_interval,
                          "Parameter of Logarithmic Distribution must be in [0,1], but was {p}")),
        lambda v: False),
    "binomial": _Family(
        "(1 - ({p}) + ({p}) * {var})^({n})", ("n", "p"),
        lambda v: _first((v["p"], _in_unit_interval,
                          "Parameter of Binomial Distribution must be in [0,1], but was {p}"),
                         (v["n"], _non_negative, "Parameter of Binomial Distribution must be in [0,oo), but was {n}")),
        lambda v: True),
    "negative_binomial": _Family(
        "(({p}) / (1 - (1 - ({p})) * {var}))^({r})", ("r", "p"), _negative_binomial, lambda v: v["r"] == 0),
}
_FAMILIES["unif_d"] = _FAMILIES["unif"]

SAMPLING_DISTRIBUTIONS = frozenset(_FAMILIES)
"""The names of the distributions of the language (``unif_d`` is ``unif``)."""


@dataclass(frozen=True)
class SamplingPGF:
    """The generating function of a sampling distribution in one variable, as engine text, and whether it is
    a polynomial (finite support)."""
    text: str
    finite: bool


def is_sampling_distribution(function: str) -> bool:
    """Whether ``function`` is a distribution of the language with an entry in the table."""
    return function in _FAMILIES


def _constant(parameter: DistributionParam) -> Optional[sympy.Expr]:
    """The exact value of a parameter without names, or None if it has names."""
    value = parse_engine_text(engine_text_of(parameter), rational=True)
    return None if value.free_symbols else value


def sampling_pgf(function: str, parameters: Sequence[DistributionParam], variable: str) -> SamplingPGF:
    """
    The generating function of the distribution ``function(parameters…)`` in ``variable``.

    Raises a ``ValueError`` if a parameter without names is outside the range of the distribution (also a
    ``negative_binomial`` whose ``r`` is not a constant natural number), and a ``NotImplementedError`` if
    ``function`` is no distribution of the table.

    .. doctest::

        >>> sampling_pgf("geometric", ["1/2"], "x")
        SamplingPGF(text='(1/2) / (1 - (1 - (1/2)) * x)', finite=False)
    """
    family = _FAMILIES.get(function)
    if family is None:
        raise NotImplementedError(f"Unsupported distribution: {function}")
    if len(parameters) != len(family.parameters):
        raise ValueError(f"The distribution {function} takes {len(family.parameters)} parameters, "
                         f"but {len(parameters)} were given.")
    named = dict(zip(family.parameters, parameters))
    values = {name: _constant(parameter) for name, parameter in named.items()}
    message = family.check(values)
    if message is not None:
        raise ValueError(message.format(**{name: engine_text_of(value) for name, value in named.items()}))
    return SamplingPGF(engine_format(family.template, var=variable, **named), family.finite(values))


def refutes_unit_mass(text: str, variable: str) -> bool:
    """
    Whether the generating function ``text`` in ``variable`` (given to ``iid``) is proven to have a mass
    ``G(1)`` other than 1, so that it is no probability generating function: ``G(1)`` (its limit at 1 from
    below where ``G`` is not defined at 1) is a constant and proven different from 1. A mass that depends on
    parameters is no refutation: the result holds for the parameter values for which ``G`` is a probability
    generating function (the analysis warns, ``sample_handler``).

    .. doctest::

        >>> refutes_unit_mass("y/2", "y"), refutes_unit_mass("1/2 + y/2", "y"), refutes_unit_mass("p*y + q", "y")
        (True, False, False)
    """
    try:
        function = parse_engine_text(text, rational=True)
        symbol = next((s for s in function.free_symbols if s.name == variable), sympy.Symbol(variable))
        mass = function.subs(symbol, 1)
        if mass.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
            mass = sympy.limit(function, symbol, 1, "-")
    except Exception:  # pylint: disable=broad-except  # not decided: no refutation
        return False
    if mass.free_symbols or mass.has(sympy.Limit):
        return False
    return decide_equal(mass, sympy.Integer(1)) is False
