"""
---------------------------------------
Exploring expected visiting times (EVT)
---------------------------------------

The strategy ``evt_iterate`` iterates the EVT operator Ψ_G(H) = G + ⟦body⟧(⟨H⟩_guard) from 0. Its iterates
H_N are not output distributions: the coefficient of a state σ in H_N is the expected number of times the loop
head is reached in σ within N iterations. Watching them grow with N helps to guess an EVT invariant (the limit
H = lim H_N), which ``evt_invariant`` then checks exactly (#324).

This module turns some iterates into data for front ends (closed forms, a coefficient table per state) and
proposes a candidate. The candidate is only a guess: nothing here is a result, and the loop's outcome does not
depend on it.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import sympy

if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution import Distribution

MAX_SAMPLES = 12
"""Iterates kept for the exploration (spread evenly over 1 … N, the last one always)."""

MAX_STATES = 30
"""Rows of the coefficient table."""

MAX_TERMS = 2000
"""Terms read from an iterate (an iterate of a loop whose input has infinite support has infinitely many)."""

MAX_CLOSED_FORM = 3000
"""Characters of a closed form that are kept."""

MAX_GUESS_SLICES = 8
"""Slices (values of the other variables) for which a univariate rational function is guessed."""

State = Tuple[int, ...]


def sample_points(iterations: int, limit: int = MAX_SAMPLES) -> List[int]:
    """The values of N whose iterates are kept: at most ``limit``, spread evenly over 1 … ``iterations``."""
    if iterations <= 0:
        return []
    if iterations <= limit:
        return list(range(1, iterations + 1))
    points = {1 + round(i * (iterations - 1) / (limit - 1)) for i in range(limit)}
    return sorted(points)


def coefficients(dist: "Distribution", variables: Sequence[str], limit: int = MAX_TERMS
                 ) -> Tuple[Dict[State, str], bool]:
    """The coefficients of ``dist`` by state (values of ``variables`` in this order), and whether there were
    more than ``limit`` terms."""
    table: Dict[State, str] = {}
    complete = True
    for index, (prob, state) in enumerate(dist):
        if index >= limit:
            complete = False
            break
        if str(prob) in ("0", ""):
            continue
        key = tuple(int(state[v]) if v in state else 0 for v in variables)
        table[key] = str(prob)
    return table, not complete


def _parse(text: str) -> Optional[sympy.Expr]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import sym
    try:
        return sym(text)
    except Exception:  # pylint: disable=broad-except
        return None


def _number(text: str) -> Optional[sympy.Expr]:
    value = _parse(text)
    return value if value is not None and value.is_Rational else None


def _float(text: str) -> Optional[float]:
    value = _parse(text)
    if value is None or not value.is_number:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extrapolate(last: Sequence[Dict[State, str]]) -> Dict[State, sympy.Expr]:
    """Estimates of lim H_N per state from the coefficients of three consecutive iterates (Aitken's Δ²
    process, which is exact for coefficients that converge like a + b·q^N); a coefficient that did not
    change is kept. Only rational coefficients; an empty dict if there are others."""
    first, second, third = ({state: _number(text) for state, text in table.items()} for table in last)
    if any(v is None for table in (first, second, third) for v in table.values()):
        return {}
    zero = sympy.S(0)
    limits: Dict[State, sympy.Expr] = {}
    for state, x2 in third.items():
        x0, x1 = first.get(state, zero), second.get(state, zero)
        step, curvature = x2 - x1, (x2 - x1) - (x1 - x0)  # type: ignore[operator]
        if step == 0 or curvature == 0 or x0 == 0:
            limits[state] = x2  # type: ignore[assignment]
        else:
            limits[state] = x2 - step ** 2 / curvature  # type: ignore[operator]
    return limits


def rational_guess(table: Dict[State, Any], variables: Sequence[str]) -> Optional[sympy.Expr]:
    """A rational generating function that matches the coefficients in ``table``, or None.

    For one variable v at a time, the states are split into slices by the values of the other variables;
    for every slice, SymPy guesses a rational function of v from the coefficients along v (a short slice is
    taken as the polynomial it is). The guess is the sum of the slices, each times the monomial of its other
    values. Only rational coefficients are used. This is a heuristic; the guess has to be checked."""
    if not table or not variables:
        return None
    values = {state: value if isinstance(value, sympy.Basic) else _number(value) for state, value in table.items()}
    if any(value is None for value in values.values()):
        return None
    symbols = [sympy.Symbol(v) for v in variables]
    # variables with long slices first: they carry the structure
    order = sorted(range(len(variables)), key=lambda i: -max(state[i] for state in table))
    for axis in order:
        slices: Dict[State, Dict[int, sympy.Expr]] = {}
        for state, value in values.items():
            rest = state[:axis] + state[axis + 1:]
            slices.setdefault(rest, {})[state[axis]] = value  # type: ignore[assignment]
        if len(slices) > MAX_GUESS_SLICES:
            continue
        guess = _guess_slices(slices, axis, symbols)
        if guess is not None:
            return guess
    return None


def _guess_slices(slices: Dict[State, Dict[int, sympy.Expr]], axis: int, symbols: List[sympy.Symbol]
                  ) -> Optional[sympy.Expr]:
    # pylint: disable=import-outside-toplevel
    from sympy.concrete.guess import guess_generating_function_rational
    variable = symbols[axis]
    others = symbols[:axis] + symbols[axis + 1:]
    total = sympy.S(0)
    for rest, entries in slices.items():
        sequence = [entries.get(k, sympy.S(0)) for k in range(max(entries) + 1)]
        if len(sequence) > 6:
            # the coefficients near the frontier of H_N have not converged yet
            sequence = sequence[:max(6, -(-2 * len(sequence) // 3))]
        if len(sequence) <= 3:
            part = sum((c * variable ** k for k, c in enumerate(sequence)), sympy.S(0))
        else:
            part = guess_generating_function_rational(sequence, X=variable)
            if part is None:
                return None
        monomial = sympy.Mul(*(s ** e for s, e in zip(others, rest)))
        total += monomial * part
    return sympy.factor(sympy.together(total))


def _closed_form(dist: "Distribution") -> Tuple[str, bool]:
    text = str(dist)
    return (text[:MAX_CLOSED_FORM], True) if len(text) > MAX_CLOSED_FORM else (text, False)


def exploration(iterates: Sequence[Tuple[int, "Distribution"]], variables: Sequence[str], syntax: str = "pgcl",
                last: Sequence["Distribution"] = ()) -> Dict[str, Any]:
    """Data for front ends: the kept iterates (N, closed form), the coefficients of the first states per
    iterate, and a candidate EVT invariant (a rational fit, or None). ``last`` are the last three consecutive
    iterates, for extrapolating the coefficients to N → ∞."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.ai_candidates import to_syntax
    variables = [str(v) for v in variables]
    tables = []
    truncated_terms = False
    for _, dist in iterates:
        table, more = coefficients(dist, variables)
        tables.append(table)
        truncated_terms = truncated_terms or more
    states = sorted(set(itertools.chain.from_iterable(tables)), key=lambda s: (sum(s), s))
    rows = []
    for state in states[:MAX_STATES]:
        exact = [table.get(state, "0") for table in tables]
        rows.append({"state": dict(zip(variables, state)),
                     "values": [to_syntax(v, syntax) for v in exact],
                     "floats": [_float(v) for v in exact]})
    forms = []
    for (n, dist) in iterates:
        text, cut = _closed_form(dist)
        forms.append({"n": n, "closed_form": to_syntax(text, syntax), "truncated": cut})
    guess = None
    try:
        fit = None
        if not truncated_terms:
            consecutive = [coefficients(dist, variables)[0] for dist in last] if len(last) == 3 else []
            limits = extrapolate(consecutive) if consecutive else {}
            fit = rational_guess(limits or tables[-1], variables)
    except Exception:  # pylint: disable=broad-except  # a guess is optional
        fit = None
    if fit is not None:
        guess = {"candidate": to_syntax(str(fit), syntax), "method": "rational_fit"}
    return {
        "variables": variables,
        "iterates": forms,
        "states": rows,
        "states_total": len(states),
        "terms_truncated": truncated_terms,
        "guess": guess,
    }
