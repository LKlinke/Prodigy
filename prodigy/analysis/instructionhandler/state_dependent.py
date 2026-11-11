"""
Probabilities and distribution arguments that depend on the program state.

A probabilistic choice ``{...} [p] {...}`` and a sampling instruction such as
``x = bernoulli(p)`` are well defined when ``p`` depends on program variables, as long as ``p``
is a probability (or an admissible argument) in every reachable state. Prodigy computes such
an instruction by a case split over the states: if the program variables in ``p`` take only
finitely many values, the distribution is split into the parts in which they have fixed values,
``p`` is evaluated in each part, and the parts are combined again. The split is exact. If the
variables take infinitely many values, the instruction is not computed (``NotImplementedError``);
multiplying the generating function by ``p`` as if it were a constant would be wrong.
"""
from __future__ import annotations

import copy
from fractions import Fraction
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

from prodigy.pgcl import Expr, NatLitExpr, VarExpr
from prodigy.pgcl.ast.walk import Walk, walk_expr
from prodigy.pgcl.util.ref import Mut

from prodigy.distribution import Distribution
from prodigy.distribution.distribution import State, constant_value


def expression_variables(expressions: List[Expr]) -> Set[str]:
    """The names of all variables and parameters that occur in ``expressions``."""
    return {ref.val.var for expr in expressions for ref in walk_expr(Walk.DOWN, Mut.alloc(expr))
            if isinstance(ref.val, VarExpr)}


def state_variables(expressions: List[Expr], distribution: Distribution) -> Set[str]:
    """The program variables of ``distribution`` that occur in ``expressions`` (parameters are not
    program variables: the result is symbolic in them)."""
    return expression_variables(expressions) & distribution.get_variables()


def instantiate(expression: Expr, state: State) -> Expr:
    """``expression`` with every variable of ``state`` replaced by its value in ``state``."""
    result: Mut[Expr] = Mut.alloc(copy.deepcopy(expression))
    for ref in walk_expr(Walk.DOWN, result):
        if isinstance(ref.val, VarExpr) and ref.val.var in state.valuations:
            ref.val = NatLitExpr(state[ref.val.var])
    return result.val


class StatePart(NamedTuple):
    """The part of a distribution in which some variables have fixed values."""
    state: State
    """The values of the variables."""
    distribution: Distribution
    """The part of the distribution in ``state``."""

    def describe(self) -> str:
        """The state and its probability (the mass of the part, if it has one), for messages."""
        values = ", ".join(f"{var} = {value}" for var, value in sorted(self.state.items()))
        try:
            mass = self.distribution.get_probability_mass()
        except (ArithmeticError, ValueError, NotImplementedError):
            return f"the state {values}"
        return f"the state {values}, which has probability {mass}"


def _is_zero(probability: str) -> bool:
    try:
        return Fraction(probability.strip()) == 0
    except ValueError:
        return False  # symbolic in the parameters


def split_states(distribution: Distribution, variables: Set[str], construct: str) -> Iterator[StatePart]:
    """
    The parts of ``distribution`` in which ``variables`` have fixed values, one for every value of
    ``variables`` in the support of ``distribution``. The parts sum to ``distribution``. Raises
    ``NotImplementedError`` (naming ``construct``) if the variables take infinitely many values
    (``NonFiniteMarginalError`` if, in addition, their marginal is not a finite measure).

    The split follows the support (:meth:`Distribution.finite_support`), not the marginal: in the
    marginal of a signed measure the mass of a value can cancel to 0, and its part would be lost (#256).
    """
    if distribution.is_zero_dist():
        return
    support = distribution.finite_support(*sorted(variables))
    if support is None:
        distribution.checked_marginal(*sorted(variables))
        names = ", ".join(sorted(variables))
        raise NotImplementedError(
            f"{construct} depends on the program variable(s) {names}, which can take infinitely "
            f"many values here. Prodigy computes a probability or distribution argument that "
            f"depends on program variables only if these variables take finitely many values "
            f"(by a case split over their values); this is not implemented for infinitely many.")
    # The iteration of an engine may list a state with coefficient 0 (e.g. of the zero distribution),
    # and a state once per term (GiNaC, see Distribution._unique_states): each state is split once.
    states: Dict[Tuple[Tuple[str, int], ...], State] = {}
    for coefficient, state in support:
        if not _is_zero(coefficient):
            states.setdefault(tuple(sorted(state.valuations.items())), state)
    for state in states.values():
        yield StatePart(state, distribution.filter_state(state))


def checked_probability(probability: Expr, construct: str,
                        part: Optional[StatePart] = None) -> Optional[Fraction]:
    """
    The value of ``probability`` if it is a constant (``None`` if it contains parameters). Raises
    ``ValueError`` if it is not a probability, i.e. undefined (e.g. a division by zero) or outside
    of [0, 1]; ``part`` is the part of the distribution the value was computed in, for the message.
    """
    if expression_variables([probability]):
        # symbolic in the parameters, like a constant parametric probability
        return None
    where = f" in {part.describe()}" if part else ""
    value = constant_value(probability)
    if value is None:
        raise ValueError(f"{construct} is undefined{where}: {probability} has no value.")
    if not 0 <= value <= 1:
        raise ValueError(f"{construct} is {value}{where}, which is not a probability "
                         f"(it must be in [0, 1]).")
    return value
