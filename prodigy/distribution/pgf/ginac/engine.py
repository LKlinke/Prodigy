"""
The GiNaC engine :class:`GinacPGF` (#511): the CAS hooks of :class:`~prodigy.distribution.pgf.hooks.PGFHooks`
on the bindings ``prodigy_ginac``, and the factory of the common distributions (``GinacPGF.geometric(…)``).
The case analysis of the operations is the shared logic of :class:`~prodigy.distribution.pgf.pgf.PGF`.
``FPS`` and ``ProdigyPGF`` are deprecated aliases for one release.
"""
from __future__ import annotations

import itertools
import re

from typing import (cast, Any, Dict, FrozenSet, Generator, Iterator, List, Optional, Set, Tuple,
                    Type, Union)

import prodigy_ginac
from sympy import Expr as Expr_, Symbol, parse_expr
from sympy.parsing.sympy_parser import convert_xor, standard_transformations

from prodigy.distribution.pgf.rules.negative_values import can_be_negative, operand
from prodigy.distribution.pgf.rules.roots_of_unity import require_single_valued
from prodigy.distribution.distribution import Distribution, Operand
from prodigy.distribution.errors import NonFiniteMarginalError
from prodigy.distribution.values import MarginalType, State, canonical_order
from prodigy.distribution.pgf.pgf import PGF
from prodigy.language import Binop, Expr, VarExpr

from prodigy.util.order import default_monomial_iterator
from prodigy.language.engine_text import engine_text_of


def _is_division_by_zero(error: ValueError) -> bool:
    """Whether the bindings raised `error` because they evaluated an expression at a pole."""
    return "division by zero" in engine_text_of(error)


def _is_singular_at_one(error: ValueError) -> bool:
    """Whether the bindings failed to evaluate a moment at 1 because of a pole or a logarithmic singularity (#371)."""
    return _is_division_by_zero(error) or "log(0)" in engine_text_of(error)


class _PrefixCoefficients:
    """The coefficients of a multivariate generating function, expanded one variable at a time: the
    coefficient of ``v0^a`` is a generating function in the other variables, and so on. The coefficients of
    every prefix ``(a, b, ...)`` are kept, and a prefix with coefficient 0 cuts off all states that extend
    it. Filtering every state on its own made the graded order (many states per level, mostly with
    probability 0) far too slow for the witness search of a comparison (#403)."""

    def __init__(self, dist: prodigy_ginac.Dist, variables: List[str], parameters: List[str]):
        self._variables = variables
        self._parameters = parameters
        # prefix -> (iterator over the coefficients in the next variable, the coefficients seen so far)
        self._expansions: Dict[Tuple[int, ...], Tuple[Any, List[Optional[prodigy_ginac.Dist]]]] = {}
        self._root = dist

    def _coefficient(self, prefix: Tuple[int, ...]) -> Optional[prodigy_ginac.Dist]:
        """The coefficient of the monomial of ``prefix`` in its variables, None if it is 0."""
        if not prefix:
            return self._root
        parent = self._coefficient(prefix[:-1])
        if parent is None:
            return None
        head = prefix[:-1]
        variable = self._variables[len(head)]
        if variable not in set(parent.get_symbols()):
            # the bindings expand only in symbols that occur: the coefficients are the parent, then 0
            return parent if prefix[-1] == 0 else None
        if head not in self._expansions:
            self._expansions[head] = (parent.coefficient_iterator(variable), [])
        iterator, seen = self._expansions[head]
        while len(seen) <= prefix[-1]:
            coefficient = prodigy_ginac.Dist(engine_text_of(iterator.next()), self._parameters)
            seen.append(None if coefficient.is_zero() == prodigy_ginac.troolean.true else coefficient)
        return seen[prefix[-1]]

    def of(self, state: Tuple[int, ...]) -> Optional[str]:
        """The probability of ``state`` (one value per variable), None if it is 0."""
        coefficient = self._coefficient(state)
        return None if coefficient is None else engine_text_of(coefficient)


class GinacPGF(PGF):  # pylint: disable=too-many-ancestors  # the layers of PGF and the two interfaces
    """
    This class models a probability distribution in terms of a formal power series.
    These formal powerseries are itself provided by `prodigy` a python binding to GiNaC,
    something similar to a computer algebra system implemented in C++.
    """

    def __init__(self,
                 expression: str,
                 *variables: str | VarExpr,
                 finite: bool | None = None):
        self._variables = set(engine_text_of(var) for var in variables if engine_text_of(var) != "")
        self._parameters = set()
        for var in prodigy_ginac.find_symbols(expression):
            if var not in self._variables:
                if len(variables) > 0:
                    self._parameters.add(var)
                else:
                    self._variables.add(var)
        self._dist = prodigy_ginac.Dist(expression, list(self._parameters))
        self._finite = finite if finite is not None else self._dist.is_polynomial(
            self._variables) == prodigy_ginac.troolean.true

    @classmethod
    def from_dist(cls,
                  dist: prodigy_ginac.Dist,
                  variables: Set[str],
                  parameters: Set[str],
                  finite: bool | None = None) -> GinacPGF:
        result = cls("0")
        result._dist = dist
        result._variables = set(variables)
        result._parameters = set(parameters)
        result._finite = finite if finite is not None else dist.is_polynomial(variables) == prodigy_ginac.troolean.true
        return result

    @staticmethod
    def factory() -> Type[GinacPGF]:
        return GinacPGF

    def __add__(self, other: Operand) -> GinacPGF:
        if isinstance(other, (str, int)):
            return GinacPGF.from_dist(
                self._dist + prodigy_ginac.Dist(
                    engine_text_of(other),
                    list(
                        set(prodigy_ginac.find_symbols(engine_text_of(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, GinacPGF):
            self._require_compatible_names(other)
            return GinacPGF.from_dist(self._dist + other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Addition of {self._dist} (type {type(self._dist)}) and {other} (type {type(other)} not supported.")

    def __sub__(self, other: Operand) -> GinacPGF:
        if isinstance(other, (str, int)):
            return GinacPGF.from_dist(
                self._dist - prodigy_ginac.Dist(
                    engine_text_of(other),
                    list(
                        set(prodigy_ginac.find_symbols(engine_text_of(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, GinacPGF):
            self._require_compatible_names(other)
            return GinacPGF.from_dist(self._dist - other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Subtraction of {self._dist} and {other} not supported.")

    def __mul__(self, other: Operand) -> GinacPGF:
        if isinstance(other, (str, int)):
            return GinacPGF.from_dist(
                self._dist * prodigy_ginac.Dist(
                    engine_text_of(other),
                    list(
                        set(prodigy_ginac.find_symbols(engine_text_of(other))) -
                        self._variables)), self._variables, self._parameters)
        elif isinstance(other, GinacPGF):
            self._require_compatible_names(other)
            return GinacPGF.from_dist(self._dist * other._dist,
                                 self._variables | other._variables,
                                 self._parameters | other._parameters)
        else:
            raise NotImplementedError(
                f"Multiplication of {type(self._dist)} and {type(other)} not supported."
            )

    def __truediv__(self, other: Operand) -> GinacPGF:
        if isinstance(other, str):
            other = GinacPGF(other, *self._variables)
        if isinstance(other, GinacPGF):
            self._require_compatible_names(other)
            parameters = self._parameters | other._parameters
            return GinacPGF.from_dist(self._dist * prodigy_ginac.Dist(f"1/({str(other)})", sorted(parameters)),
                                 self._variables | other._variables,
                                 parameters)
        raise NotImplementedError(
            f"Division of {type(self._dist)} and {type(other)} not supported.")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = GinacPGF(other)
        if isinstance(other, GinacPGF):
            if not self._variables == other._variables:
                return False
            if not self._parameters == other._parameters:
                return False
            return self._dist == other._dist
        else:
            return False

    def __str__(self) -> str:
        return engine_text_of(self._dist)

    def __repr__(self) -> str:
        return self._dist.__repr__()

    def __iter__(self) -> Iterator[Tuple[str, State]]:
        if not self._finite:
            if len(self._variables) == 1:
                variable = list(self._variables)[0]
                it = self._dist.coefficient_iterator(variable)
                i = 0
                while True:
                    yield engine_text_of(it.next()), State({variable: i})
                    i += 1
            else:
                variables = sorted(self.get_variables())  # deterministic, independent of the hash seed
                coefficients = _PrefixCoefficients(self._dist, variables, sorted(self._parameters))
                for vals in default_monomial_iterator(len(variables)):
                    mass = coefficients.of(tuple(vals))
                    if mass is not None:
                        yield mass, State(dict(zip(variables, vals, strict=True)))
        else:
            # one item per state with its full coefficient since prodigy-ppl-ginac 0.0.7 (#444)
            terms: List[Tuple[str, Dict[str, int]]] = self._dist.get_terms(self._variables)
            yield from canonical_order((prob, State(vals)) for prob, vals in terms)

    def native_coefficients(self, variable: str) -> Optional[Iterator[str]]:
        # the bindings expand the series themselves, one coefficient at a time (as `__iter__` does)
        if self._finite or self._variables != {variable}:
            return None
        iterator = self._dist.coefficient_iterator(variable)
        return (engine_text_of(iterator.next()) for _ in itertools.count())

    def copy(self, deep: bool = True) -> GinacPGF:
        return GinacPGF.from_dist(self._dist, self._variables, self._parameters,
                             self._finite)

    def _probability_mass(self) -> str:
        try:
            return self._dist.mass(self._variables & set(self._dist.get_symbols()))
        except ValueError as error:
            if not _is_division_by_zero(error):
                raise
            # a pole at 1 (infinite or undefined mass, e.g. the test distribution of the second-order check,
            # #403) or a removable singularity whose limit GiNaC cannot take; as for marginals (#349)
            raise NonFiniteMarginalError(
                "The mass is not finite, or this engine cannot evaluate it: division by zero (a pole at 1, or a "
                f"singularity whose limit it cannot take) in {self._dist}") from error

    def _expected_value_of(self, expression: Union[Expr, str]) -> str:
        # Symbols that are neither variables nor parameters have no value: reject them, as the SymPy engine
        # does (the bindings reject them as well since prodigy-ppl-ginac 0.0.5).
        symbols = self._find_symbols(engine_text_of(expression))
        if not symbols & self._variables:
            return engine_text_of(expression)
        if not symbols <= self._variables | self._parameters:
            raise ValueError(f"Cannot compute expected value of {expression} because it contains unknown symbols")
        # A variable that does not occur in the generating function is 0 in every state, but the bindings
        # do not know it as a variable; substitute its value.
        absent = (symbols & self._variables) - set(self._dist.get_symbols())
        text = engine_text_of(expression)
        if absent:
            text = re.sub(r"\b(" + "|".join(sorted(map(re.escape, absent))) + r")\b", "0", text)
        try:
            return self._dist.E(text)
        except ValueError as error:
            if not _is_singular_at_one(error):
                raise
        # The bindings evaluate the moments at 1 and fail at a pole (#227) or at log(0) for a logarithmic
        # generating function (#371), e.g. an infinite expected value.
        # The shared rule takes the limit from below, which is the value of the series (Abel's theorem), on the
        # closed form read by SymPy (the bindings' marginal would evaluate at the pole as well).
        return self._series_expected_value(text, self._sympy_expression())

    def _normalize(self) -> GinacPGF:
        return GinacPGF.from_dist(self._dist.normalize(), self._variables, self._parameters)

    def get_variables(self) -> Set[str]:
        return set(self._variables)

    def get_parameters(self) -> Set[str]:
        return set(self._parameters)

    def _find_symbols(self, expr: str) -> Set[str]:
        return set(prodigy_ginac.find_symbols(expr))

    def get_symbols(self) -> Set[str]:
        return set(self._dist.get_symbols())

    @staticmethod
    def evaluate(expression: str, state: State) -> int:
        return prodigy_ginac.evaluate(expression, state.valuations)

    def _exhaustive_search(self, condition: Expr) -> GinacPGF:
        res = prodigy_ginac.Dist('0')
        for prob, state in self:
            if self.evaluate_condition(condition, state):
                # a coefficient with parameters is a sum: in parentheses (#444)
                res += prodigy_ginac.Dist(f"({prob}) * {state.to_monomial()}", sorted(self._parameters))
        return GinacPGF.from_dist(res,
                             self._variables,
                             self._parameters,
                             finite=True)

    def _filter_relation(self, variable: str, relation: Binop, bound: int) -> GinacPGF:
        filtered = (self._dist.filterLeq if relation == Binop.LEQ else self._dist.filterEq)(variable, str(bound))
        return GinacPGF.from_dist(filtered, self._variables, self._parameters)

    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> List[GinacPGF]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """
        if not self._finite:
            require_single_valued(self._sympy_function(), variable, f"{variable} % {modulus}")
        return [
            GinacPGF.from_dist(dist, self._variables, self._parameters)
            for dist in self._dist.arithmetic_progression(variable, modulus)
        ]

    def hadamard_product(self, other: Distribution) -> GinacPGF:
        raise NotImplementedError("Hadamard product currently not implemented in GiNaC")

    def is_zero_dist(self) -> bool:
        res = self._dist.is_zero()
        if res == prodigy_ginac.troolean.false:
            return False
        elif res == prodigy_ginac.troolean.true:
            return True
        else:
            raise ValueError('Cannot determine whether this GinacPGF is zero')

    def is_finite(self) -> bool:
        return self._finite

    def get_fresh_variable(
            self, exclude: Set[str] | FrozenSet[str] = frozenset()) -> str:
        res: str = self._dist.get_fresh_variable()
        while res in exclude:
            res = self._dist.get_fresh_variable()
        return res

    def _update_var(self, updated_var: str, assign_var: str | int) -> GinacPGF:
        if engine_text_of(assign_var) in self._parameters:
            raise ValueError("Not implemented: a parameter cannot be the source of an assignment "
                             f"({updated_var} = {assign_var}); parameters are unknown constants, "
                             "not values a program variable can take")
        # A variable that does not occur in the generating function is 0 (prodigy-ppl-ginac >= 0.0.6, #232).
        return GinacPGF.from_dist(
            self._dist.update_var(updated_var, engine_text_of(assign_var)),
            self._variables, self._parameters, self._finite)

    def _update_sum(self, temp_var: str, first_summand: str | int,
                    second_summand: str | int) -> GinacPGF:
        return GinacPGF.from_dist(
            self._dist.update_sum(temp_var, engine_text_of(first_summand),
                                  engine_text_of(second_summand)), self._variables,
            self._parameters)

    def _update_subtraction(self, temp_var: str, sub_from: str | int,
                            sub: str | int) -> GinacPGF:
        res = self._dist.update_subtraction(temp_var, engine_text_of(sub_from), engine_text_of(sub))
        return GinacPGF.from_dist(res, self._variables, self._parameters)

    def _subtraction_never_negative(self, minuend: str, subtrahend: str) -> bool:
        variables = self.get_variables()
        return can_be_negative(self._sympy_function(), {Symbol(var) for var in variables},
                               operand(minuend, variables), operand(subtrahend, variables)) is False

    def _sympy_function(self) -> Expr_:
        """This generating function as a SymPy expression (all symbols without assumptions)."""
        text = engine_text_of(self._dist)
        symbols = {name: Symbol(name) for name in prodigy_ginac.find_symbols(text)}
        return parse_expr(text, local_dict=symbols,
                          transformations=standard_transformations + (convert_xor,))

    def _sympy_expression(self) -> Expr_:
        return self._sympy_function()

    def _from_sympy_expression(self, expression: Expr_) -> GinacPGF:
        dist = prodigy_ginac.Dist(engine_text_of(expression).replace('**', '^'), list(self._parameters))
        return GinacPGF.from_dist(dist, set(self._variables), set(self._parameters))

    def _update_scaled(self, temp_var: str, variable: str, factor: int) -> GinacPGF:
        return GinacPGF.from_dist(
            self._dist.update_product(temp_var, variable, str(factor), self._variables, self._finite, None),
            self._variables, self._parameters)

    def _termwise(self, binop: Binop, temp_var: str, left: str, right: str) -> Optional[GinacPGF]:
        # the bindings multiply and take remainders term by term on a polynomial; for / and ** the shared
        # logic fixes the operands' values
        if binop == Binop.TIMES:
            dist = self._dist.update_product(temp_var, left, right, self._variables, True, None)
        elif binop == Binop.MODULO:
            dist = self._dist.update_modulo(temp_var, left, right, self._variables, True, None)
        else:
            return None
        return GinacPGF.from_dist(dist, self._variables, self._parameters)

    def _compose_iid(self, sample: PGF, count: Union[str, int], variable: str) -> GinacPGF:
        result = self._dist.updateIid(variable, cast(GinacPGF, sample)._dist,  # pylint: disable=protected-access
                                      str(count))
        return GinacPGF.from_dist(result, self._variables,
                                  self._parameters | (sample.get_parameters() - self._variables))

    def _marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> GinacPGF:
        result = self._dist
        remove_vars = {
            MarginalType.EXCLUDE: {str(var)
                                   for var in variables},
            MarginalType.INCLUDE:
                self._variables - {str(var)
                                   for var in variables}
        }
        for var in remove_vars[method]:
            try:
                result = result.update_var(engine_text_of(var), "0")
            except ValueError as error:
                if not _is_division_by_zero(error):
                    raise
                # division by zero at var = 1: a pole (infinite or undefined mass, #242) or a removable singularity
                # whose limit GiNaC cannot take; the message does not claim a pole (#349)
                kept = ", ".join(sorted(self._variables - remove_vars[method])) or "no variable"
                raise NonFiniteMarginalError(
                    f"The marginal of {kept} is not a finite measure, or this engine cannot evaluate it: "
                    f"division by zero (a pole at {var} = 1, or a singularity whose limit it cannot take) "
                    f"in {self._dist}") from error
        return GinacPGF.from_dist(result, self._variables - remove_vars[method],
                             self._parameters)

    def _with_names(self, variables: Set[str], parameters: Set[str]) -> GinacPGF:
        dist = self._dist
        if parameters != self._parameters:
            # the bindings keep the parameters of a function: a name that changes sides is read anew
            dist = prodigy_ginac.Dist(engine_text_of(dist), sorted(parameters))
        return GinacPGF.from_dist(dist, variables, parameters)

    def approximate(
            self,
            threshold: Union[str, int]) -> Generator[Distribution, None, None]:
        raise NotImplementedError("Not implemented: approximate (the partial sums of the series) on the GiNaC "
                                  "engine")

    def approximate_unilaterally(
            self, variable: str,
            probability_mass: str | float) -> GinacPGF:
        return GinacPGF.from_dist(
            self._dist.approximate_unilaterally(variable,
                                                engine_text_of(probability_mass)),
            self._variables, self._parameters)

    # The common distributions (the factory, #511)


    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> GinacPGF:
        return GinacPGF("1", *variables, finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables: Union[str, VarExpr], **kwargs: Optional[bool]) -> GinacPGF:
        return GinacPGF(engine_text_of(expression), *variables, finite=kwargs.get("finite"))

    @staticmethod
    def zero(*variables: Union[str, VarExpr]) -> GinacPGF:
        return GinacPGF("0", *variables, finite=True)


FPS = GinacPGF
"""Deprecated name of :class:`GinacPGF` (until v0.8.0, #511)."""

ProdigyPGF = GinacPGF
"""Deprecated name of :class:`GinacPGF` as the factory (until v0.8.0, #511)."""
