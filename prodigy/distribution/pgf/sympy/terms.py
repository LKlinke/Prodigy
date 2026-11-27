"""
The terms of the SymPy engine's generating functions (#511, P4c): iteration over (coefficient, state) pairs, the
coefficient of a state and the conversion between monomials and states. A mixin of
:class:`~prodigy.distribution.pgf.sympy.engine.SympyPGF`; it only reads the closed form and the names.
"""
from __future__ import annotations

import logging
from typing import Iterator, Set, Tuple

import sympy

from prodigy.distribution.values import State, canonical_order
from prodigy.distribution.pgf.rules.value_at_one import _cancel
from prodigy.distribution.pgf.sympy.expressions import (Term, _parse_to_sympy, _sympy_symbol, _taylor_coefficient,
                                                        _term_generator)
from prodigy.language.engine_text import engine_text_of

logger = logging.getLogger("generating_function")

__all__ = ["SympyTerms"]


class SympyTerms:
    """The terms of a SymPy closed form ``_function`` in the variables ``_variables``."""

    _function: sympy.Expr
    _variables: Set[sympy.Symbol]
    _parameters: Set[sympy.Symbol]
    _is_closed_form: bool
    _is_finite: bool

    def _mult_term_generator(self) -> Iterator[Term]:
        """
            Generates terms of multivariate generating function in `grlex` order.
        """
        i = 1
        # sorted by name: the order of the set of variables depends on the hash seed of the process, and with it
        # which states a search such as get_state visits first (and how long it takes)
        variables = sorted(self._variables, key=str)
        while True:
            # The easiest method is to create all monomials of until total degree `i` and sort them.
            logger.debug("Generating and sorting of new monomials")
            new_monomials = sorted(
                sympy.polys.monomials.itermonomials(variables, i),
                key=sympy.polys.orderings.monomial_key("grlex", variables))
            # instead of repeating all the monomials for higher total degrees, just cut-off the already created ones.
            if i > 1:
                new_monomials = new_monomials[
                                sympy.polys.monomials.
                                monomial_count(len(self._variables), i - 1):]
            logger.debug("Monomial_generation done")

            # after we have the list of new monomials, create (probability, state) pais and yielding them.
            for monomial in new_monomials:
                state: dict[str, int] = {}
                if monomial == 1:
                    for var in self._variables:
                        state[var.name] = 0
                else:
                    state = {
                        engine_text_of(var): val
                        for var, val in
                        monomial.as_expr().as_powers_dict().items()
                    }
                yield self._probability_of_state(State(valuations=state)), monomial.as_expr()
            logger.debug("\t>Terms generated until total degree of %d", i)
            i += 1

    @staticmethod
    def _split_addend(addend: sympy.Expr) -> Tuple[sympy.Expr, sympy.Expr]:
        r"""
        This method assumes that the addend is given in terms of :math:`\alpha \times X^{\sigma}`, where
        :math:`\alpha \in [0,1], \sigma \in \mathbb{N}^k`.

        :param addend: the addend to split into its factor and monomial.

        :return: a tuple (factor, monomial)
        """
        if addend.free_symbols == set():
            return addend, sympy.Integer(1)
        else:
            factor_powers = addend.as_powers_dict()
            result = (sympy.Integer(1), sympy.Integer(1))
            for factor in factor_powers:
                if factor in addend.free_symbols:
                    result = (result[0],
                              result[1] * factor ** factor_powers[factor])
                else:
                    result = (result[0] * factor ** factor_powers[factor],
                              result[1])
            return result

    def _monomial_to_state(self, monomial: sympy.Expr) -> State:
        """ Converts a `monomial` into a state representation, i.e., a (variable_name, variable_value) dict.
            This is tied to a specific instance of a generating function as a variable and parameter context
            is needed to resolve the types of indeterminates in the monomial.
        """
        result = State()
        if monomial.free_symbols == set():
            for var in self._variables:
                result[engine_text_of(var)] = 0
        else:
            variables_and_powers = monomial.as_powers_dict()
            for var in self._variables:
                if var in variables_and_powers.keys():
                    result[engine_text_of(var)] = int(variables_and_powers[var])
                else:
                    result[engine_text_of(var)] = 0
        return result

    def _probability_of_state(self, state: State) -> sympy.Expr:
        """
        Determines the probability of a single program state encoded by a monomial (discrete only).

        :param state: The queried program state.

        :return: The probability for that state.
        """
        logger.debug("probability_of(%s) call", state)

        # complete state by assuming every variable which is not occurring is zero.
        complete_state = state
        for s_var in self._variables:
            if engine_text_of(s_var) not in state:
                complete_state[engine_text_of(s_var)] = 0

        # When its a closed form, or not finite, do the maths.
        if self._is_closed_form or not self._is_finite:
            result = self._function
            for variable, value in complete_state.items():
                result = _taylor_coefficient(result, _sympy_symbol(variable), value)
            return result
        # Otherwise use poly module and simply extract the correct coefficient from it
        else:
            monomial = 1
            for variable, value in complete_state.items():
                monomial *= _sympy_symbol(variable) ** value
            probability = self._function.as_poly(
                *self._variables).coeff_monomial(monomial)
            return probability if probability else sympy.core.numbers.Zero()

    # Distribution interface implementation

    def __iter__(self) -> Iterator[Tuple[str, State]]:
        """ Iterates over the generating function yielding (coefficient, state) pairs.

            For implicit (closed-form) generating functions, we generate terms via tailor expansion.
            If the generating function is even multivariate this might be costly as we try to enumerate all different
            k-tuples, for which the coefficient might be zero (this we do not know upfront).
        """
        logger.debug("iterating over %s...", self)
        items = map(lambda term: (engine_text_of(term[0]), self._monomial_to_state(term[1])), self._iter_expr_expr())
        return iter(canonical_order(items)) if self._is_finite else items

    def _iter_expr_state(self) -> Iterator[Tuple[sympy.Expr, State]]:
        """Iterates over self and yields the (coefficient, state) pairs as a sympy expression and a `State`"""
        return map(lambda term: (term[0], self._monomial_to_state(term[1])),
                   self._iter_expr_expr())

    def _iter_expr_expr(self) -> Iterator[Tuple[sympy.Expr, sympy.Expr]]:
        """Iterates over self and yields the (coefficient, state) pairs as sympy expressions"""

        if self._is_finite:
            if self._is_closed_form:
                # finite: the cancelled form is a polynomial over a constant denominator
                func = _cancel(self._function).expand().as_poly(*self._variables)
            else:
                if not self._variables:
                    logger.warning(
                        "Empty Polynomial, introducing auxilliary variable to create polynomial."
                    )
                    func = self._function.as_poly(
                        _parse_to_sympy("empty_generator"))
                else:
                    func = self._function.as_poly(*self._variables)
            return _term_generator(func)
        else:
            logger.debug("Multivariate Taylor expansion might take a while...")
            return self._mult_term_generator()

    # FIXME: It's not nice to have different behaviour depending on the variable type of `threshold`.
