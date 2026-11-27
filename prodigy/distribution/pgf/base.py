"""
The shared logic of generating functions, first layer (#511): names, comparisons, probability mass, supports,
marginals and the convergence proofs of every evaluation at 1.
"""
from __future__ import annotations

import contextlib
from fractions import Fraction
from typing import TYPE_CHECKING, cast, ContextManager, Dict, List, Optional, Self, Set, Tuple, Union

import sympy

from prodigy.language import Binop, BinopExpr, Expr, RealLitExpr, Unop, UnopExpr, VarExpr
from prodigy.language.engine_text import engine_text_of
from prodigy.distribution.distribution import Distribution
if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.pgf.pgf import PGF
from prodigy.distribution.errors import NonFiniteMarginalError, declared_convergence, known_convergent_series
from prodigy.distribution.values import MarginalType, OrderVerdict, State, Value
from prodigy.distribution.terms import is_true, state_value
from prodigy.distribution.pgf.messages import with_terms
from prodigy.distribution.pgf.hooks import PGFHooks
from prodigy.distribution.engine_text import parse_engine_text, sympy_text
from prodigy.distribution.pgf.rules.convergence import marginal_obstacle
from prodigy.distribution.pgf.rules.value_at_one import _substitute_one
from prodigy.distribution.pgf.rules.equality import decide_equal_strings, sympify_distribution_string
from prodigy.distribution.pgf.rules.finite_measure import check_input_expression
from prodigy.distribution.pgf.rules.ordering import dominates_finite, nonneg_verdict


class PGFBase(PGFHooks):  # pylint: disable=abstract-method  # a layer of PGF
    """Names, comparisons, supports, marginals and convergence proofs of generating functions."""

    def _declared_context(self) -> ContextManager[None]:
        """The context of an evaluation of this object: :func:`known_convergent_series` if it is declared."""
        return known_convergent_series(True) if self._declared_finite_measure else contextlib.nullcontext()

    def _require_compatible_names(self, other: PGF) -> None:
        """
        Raises `ArithmeticError` if a name is a variable of one operand of an arithmetic operation and a
        parameter of the other: the result would have to treat it as both (the same on all engines, #361).
        """
        clash = (self.get_variables() & other.get_parameters()) | (self.get_parameters() & other.get_variables())
        if clash:
            raise ArithmeticError(f"Name clash: {', '.join(sorted(clash))} is a variable of one distribution and a "
                                  f"parameter of the other.")

    def decide_equal(self, other: object) -> Optional[bool]:
        if self == other:
            return True
        if not isinstance(other, Distribution):
            return None
        names = self.get_variables() | self.get_parameters() | other.get_variables() | other.get_parameters()
        return decide_equal_strings(engine_text_of(self), engine_text_of(other), names)

    def _series_expression(self) -> sympy.Expr:
        """
        The generating function as a SymPy expression, for the shared coefficient-sign checks. Engines whose
        generating function already is a SymPy expression override this hook.
        """
        return sympify_distribution_string(self)


    def _require_comparable(self, other: object) -> PGF:
        if not isinstance(other, Distribution) or type(other) is not type(self):
            raise TypeError(f"Incomparable types {type(self)} and {type(other)}.")
        if self.get_variables() != other.get_variables() or self.get_parameters() != other.get_parameters():
            raise TypeError(
                f"Incomparable distributions: variables {sorted(self.get_variables())} and parameters "
                f"{sorted(self.get_parameters())} differ from variables {sorted(other.get_variables())} and "
                f"parameters {sorted(other.get_parameters())}.")
        return cast("PGF", other)  # of the class of self

    def decide_le_verdict(self, other: Distribution) -> OrderVerdict:
        other = self._require_comparable(other)
        if self.decide_equal(other) is True:
            return OrderVerdict(True)
        difference = other - self
        verdict = nonneg_verdict(difference, difference._series_expression())  # pylint: disable=protected-access
        if verdict.holds is None and self.is_finite() and not other.is_finite() \
                and nonneg_verdict(other, other._series_expression()).holds is True \
                and dominates_finite(self, other):  # pylint: disable=protected-access
            return OrderVerdict(True)
        return verdict

    def get_probability_mass(self) -> str:
        self._require_convergent_evaluation(set(), "The probability mass")
        return self._probability_mass()

    def normalize(self) -> PGF:
        if self.is_zero_dist():
            # SymPy raised a bare ZeroDivisionError, GiNaC its own division error (#361)
            raise ZeroDivisionError("Cannot normalize the zero distribution: its probability mass is 0")
        self._require_convergent_evaluation(set(), "The normalization")
        return self._normalize()

    def get_symbols(self) -> Set[str]:
        return self._find_symbols(engine_text_of(self))

    @classmethod
    def evaluate_condition(cls, condition: Expr,
                           state: State) -> bool:

        if isinstance(condition, UnopExpr) and condition.operator == Unop.NEG:
            return not cls.evaluate_condition(condition.expr, state)
        if isinstance(condition, VarExpr):
            return cls.evaluate_condition(is_true(condition), state)
        if not isinstance(condition, BinopExpr):
            raise AssertionError(
                f"Expression must be an (in-)equation or negation, was {condition}"
            )

        op = condition.operator
        if op in (Binop.EQ, Binop.LEQ, Binop.LT, Binop.GT, Binop.GEQ):
            # exactly, with truncated subtraction, if both sides are numbers in this state
            left, right = state_value(condition.lhs, state), state_value(condition.rhs, state)
            lhs: Union[Fraction, Value]
            rhs: Union[Fraction, Value]
            if left is not None and right is not None:
                lhs, rhs = left, right
            else:
                lhs = cls.evaluate(engine_text_of(condition.lhs), state)
                rhs = cls.evaluate(engine_text_of(condition.rhs), state)
            if op == Binop.EQ:
                return lhs == rhs
            if op == Binop.LEQ:
                return lhs <= rhs
            if op == Binop.LT:
                return lhs < rhs
            if op == Binop.GT:
                return lhs > rhs
            return lhs >= rhs
        if op == Binop.OR:
            return cls.evaluate_condition(condition.lhs,
                                          state) or cls.evaluate_condition(
                condition.rhs, state)
        if op == Binop.AND:
            return cls.evaluate_condition(condition.lhs,
                                          state) and cls.evaluate_condition(
                condition.rhs, state)
        raise AssertionError(f"Unexpected condition type. {condition}")

    @staticmethod
    def _unique_states(values: Distribution) -> List[State]:
        """
        The states of `values` (a :meth:`finite_support` or a marginal) for a case split, each once.
        The iteration lists each state once (the contract of :meth:`__iter__`; GiNaC listed a state
        once per term before #444, and a case split counted its part twice); this guards a case split
        against an engine that breaks the contract.
        """
        states: Dict[Tuple[Tuple[str, int], ...], State] = {}
        for _, state in values:
            states.setdefault(tuple(sorted(state.valuations.items())), state)
        return list(states.values())

    def finite_support(self, *variables: Union[str, VarExpr]) -> Optional[PGF]:
        names = [engine_text_of(var) for var in variables]
        support = self.from_expr(self._engine_text(), *names)
        return support if support.is_finite() else None

    def marginal(self,
                 *variables: Union[str, VarExpr],
                 method: MarginalType = MarginalType.INCLUDE) -> PGF:
        named = {str(var) for var in variables}
        if not named:
            raise ValueError("No variables were provided")
        if not named <= self.get_variables():
            # GiNaC returned the mass as the marginal of a name that is not a variable (#361)
            raise ValueError(f"Unknown variable(s): {{{', '.join(sorted(named - self.get_variables()))}}}")
        marginal = self._marginal(*variables, method=method)
        kept = named if method == MarginalType.INCLUDE else self.get_variables() - named
        self._require_convergent_evaluation(kept, f"The marginal of {', '.join(sorted(kept)) or 'no variable'}")
        return marginal

    def checked_marginal(self, *variables: Union[str, VarExpr], proof: bool = True) -> PGF:
        names = ", ".join(engine_text_of(var) for var in variables)
        try:
            marginal = self._marginal(*variables)
        except ZeroDivisionError as exc:
            raise NonFiniteMarginalError(
                f"The marginal of {names} is undefined (division by zero) in {self}") from exc
        except ValueError as exc:
            # e.g. GiNaC: power::eval(): division by zero
            if "division by zero" not in engine_text_of(exc):
                raise
            raise NonFiniteMarginalError(
                f"The marginal of {names} is undefined (division by zero) in {self}") from exc
        self._require_finite_values(marginal, *variables)
        if proof:
            self._require_convergent_marginal(marginal, *variables)
        return marginal

    def _require_finite_values(self, marginal: PGF, *variables: Union[str, VarExpr]) -> None:
        """Raises `NonFiniteMarginalError` if `marginal`, the marginal of `variables`, is not finite."""
        if marginal.has_non_finite_value():
            names = ", ".join(engine_text_of(var) for var in variables)
            raise NonFiniteMarginalError(
                f"The marginal of {names} is {marginal}, not a finite measure: {self} has infinite "
                f"mass, so a case split over the values of {names} is not computable.")

    def _require_convergent_marginal(self, marginal: PGF, *variables: Union[str, VarExpr]) -> None:
        """
        Raises `NonFiniteMarginalError` if it is not proven that `marginal`, the closed form of this
        generating function at 1 in the variables other than `variables`, is the sum of its series
        (#393): a series that diverges at 1 can have a closed form with a finite value there, e.g.
        `1/((1 - 2x)(1 - y))` at x = 1. The decision is :func:`marginal_obstacle`; a finite support
        (a polynomial) sums finitely many terms and needs no check, and neither do the distributions of
        a program analysis (:func:`known_convergent_series`).
        """
        if declared_convergence():
            return
        kept = {str(var) for var in variables}
        obstacle = self._convergence_obstacle(kept)
        if obstacle is not None:
            names = ", ".join(engine_text_of(var) for var in variables)
            summed = ", ".join(sorted(self.get_variables() - kept))
            raise NonFiniteMarginalError(
                f"The marginal of {names} is not proven to be a finite measure: summing out "
                f"{summed} may diverge, as {obstacle}; its closed form {marginal} "
                f"need not be the sum of the series of {self}.")

    def _require_convergent_evaluation(self, kept: Set[str], what: str,
                                       terms: Optional[Dict[str, str]] = None) -> None:
        """
        The convergence proof of the public entry points that evaluate this generating function at 1 in
        the variables other than `kept` (#398): outside an analysis (:func:`known_convergent_series` not
        declared), raises `NonFiniteMarginalError` naming `what` if :func:`marginal_obstacle` does not
        prove that the closed form there is the sum of the series. Inside analyses this is a context
        lookup only. `terms` maps summed variables that the program did not write (the temporary variable of
        a term like `x - 1`, #425) to the term as written; the message names the terms.
        """
        if declared_convergence() is not None or self._declared_finite_measure:
            return
        obstacle = self._convergence_obstacle(kept)
        if obstacle is not None:
            terms = terms or {}
            summed_variables = sorted(self.get_variables() - kept)
            summed = ", ".join(terms.get(name, name) for name in summed_variables)
            raise NonFiniteMarginalError(with_terms(
                f"{what} is not proven to converge: it sets {summed} to 1 in {self}, and the series "
                f"need not converge there, as {obstacle}. Prodigy does not report the value of the "
                "closed form, which need not be the sum of the series.", terms))

    def _convergence_obstacle(self, kept: Set[str]) -> Optional[str]:
        """
        Why it is not proven that summing out every variable except `kept` gives the sum of the series, or
        None if it is: a finite support or nothing summed out, :func:`marginal_obstacle`, or (#398) the proof
        that the analysis requires of its inputs (#296): non-negative coefficients and finite mass. Then the
        series converges absolutely on the closed unit polydisc, and its sums are the values of the closed form
        there (Abel), also where the closed form is not analytic at 1, as for ``(1 - sqrt(1 - x^2))/x``.
        """
        summed = self.get_variables() - kept
        if not summed or self.is_finite():
            return None
        obstacle = marginal_obstacle(self._sympy_expression(), summed, kept & self.get_variables())
        if obstacle is not None and self._is_proven_finite_measure():
            return None
        return obstacle

    def _is_proven_finite_measure(self) -> bool:
        """Whether this is proven to be a power series with non-negative coefficients and finite mass, by the
        check of input distributions (:func:`~prodigy.distribution.pgf.rules.finite_measure.check_input_expression`)."""
        text = engine_text_of(self._sympy_expression()).replace("**", "^")
        verdict, _ = check_input_expression(text, sorted(self.get_variables()))
        return verdict is True

    def _expected_value_of(self, expression: Union[Expr, str]) -> str:
        """The expected value of `expression` (see :meth:`get_expected_value_of`) by :meth:`_series_expected_value`
        on the engine's marginal; an engine may compute it itself first."""
        return self._series_expected_value(expression)

    def _series_expected_value(self, expression: Union[Expr, str], function: Optional[sympy.Expr] = None) -> str:
        """
        The expected value of the polynomial `expression` in the variables and parameters: for each term
        ``c * x1^k1 * … * xm^km``, ``c`` times the value at 1 of ``(x1 d/dx1)^k1 … (xm d/dxm)^km`` applied to the
        marginal of ``x1, …, xm`` (the moments of a generating function). The value at 1 is the limit from below
        (Abel's theorem, :func:`~prodigy.distribution.pgf.rules.value_at_one._substitute_one`), so an infinite
        moment is ∞. The marginal is the engine's (:meth:`_marginal`), or, with `function` (this distribution as
        a SymPy expression), the other variables of `function` summed out by their values at 1.
        """
        expr = parse_engine_text(engine_text_of(expression), rational=True).ratsimp().expand()
        if not expr.is_polynomial():
            raise NotImplementedError("Expected Value only computable for polynomial expressions.")
        names = {symbol.name for symbol in expr.free_symbols}
        variables = sorted(names & self.get_variables())
        if not variables:
            return sympy_text(expr)
        if not names <= self.get_variables() | self.get_parameters():
            raise ValueError(f"Cannot compute expected value of {expression} because it contains unknown symbols")
        if function is None:
            marginal = self._marginal(*variables, method=MarginalType.INCLUDE)._sympy_expression()  # pylint: disable=protected-access
        else:
            marginal = function
            for symbol in sorted(function.free_symbols, key=lambda s: s.name):
                if symbol.name in self.get_variables() and symbol.name not in variables:
                    marginal = _substitute_one(marginal, symbol)
        symbols = {symbol.name: symbol for symbol in marginal.free_symbols}
        expr = expr.xreplace({symbol: symbols[symbol.name] for symbol in expr.free_symbols if symbol.name in symbols})
        generators = [symbols.get(name, sympy.Symbol(name)) for name in variables]
        expected_value = sympy.Integer(0)
        for exponents, coefficient in sympy.Poly(expr, *generators).terms():
            moment = marginal
            for symbol, exponent in zip(generators, exponents):
                for _ in range(exponent):
                    moment = moment.diff(symbol) * symbol
                moment = _substitute_one(moment, symbol)
            # subtractions of natural numbers are computed before (monus, get_expected_value_of), so a negative
            # summand is a negative coefficient of the (rational) expression itself
            expected_value += coefficient * moment
        if expected_value == sympy.oo:
            return engine_text_of(RealLitExpr.infinity())
        return sympy_text(expected_value)

    def set_variables(self, *variables: str) -> Self:
        new_variables = set(variables)
        dropped = (self.get_variables() - new_variables) & self.get_symbols()
        return self._with_names(new_variables, (self.get_parameters() - new_variables) | dropped)

    def set_parameters(self, *parameters: str) -> Self:
        new_parameters = set(parameters)
        clash = new_parameters & self.get_variables()
        if clash:
            raise ValueError(f"Indeterminate(s) {', '.join(sorted(clash))} cannot be parameters and variables at "
                             f"the same time.")
        dropped = (self.get_parameters() - new_parameters) & self.get_symbols()
        return self._with_names(self.get_variables() | dropped, new_parameters)
