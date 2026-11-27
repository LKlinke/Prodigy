"""
The shared logic of generating functions, second layer (#511): filters and the queries computed with them
(probabilities, expected values).
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union


from prodigy.language import Binop, BinopExpr, BoolLitExpr, Expr, NatLitExpr, Unop, UnopExpr, VarExpr
from prodigy.language.checks import check_is_constant_constraint, has_variable, modulus_condition
from prodigy.language.engine_text import engine_text_of
from prodigy.language.operations import state_to_equality_expression
from prodigy.language.ppy_parser import parse_ppy_expr
from prodigy.language.syntax_errors import ProgramSyntaxError
from prodigy.distribution.errors import NonFiniteMarginalError
if TYPE_CHECKING:  # pragma: no cover
    from prodigy.distribution.pgf.pgf import PGF
from prodigy.distribution.values import MarginalType, State
from prodigy.distribution.terms import (computed_term, constant_value, is_true, replace_term, spelled,
                                        substitute_value, variables_of)
from prodigy.distribution.pgf.messages import in_programs_terms
from prodigy.distribution.pgf.base import PGFBase
from prodigy.distribution.pgf.rules.algebraic_comparison import filter_difference_algebraic
from prodigy.distribution.pgf.rules.independent_comparison import (DEPENDENT, NOT_A_DIFFERENCE,
                                                                   ComparisonNotImplemented, filter_difference)
from prodigy.distribution.pgf.rules.linear_guards import difference_comparison, normalize_guard, single_variable_guard


CASE_SPLIT_LIMIT = 1000
"""The most values of one variable that the case split of :meth:`PGFFiltering._filter_by_cases` enumerates."""


class PGFFiltering(PGFBase):  # pylint: disable=abstract-method  # a layer of PGF
    """Filters of generating functions and the queries computed with them."""

    def get_probability_of(self, condition: Union[Expr, str]) -> str:

        expr = condition
        if isinstance(expr, str):
            expr = parse_ppy_expr(expr)
        with self._declared_context():
            try:
                filtered = self._filter_by_cases(expr)
            except NotImplementedError as error:
                filtered = self.query_marginal(expr, error)._filter_by_cases(expr)  # pylint: disable=protected-access
            # the mass of the filtered measure, proven outside analyses (#398)
            filtered._require_convergent_evaluation(set(), f"Pr[{expr}]")  # pylint: disable=protected-access
            return filtered._probability_mass()  # pylint: disable=protected-access

    def query_marginal(self, expression: Expr, error: NotImplementedError) -> PGF:
        names = variables_of(expression) & self.get_variables()
        if not names or names == self.get_variables():
            raise error
        try:
            return self.checked_marginal(*sorted(names))
        except NonFiniteMarginalError:
            raise error from None

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:
        with self._declared_context():
            return self._expected_value(expression)

    def _expected_value(self, expression: Union[Expr, str]) -> str:
        """:meth:`get_expected_value_of` in the context of this object."""
        parsed: Union[Expr, str] = expression
        if isinstance(expression, str):
            try:
                parsed = parse_ppy_expr(expression)
            except ProgramSyntaxError:
                # not in the .ppy syntax (e.g. `x^2`): the engine reads it itself
                return self._checked_expected_value_of(expression)
        assert not isinstance(parsed, str)
        try:
            extended, replaced, _ = self.without_computed_terms(parsed)
        except NotImplementedError as error:
            marginal = self.query_marginal(parsed, error)
            extended, replaced, _ = marginal.without_computed_terms(parsed)
        return extended._checked_expected_value_of(replaced)  # pylint: disable=protected-access

    def _checked_expected_value_of(self, expression: Union[Expr, str]) -> str:
        """:meth:`_expected_value_of` after the convergence proof of :meth:`get_expected_value_of`."""
        self._require_convergent_evaluation(set(), f"The expected value of {expression}")
        return self._expected_value_of(expression)

    def filter_state(self, state: State) -> PGF:

        if not state.valuations.keys() <= self.get_variables():
            raise ValueError("Unknown variable in state")

        res = self._pgf()
        for var, val in state.items():
            res = res._filter_relation(var, Binop.EQ, int(val))  # pylint: disable=protected-access
        return res

    def _filter_constant(self, condition: BinopExpr) -> PGF:
        """
        The filter of a comparison of a variable with a constant (`x < 5/2`, `3 >= x`, `x = 1/2`), normalised
        for every engine to the hook :meth:`_filter_relation` with `=` or `<=` and a natural bound (#511, P4d):
        the comparison is turned to `x op c`, `c` is evaluated exactly (:func:`constant_value`) and rounded for a
        natural `x` (:func:`~prodigy.distribution.pgf.rules.linear_guards.single_variable_guard`: `x < c` is
        `x < ceil(c)`, `x <= c` is `x <= floor(c)`, `x = c` holds nowhere unless `c` is a natural number), then
        `x < n` is `x <= n - 1`, and `x > n`, `x >= n` are the complements of `x <= n`, `x < n`.
        """
        flipped = {Binop.LT: Binop.GT, Binop.LEQ: Binop.GEQ, Binop.GT: Binop.LT, Binop.GEQ: Binop.LEQ,
                   Binop.EQ: Binop.EQ}
        if isinstance(condition.lhs, VarExpr):
            variable, relation, bound_expression = condition.lhs.var, condition.operator, condition.rhs
        else:
            assert isinstance(condition.rhs, VarExpr)
            variable, relation, bound_expression = condition.rhs.var, flipped[condition.operator], condition.lhs
        bound = constant_value(bound_expression)
        if bound is None:
            raise ValueError(f"Cannot filter for {spelled(condition)}: the bound is not a constant number")
        guard = single_variable_guard(variable, relation, bound)
        if not isinstance(guard, BinopExpr):
            return self.filter(guard)  # always or never
        assert isinstance(guard.rhs, NatLitExpr)
        if guard.operator == Binop.EQ:
            return self._filter_relation(variable, Binop.EQ, guard.rhs.value)
        # x <= n, and x < n as x <= n - 1 (n > 0); x > n and x >= n are their complements
        below = self._filter_relation(variable, Binop.LEQ,
                                      guard.rhs.value - (1 if guard.operator in (Binop.LT, Binop.GEQ) else 0))
        return below if guard.operator in (Binop.LEQ, Binop.LT) else self._pgf() - below

    def _filter_by_cases(self, condition: Expr) -> PGF:
        """
        ``self.filter(condition)``; if no filter rule applies to `condition` (e.g. ``x + y < 3`` where ``y`` has
        infinite support), the condition is split into cases over the values of a variable with a finite support
        (at most :data:`CASE_SPLIT_LIMIT` values): the filter is the sum over the values ``v`` of ``x`` of the
        filter of ``x = v`` and ``condition[x := v]`` (#306). The cases are disjoint, so the sum is exact. Raises
        the filter's NotImplementedError if no variable of the condition has such a support. Used by
        :meth:`get_probability_of`; guards of ``if`` and ``while`` use :meth:`filter` (#507).
        """
        try:
            return self.filter(condition)
        except NotImplementedError as error:
            for name in sorted(variables_of(condition) & self.get_variables()):
                support = self.finite_support(name)
                if support is None:
                    continue
                values = sorted({int(state.valuations[name]) for _, state in support})
                if len(values) > CASE_SPLIT_LIMIT:
                    continue
                cases = [substitute_value(condition, name, value) for value in values]
                if any(case is None for case in cases):
                    break
                total: Optional[PGF] = None
                for value, case in zip(values, cases, strict=True):
                    assert case is not None
                    part = self._filter_relation(name, Binop.EQ, value)._filter_by_cases(case)  # pylint: disable=protected-access
                    total = part if total is None else total + part
                if total is not None:
                    return total
            raise error

    def filter(self, condition: Expr) -> PGF:
        connective = self._filter_connective(condition)
        if connective is not None:
            return connective

        # A parameter is an unknown constant: whether `x < p` holds in a state depends on its value, which
        # the generating function does not fix. The engines read it as 0 or did not terminate (#361).
        parameters = sorted(variables_of(condition) & self.get_parameters())
        if parameters:
            raise NotImplementedError(
                f"Not implemented: the condition {spelled(condition)} depends on the parameter(s) "
                f"{', '.join(parameters)}, whose values are unknown, so it is not decided in any state.")

        # Floor divisions and subtractions (monus) are computed into fresh variables first, e.g.
        # x // 2 < 3 becomes t < 3 for t := x // 2 (an exact update); t is a function of the state,
        # so it can be dropped again
        if computed_term(condition, self.get_variables()) is not None:
            return self._filter_computed_terms(condition)
        return self._filter_comparison(condition)

    def _filter_connective(self, condition: Expr) -> Optional[PGF]:
        """The filter of a Boolean literal, `and`, `or`, `not` or a `bool` variable, from the filters of the
        parts; None for any other condition."""
        if isinstance(condition, BoolLitExpr):
            return self._pgf() if condition.value else self.from_expr("0", *self.get_variables())
        if isinstance(condition, BinopExpr) and condition.operator == Binop.AND:
            return self.filter(condition.lhs).filter(condition.rhs)
        if isinstance(condition, BinopExpr) and condition.operator == Binop.OR:
            filtered_left = self.filter(condition.lhs)
            return filtered_left + self.filter(condition.rhs) - filtered_left.filter(condition.rhs)
        if isinstance(condition, VarExpr):
            # a bool variable as condition: it holds where the variable is 1, i.e. not 0
            return self.filter(is_true(condition))
        if isinstance(condition, UnopExpr):
            if condition.operator == Unop.NEG:
                return self - self.filter(condition.expr)
            raise SyntaxError(f"We do not support filtering for {condition.operator} expressions.")
        return None

    def _filter_computed_terms(self, condition: Expr) -> PGF:
        """The filter of a condition with floor divisions or subtractions, on the distribution extended by
        their values (:meth:`_replace_computed_terms`), which are summed out again."""
        extended, replaced, fresh, names = self._replace_computed_terms(condition)
        try:
            filtered = extended.filter(replaced)
        except (NotImplementedError, NonFiniteMarginalError, ValueError) as error:
            # the guard names the fresh variables; the program wrote the terms (#425)
            raise in_programs_terms(error, names) from error
        if not fresh:
            return filtered
        # summing out the fresh variables evaluates at 1 (#427)
        filtered._require_convergent_evaluation(  # pylint: disable=protected-access
            filtered.get_variables() - set(fresh), f"The filter of {spelled(condition)}", names)
        return filtered._marginal(*fresh, method=MarginalType.EXCLUDE)  # pylint: disable=protected-access

    def _filter_comparison(self, condition: Expr) -> PGF:
        """The filter of a comparison without computed terms and parameters: linear guards are normalised, constant
        ones decided, modulus conditions and comparisons with constants filtered exactly, finite distributions
        searched, and otherwise the finite supports unfolded or a difference of two variables filtered."""
        # Linear comparisons are evaluated exactly, e.g. 1/2*x < 2 becomes x < 4
        normalized = normalize_guard(condition, self.get_variables())
        if normalized is not None and normalized != condition:
            return self.filter(normalized)

        if isinstance(condition,
                      BinopExpr) and not has_variable(condition, None):
            return self.filter(
                BoolLitExpr(self.evaluate_condition(condition, State())))

        if isinstance(condition, BinopExpr) and not self._find_symbols(
                engine_text_of(condition.lhs)) | self._find_symbols(engine_text_of(
            condition.rhs)) <= (self.get_variables()
                                | self.get_parameters()):
            raise ValueError(
                f"Cannot filter based on the expression {str(condition)} because it contains unknown variables"
            )

        # Modulo extractions
        modulus = modulus_condition(condition)
        if modulus is not None:
            return self._arithmetic_progression(
                engine_text_of(modulus.variable),
                engine_text_of(modulus.modulus))[modulus.remainder.value]

        # Constant expressions
        if check_is_constant_constraint(condition, self.get_parameters()):
            return self._filter_constant(condition)

        return self._filter_by_search(condition)

    def _filter_by_search(self, condition: Expr) -> PGF:
        """The filter of any other condition: by exhaustive search on a finite distribution, else by unfolding the
        finite supports of its variables or, for two variables with infinite supports, as a difference
        (:meth:`_filter_difference`)."""
        # all other conditions given that the Generating Function is finite (exhaustive search)
        if self.is_finite():
            return self._exhaustive_search(condition)

        # Worst case: infinite Generating function and  non-standard condition.
        # Here we try marginalization and hope that the marginal is finite so we can do
        # exhaustive search again. If this is not possible, we raise an NotComputableException
        try:
            expression = self._explicit_state_unfolding(condition)
        except (NotImplementedError, NonFiniteMarginalError) as error:
            # two variables with infinite support: exact if they are independent (#243). The filter is
            # an identity of formal power series, so it also applies to measures with infinite mass,
            # where the case split over values fails (#366)
            return self._filter_difference(condition, error)
        return self.filter(expression)

    def _filter_difference(self, condition: Expr,
                           error: NotImplementedError | NonFiniteMarginalError) -> PGF:
        """
        Filters a comparison of two variables such as `x < y + c` exactly if the generating
        function is a product of a rational function in x and a rational function in y (given the
        other variables, see :mod:`prodigy.distribution.pgf.rules.independent_comparison`). Raises a
        :class:`ComparisonNotImplemented` naming the reason for every other comparison of several
        variables, and `error` (from the unfolding of finite supports) for all other conditions.
        """
        comparison = difference_comparison(condition, self.get_variables())
        if isinstance(comparison, BoolLitExpr):
            return self.filter(comparison)
        if comparison is None:
            if isinstance(error, NotImplementedError) and isinstance(condition, BinopExpr) \
                    and len(variables_of(condition) & self.get_variables()) >= 2:
                raise ComparisonNotImplemented(engine_text_of(condition), NOT_A_DIFFERENCE) from error
            raise error
        function = self._sympy_expression()
        arguments = (comparison.x, comparison.y, comparison.relation, comparison.shift)
        try:
            filtered = self._from_sympy_expression(filter_difference(function, *arguments, engine_text_of(condition)))
        except ComparisonNotImplemented as not_independent:
            if not_independent.reason != DEPENDENT:
                raise
            # dependent variables: the residue method, whose result may be algebraic (not rational)
            filtered = self._from_sympy_expression(filter_difference_algebraic(
                function, *arguments, self.get_variables(), engine_text_of(condition)))
        return self - filtered if comparison.complement else filtered

    def without_computed_terms(self, expression: Expr) -> Tuple[PGF, Expr, List[str]]:
        dist, expression, fresh, _ = self._replace_computed_terms(expression)
        return dist, expression, fresh

    def _replace_computed_terms(self, expression: Expr) -> Tuple[PGF, Expr, List[str], Dict[str, str]]:
        """:meth:`without_computed_terms`, and the program text of each fresh variable's term (#425)."""
        dist: PGF = self._pgf()
        fresh: List[str] = []
        names: Dict[str, str] = {}
        term = computed_term(expression, self.get_variables())
        while term is not None:
            constant = constant_value(term)
            if constant is not None:
                replacement: Expr = NatLitExpr(int(constant))
            else:
                name = dist.get_fresh_variable()
                dist = dist.set_variables(*dist.get_variables(), name).update(
                    BinopExpr(Binop.EQ, VarExpr(name), term))
                fresh.append(name)
                # a term inside this one was replaced before: name it as written, too
                names[name] = engine_text_of(in_programs_terms(ValueError(spelled(term)), names))
                replacement = VarExpr(name)
            expression = replace_term(expression, term, replacement)
            term = computed_term(expression, dist.get_variables())
        return dist, expression, fresh, names

    def _explicit_state_unfolding(self, condition: Expr) -> BinopExpr:
        """
        Checks whether one side of the condition has only finitely many valuations and explicitly creates a new
        condition which is the disjunction of each individual evaluations.

        :param condition: The condition to unfold.

        :return: The disjunction condition of explicitly encoded state conditions.
        """
        if not isinstance(condition, BinopExpr):
            raise NotImplementedError(
                f"The guard {condition} is not implemented on infinite generating functions: guards "
                "built from comparisons of variables and numbers are")
        # The side whose variables take finitely many values is unfolded: the right-hand side if
        # possible (`left_side_original`: the left-hand side stays in the condition), else the left.
        # Only the values of the side matter, not their mass, so its finite support is unfolded
        # (#249), not its marginal: a marginal that is not a finite measure (infinite mass) says
        # nothing about these values (e.g. `zoo`, a constant, would claim the single value 0, #242),
        # and in the marginal of a signed measure the mass of a value can cancel to 0.
        marginal: Optional[PGF] = None
        for left_side_original, side in ((True, condition.rhs), (False, condition.lhs)):
            expr: str = engine_text_of(side)
            syms = self._find_symbols(expr)
            if not syms:
                continue
            marginal = self.finite_support(*syms)
            if marginal is not None:
                break
        if marginal is None:
            for side in (condition.rhs, condition.lhs):
                syms = self._find_symbols(engine_text_of(side))
                if syms:
                    # raises NonFiniteMarginalError on a measure with infinite mass; nothing is computed
                    # from the marginal, so one that is only not proven to converge (#393) does not keep
                    # the guard from the formal comparison (`_filter_difference`)
                    self.checked_marginal(*syms, proof=False)
            # We are not able to marginalize into a finite amount of states! -> FAIL filtering.
            raise NotImplementedError(
                f"The guard {condition} is not implemented on infinite generating functions: neither side "
                "takes finitely many values, so the guard cannot be unfolded into finitely many cases")

        # Now we know that `expr` can be instantiated with finitely many states.
        # We generate these explicit state.
        state_expressions: List[BinopExpr] = []

        # Compute for all states the explicit condition checking that specific valuation.
        for _, state in marginal:

            # Evaluate the current expression
            evaluated_expr = self.evaluate(expr, state)

            # create the equalities for each variable, value pair in a given state
            # i.e., {x:1, y:0, c:3} -> [x=1, y=0, c=3]
            encoded_state = state_to_equality_expression(state.valuations)

            # Create the equality which assigns the original side the anticipated value.
            other_side_expr = BinopExpr(condition.operator, condition.lhs,
                                        NatLitExpr(int(evaluated_expr)))
            if not left_side_original:
                other_side_expr = BinopExpr(condition.operator,
                                            NatLitExpr(int(evaluated_expr)),
                                            condition.rhs)

            state_expressions.append(
                BinopExpr(Binop.AND, encoded_state, other_side_expr))

        # Get all individual conditions and make one big disjunction.
        return functools.reduce(
            lambda left, right: BinopExpr(
                operator=Binop.OR, lhs=left, rhs=right), state_expressions)
