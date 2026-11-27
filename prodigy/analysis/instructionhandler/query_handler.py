from __future__ import annotations

import logging
from typing import Callable, Optional, Union, Sequence

import sympy

from prodigy.language import Instr, Query, ExpectationInstr, VarianceInstr, CovarianceInstr, VarExpr, NatLitExpr, \
    RealLitExpr, BinopExpr, Binop, \
    UnopExpr, Unop, ProbabilityQueryInstr, PlotInstr, PrintInstr, OptimizationQuery, OptimizationType, Expr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.loops.api import Bound
from prodigy.analysis.values import (QueryResult, describe_distribution, error_type, finish_query, is_non_finite,
                                     is_monotone_nonnegative, joint_table, show, sym, tidy, univariate_table,
                                     with_upper)
from prodigy.analysis.context import AnalysisContext
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.util.logger import log_setup

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


def handle(frame: Frame, instruction: Query, state: ProgramState) -> ProgramState:
    """Answers a query. Results are printed (CLI) and booked in the context, where they get sound intervals (see
    :mod:`prodigy.analysis.result`). A query does not change the state."""
    session = frame.context
    recorded = len(session.queries)
    try:
        _Queries.answer(instruction, frame.program, state.dist, frame.config, session)
    except (NotImplementedError, NonFiniteMarginalError, ArithmeticError, ValueError) as exc:
        # A query does not change the distribution: one that cannot be computed fails only
        # itself, and the distribution and the other queries are kept (#306, #316). This includes an
        # engine's arithmetic error, e.g. GiNaC's log(0) for Ex[x] of a logarithmic generating function.
        if len(session.queries) != recorded:
            raise
        _Queries.record_failure(instruction, exc, session)
    return state


class _Queries:
    """The answers of the query kinds."""

    @staticmethod
    def record_failure(instruction: Instr, exc: Exception, session: AnalysisContext) -> None:
        message = str(exc) or type(exc).__name__
        if isinstance(instruction, ExpectationInstr):
            kind, expr = "expectation", session.spell(instruction.expr)
        elif isinstance(instruction, VarianceInstr):
            kind, expr = "variance", session.spell(instruction.expr)
        elif isinstance(instruction, CovarianceInstr):
            kind, expr = "covariance", f"{session.spell(instruction.lhs)}, {session.spell(instruction.rhs)}"
        elif isinstance(instruction, ProbabilityQueryInstr):
            kind = "marginal" if isinstance(instruction.expr, VarExpr) else "probability"
            expr = session.spell(instruction.expr)
        elif isinstance(instruction, OptimizationQuery):
            kind, expr = "optimization", session.spell(instruction.expr)
        elif isinstance(instruction, PlotInstr):
            kind, expr = "plot", ", ".join([instruction.var_1.var] + ([instruction.var_2.var] if instruction.var_2
                                                                       else []))
        else:
            kind, expr = "print", None
        print(f"Query {expr if expr is not None else kind} could not be computed: {message}")
        query = QueryResult(kind, expr, bound=Bound.UNKNOWN, note=f"Not computed: {message}",
                            error={"type": error_type(exc), "message": message})
        session.record_query(query)
        where = f" (line {query.line})" if query.line is not None else ""
        session.warn(f"`{session.spell(instruction).strip()}`{where} could not be computed; the distribution "
                     "and the other queries are not affected.")

    @staticmethod
    def answer(instruction: Instr, prog_info: ProgramInfo, distribution: Distribution,
               config: ForwardAnalysisConfig, session: AnalysisContext) -> None:
        # User wants to compute an expected value of an expression
        if isinstance(instruction, ExpectationInstr):
            expression = instruction.expr
            if isinstance(expression, BinopExpr):
                if expression.operator not in (Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.FLOOR_DIVIDE,
                                               Binop.POWER):
                    raise SyntaxError("Expression cannot be a condition.")
            elif not (isinstance(expression, (VarExpr, NatLitExpr, RealLitExpr))
                      or (isinstance(expression, UnopExpr) and expression.operator == Unop.NEG)):
                raise SyntaxError("Expression has wrong format.")
            # the loop strategy right before may know the exact value already (e.g. from its certificate)
            result = session.loop_query_value("?Ex", expression, distribution)
            if result is None:
                # floor divisions and subtractions (monus) are computed exactly as in an assignment
                result = distribution.get_expected_value_of(expression)

            print(f"Expected value: {result}")
            monotone = is_monotone_nonnegative(expression, prog_info.program)
            known = session.loop_query_bounds("?Ex", expression)
            if monotone and (known is None or known[1] is None):
                known = with_upper(known, _upper_value(session, lambda upper: upper.get_expected_value_of(expression)))
            session.record_query(finish_query(
                session, QueryResult("expectation", session.spell(expression), value=result),
                monotone=monotone, probability=False, known=known))
            return

        # User wants the variance or covariance of expressions
        elif isinstance(instruction, (VarianceInstr, CovarianceInstr)):
            _Queries.__query_spread(instruction, distribution, session)
            return

        # User wants to compute a marginal, or the probability of a condition.
        elif isinstance(instruction, ProbabilityQueryInstr):
            _Queries.__query_probability_of(instruction.expr, distribution, session)

        # User wants to Plot something
        elif isinstance(instruction, PlotInstr):
            _Queries.__query_plot(instruction, distribution, session)

        # User wants to print the current distribution.
        elif isinstance(instruction, PrintInstr):
            print(distribution)
            query = QueryResult("print", None, value=str(distribution))
            if session.collect_details:
                query.data = describe_distribution(distribution, limit=0, expectations=False)
            query.value_float = None
            query.bound = session.bound_kind
            session.record_query(query)
            return

        elif isinstance(instruction, OptimizationQuery):
            _Queries.__query_optimization(instruction, distribution, config, session)

        else:
            raise SyntaxError("This should not happen.")

    @staticmethod
    def __query_spread(instr: Union[VarianceInstr, CovarianceInstr], dist: Distribution,
                       session: AnalysisContext) -> None:
        """``Var[e]`` = ``Ex[e * e] - Ex[e]^2`` and ``Cov[a, b]`` = ``Ex[a * b] - Ex[a] * Ex[b]``,
        each ``Ex`` an exact derivative of the generating function. Only reported for a distribution
        that is exact and has mass 1: a variance of a lower or upper bound is not a bound, and the
        moments of a sub-distribution (error, observe, non-termination) are not normalised."""
        if isinstance(instr, VarianceInstr):
            kind, name, operands = "variance", "Var", [instr.expr, instr.expr]
        else:
            kind, name, operands = "covariance", "Cov", [instr.lhs, instr.rhs]
        for operand in operands:
            if isinstance(operand, BinopExpr):
                if operand.operator not in (Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.FLOOR_DIVIDE,
                                            Binop.POWER):
                    raise SyntaxError("Expression cannot be a condition.")
            elif not (isinstance(operand, (VarExpr, NatLitExpr, RealLitExpr))
                      or (isinstance(operand, UnopExpr) and operand.operator == Unop.NEG)):
                raise SyntaxError("Expression has wrong format.")
        text = ", ".join(session.spell(operand) for operand in (operands if kind == "covariance" else operands[:1]))
        if session.bound_kind != Bound.EXACT:
            raise NotImplementedError(
                f"{name}[{text}] is only available for an exact distribution; this one is only a "
                f"{session.bound_kind} bound, and a {kind} of a bound is not a bound.")
        mass = sym(dist.get_probability_mass())
        if sympy.simplify(mass - 1) != 0:
            raise NotImplementedError(
                f"{name}[{text}] needs a distribution of mass 1, but the mass is {show(mass)} "
                "(an error, `observe` or non-termination removed some). Use Ex[…] for unnormalised moments.")
        first, second = operands
        mean_first = sym(dist.get_expected_value_of(first))
        mean_second = mean_first if first is second else sym(dist.get_expected_value_of(second))
        mixed = sym(dist.get_expected_value_of(BinopExpr(Binop.TIMES, first, second)))
        means = [mean_first] if first is second else [mean_first, mean_second]
        for mean in means:
            if is_non_finite(mean):
                raise NonFiniteMarginalError(
                    f"{name}[{text}] is undefined: the expected value of an operand is {show(mean)}.")
        if is_non_finite(mixed, infinity=True) or mixed == sympy.nan:
            raise NonFiniteMarginalError(f"{name}[{text}] is undefined: a moment is not a number.")
        if mixed == sympy.oo:
            if kind == "covariance":
                raise NonFiniteMarginalError(
                    f"{name}[{text}] is undefined: the expected value of the product is infinite.")
            value = sympy.oo  # an integrable variable with an infinite second moment
        else:
            value = tidy(mixed - mean_first * mean_second)
        print(f"{'Variance' if kind == 'variance' else 'Covariance'}: {show(value)}")
        query = QueryResult(kind, text, value=show(value))
        query.note = "Exact: derivatives of the generating function at 1."
        session.record_query(finish_query(session, query, monotone=False, probability=False))

    @staticmethod
    def __query_optimization(instr: OptimizationQuery, dist: Distribution,
                             config: ForwardAnalysisConfig, session: AnalysisContext) -> Distribution:
        logger.debug(
            "Computing the optimal value for parameter %s in order to %s the distribution %s with respect to %s",
            instr.parameter, 'maximize' \
                if instr.type == OptimizationType.MAXIMIZE else 'minimize', dist, instr.expr)
        result = config.optimizer.optimize(instr.expr,
                                           dist,
                                           instr.parameter,
                                           method=instr.type)
        if not result:
            print("No solutions could be found.")
        elif len(result) == 1:
            print(f"The optimal value is at {instr.parameter}={result[0]}.")
        else:
            print(
                f"The maximal values can be achieved choosing the parameter {instr.parameter} in {result}."
            )
        query = QueryResult("optimization", session.spell(instr.expr),
                            value=", ".join(str(r) for r in result) if result else None)
        query.data = {"parameter": str(instr.parameter),
                      "goal": "maximize" if instr.type == OptimizationType.MAXIMIZE else "minimize",
                      "solutions": [str(r) for r in result] if result else []}
        query.bound = session.bound_kind
        query.note = "Optimisation over the computed distribution."
        session.record_query(query)
        return dist

    @staticmethod
    def __query_plot(instr: PlotInstr, dist: Distribution, session: AnalysisContext) -> Distribution:
        # Gather the variables to plot.
        variables = [instr.var_1.var]
        variables += [instr.var_2.var] if instr.var_2 else []

        # the analysis only records the data; the front ends draw it (GUI in the browser, CLI with prodigy.plotting)
        session.record_query(_Queries.__plot_data(instr, variables, dist, session))
        return dist

    @staticmethod
    def __plot_data(instr: PlotInstr, variables: Sequence[str], dist: Distribution,
                    session: AnalysisContext) -> QueryResult:
        limit = session.coefficient_limit
        threshold = None
        if instr.term_count:
            limit = int(instr.term_count.value)
        elif instr.prob and not instr.prob.is_infinite():
            threshold = sym(str(instr.prob))
            limit = max(limit, 100)
        elif instr.prob and instr.prob.is_infinite():
            limit = max(limit, 200)
        query = QueryResult("plot", ", ".join(variables), bound=session.bound_kind)
        if len(variables) == 1:
            table = univariate_table(dist, variables[0], limit, kind=session.bound_kind,
                                     residual=session.residual, upper=session.upper_distribution)
            if threshold is not None:
                kept, total = [], 0
                for row in table["rows"]:
                    kept.append(row)
                    total += sym(row["prob"])
                    if total >= threshold:
                        break
                table["rows"] = kept
            query.data = {"variables": variables, "marginal": table}
        else:
            marginal = dist.marginal(*variables)
            query.data = {"variables": variables, "joint": joint_table(marginal, limit * limit)}
            if query.data["joint"] is None:
                query.note = "Joint plots of infinite-support distributions are not available yet."
        return query

    @staticmethod
    def __query_probability_of(expression: Expr,
                               dist: Distribution, session: AnalysisContext) -> Distribution:
        # Marginal computation
        if isinstance(expression, VarExpr):
            marginal = dist.marginal(expression, method=MarginalType.INCLUDE)
            print(f"Marginal distribution of {session.spell(expression)}: {marginal}")
            query = QueryResult("marginal", session.spell(expression), value=str(marginal), bound=session.bound_kind)
            if session.collect_details:
                query.data = {"marginal": univariate_table(dist, expression.var, session.coefficient_limit,
                                                           kind=session.bound_kind,
                                                           residual=session.residual,
                                                           upper=session.upper_distribution)}
            session.record_query(query)
        # Probability of condition computation.
        else:
            # the loop strategy right before may know the exact value already (e.g. from its certificate)
            prob = session.loop_query_value("?Pr", expression, dist)
            if prob is None:
                prob = dist.get_probability_of(expression)
            text = session.spell(expression)
            print(f"Probability of {text}: {prob}")
            known = session.loop_query_bounds("?Pr", expression)
            if known is None or known[1] is None:
                known = with_upper(known, _upper_value(session, lambda upper: _probability(upper, expression)))
            session.record_query(finish_query(session, QueryResult("probability", text, value=prob),
                                              monotone=True, probability=True, known=known))
        return dist


def _probability(dist: Distribution, condition: Expr) -> str:
    """The probability of ``condition`` under ``dist``, as for a ``?Pr`` query."""
    return dist.get_probability_of(condition)


def _upper_value(session: AnalysisContext, evaluate: Callable[[Distribution], str]) -> Optional[sympy.Expr]:
    """A monotone query (a probability, or the expectation of a non-negative expression) evaluated on the
    session's proven upper bound (:mod:`prodigy.analysis.upper_track`): an upper bound of its exact value, since
    the query is a positive linear functional. ``None`` without an upper track, or if the value is not a finite
    non-negative number (the bound is only an addition). Callers skip it for a query that the loop right before
    already bounds from above (:meth:`AnalysisContext.loop_query_bounds`): that bound comes from the same
    certificate, evaluated exactly on its template, and evaluating the closed form again can be expensive."""
    upper = session.upper_distribution
    if upper is None:
        return None
    try:
        value = sym(evaluate(upper))
        if value.is_number and value.is_real and value.is_finite and bool(value >= 0):
            return tidy(value)
    except Exception:  # pylint: disable=broad-except  # the bound is only an addition
        logger.debug("A query on the upper bound failed", exc_info=True)
    return None
