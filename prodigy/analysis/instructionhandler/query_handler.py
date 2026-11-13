from __future__ import annotations

import logging
from typing import get_args, Callable, Optional, Set, Union, Sequence

import sympy

from prodigy.pgcl import Instr, Query, ExpectationInstr, VarianceInstr, CovarianceInstr, VarExpr, NatLitExpr, \
    RealLitExpr, BinopExpr, Binop, \
    UnopExpr, Unop, BoolLitExpr, ProbabilityQueryInstr, PlotInstr, PrintInstr, OptimizationQuery, OptimizationType, Expr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import (QueryResult, describe_distribution, finish_query, is_non_finite,
                                     is_monotone_nonnegative, joint_table, show, sym, tidy, univariate_table)
from prodigy.analysis.session import AnalysisSession, current_session
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.distribution import NonFiniteMarginalError
from prodigy.util.logger import log_setup
from prodigy.util.plotter import Plotter

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


class QueryHandler(InstructionHandler):
    """Answers queries. Results are printed (CLI) and booked in the current session, where
    they get sound intervals (see :mod:`prodigy.analysis.result`)."""

    @staticmethod
    def compute(
            instruction: Union[Instr, Sequence[Instr]], prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, get_args(Query), 'QueryHandler')
        assert isinstance(instruction, get_args(Query))
        session = current_session()
        recorded = len(session.queries)
        try:
            return QueryHandler.__answer(instruction, prog_info, distribution, error_prob, config, session)
        except (NotImplementedError, NonFiniteMarginalError) as exc:
            # A query does not change the distribution: one that cannot be computed fails only
            # itself, and the distribution and the other queries are kept (#306).
            if len(session.queries) != recorded:
                raise
            QueryHandler.__record_failure(instruction, exc, session)
            return distribution, error_prob

    @staticmethod
    def __record_failure(instruction: Instr, exc: Exception, session: AnalysisSession) -> None:
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
                            error={"type": type(exc).__name__, "message": message})
        session.record_query(query)
        where = f" (line {query.line})" if query.line is not None else ""
        session.warn(f"`{session.spell(instruction).strip()}`{where} could not be computed; the distribution "
                     "and the other queries are not affected.")

    @staticmethod
    def __answer(instruction: Instr, prog_info: ProgramInfo, distribution: Distribution,
                 error_prob: Distribution, config: ForwardAnalysisConfig,
                 session: AnalysisSession) -> tuple[Distribution, Distribution]:
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
            result = session.loop_query_value("?Ex", str(expression), distribution)
            if result is None:
                # floor divisions and subtractions (monus) are computed exactly as in an assignment
                result = distribution.get_expected_value_of(expression)

            print(f"Expected value: {result}")
            session.record_query(finish_query(
                session, QueryResult("expectation", session.spell(expression), value=result),
                monotone=is_monotone_nonnegative(expression, prog_info.program), probability=False,
                known=session.loop_query_bounds("?Ex", str(expression))))
            return distribution, error_prob

        # User wants the variance or covariance of expressions
        elif isinstance(instruction, (VarianceInstr, CovarianceInstr)):
            QueryHandler.__query_spread(instruction, distribution, session)
            return distribution, error_prob

        # User wants to compute a marginal, or the probability of a condition.
        elif isinstance(instruction, ProbabilityQueryInstr):
            return QueryHandler.__query_probability_of(instruction.expr,
                                                       distribution, session), \
                error_prob

        # User wants to Plot something
        elif isinstance(instruction, PlotInstr):
            return QueryHandler.__query_plot(instruction, distribution, session), \
                error_prob

        # User wants to print the current distribution.
        elif isinstance(instruction, PrintInstr):
            print(distribution)
            query = QueryResult("print", None, value=str(distribution))
            if session.collect_details:
                query.data = describe_distribution(distribution, limit=0, expectations=False)
            query.value_float = None
            query.bound = session.bound_kind
            session.record_query(query)
            return distribution, error_prob

        elif isinstance(instruction, OptimizationQuery):
            return QueryHandler.__query_optimization(instruction, distribution,
                                                     config, session), \
                error_prob

        else:
            raise SyntaxError("This should not happen.")

    @staticmethod
    def __query_spread(instr: Union[VarianceInstr, CovarianceInstr], dist: Distribution,
                       session: AnalysisSession) -> None:
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
                             config: ForwardAnalysisConfig, session: AnalysisSession):
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
    def __query_plot(instr: PlotInstr, dist: Distribution, session: AnalysisSession) -> Distribution:
        # Gather the variables to plot.
        variables = [instr.var_1.var]
        variables += [instr.var_2.var] if instr.var_2 else []

        if not session.provider.interactive:
            # Front ends draw the plot themselves from the recorded coefficients.
            session.record_query(QueryHandler.__plot_data(instr, variables, dist, session))
            return dist

        # User can specify either a probability threshold or a number of terms which are shown in the histogram.
        if instr.prob:
            # User chose a probability or oo (meaning, try to compute the whole histogram).
            if instr.prob.is_infinite():
                Plotter.plot(dist, *variables, threshold=str(1))
            else:
                Plotter.plot(dist, *variables, threshold=str(instr.prob))

        elif instr.term_count:
            # User has chosen a term limit.
            Plotter.plot(dist, *variables, threshold=instr.term_count.value)
            # User did neither specify a term limit nor a threshold probability. We try an iterative approach...
        else:
            p = .1
            inc = .1
            while True:
                Plotter.plot(dist, *variables, threshold=str(p))
                p = p + inc if p + inc < 1 else 1
                if not session.provider.continue_plot(p):
                    break
        return dist

    @staticmethod
    def __plot_data(instr: PlotInstr, variables, dist: Distribution, session: AnalysisSession) -> QueryResult:
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
                                     residual=session.residual)
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
                               dist: Distribution, session: AnalysisSession) -> Distribution:
        # Marginal computation
        if isinstance(expression, VarExpr):
            marginal = dist.marginal(expression, method=MarginalType.INCLUDE)
            print(f"Marginal distribution of {session.spell(expression)}: {marginal}")
            query = QueryResult("marginal", session.spell(expression), value=str(marginal), bound=session.bound_kind)
            if session.collect_details:
                query.data = {"marginal": univariate_table(dist, expression.var, session.coefficient_limit,
                                                           kind=session.bound_kind,
                                                           residual=session.residual)}
            session.record_query(query)
        # Probability of condition computation.
        else:
            # the loop strategy right before may know the exact value already (e.g. from its certificate)
            prob = session.loop_query_value("?Pr", str(expression), dist)
            if prob is None:
                prob = filter_by_cases(dist, expression).get_probability_mass()
            text = session.spell(expression)
            print(f"Probability of {text}: {prob}")
            session.record_query(finish_query(session, QueryResult("probability", text, value=prob),
                                              monotone=True, probability=True,
                                              known=session.loop_query_bounds("?Pr", str(expression))))
        return dist


# The most values of one variable that a case split (see :func:`filter_by_cases`) enumerates
CASE_SPLIT_LIMIT = 1000


def filter_by_cases(dist: Distribution, condition: Expr) -> Distribution:
    """
    ``dist.filter(condition)``; if the engine cannot filter on `condition` (e.g. ``x + y < 3`` where
    ``y`` has infinite support), the condition is split into cases over the values of a variable
    with finite support: the filter is the sum over the values ``v`` of ``x`` of the filter of
    ``x == v`` and ``condition[x := v]`` (#306). The cases are disjoint, so the sum is exact. Raises
    the engine's NotImplementedError if no variable of the condition has a finite support.
    """
    try:
        return dist.filter(condition)
    except NotImplementedError as error:
        names = sorted(_variables_of(condition) & set(map(str, dist.get_variables())))
        for name in names:
            support = dist.finite_support(name)
            if support is None:
                continue
            values = sorted({int(state.valuations[name]) for _, state in support})
            if len(values) > CASE_SPLIT_LIMIT:
                continue
            cases = [_substitute(condition, name, value) for value in values]
            if any(case is None for case in cases):
                break
            total: Optional[Distribution] = None
            for value, case in zip(values, cases):
                assert case is not None
                part = dist.filter(BinopExpr(Binop.EQ, VarExpr(name), NatLitExpr(value)))
                filtered = filter_by_cases(part, case)
                total = filtered if total is None else total + filtered
            if total is not None:
                return total
        raise error


def _variables_of(expr: Expr) -> Set[str]:
    if isinstance(expr, VarExpr):
        return {expr.var}
    if isinstance(expr, BinopExpr):
        return _variables_of(expr.lhs) | _variables_of(expr.rhs)
    if isinstance(expr, UnopExpr):
        return _variables_of(expr.expr)
    return set()


def _substitute(expr: Expr, name: str, value: int) -> Optional[Expr]:
    """`expr` with the variable `name` replaced by `value`, or None for an expression it does not know."""
    if isinstance(expr, VarExpr):
        return NatLitExpr(value) if expr.var == name else expr
    if isinstance(expr, (NatLitExpr, RealLitExpr, BoolLitExpr)):
        return expr
    if isinstance(expr, BinopExpr):
        lhs, rhs = _substitute(expr.lhs, name, value), _substitute(expr.rhs, name, value)
        return None if lhs is None or rhs is None else BinopExpr(expr.operator, lhs, rhs)
    if isinstance(expr, UnopExpr):
        inner = _substitute(expr.expr, name, value)
        return None if inner is None else UnopExpr(expr.operator, inner)
    return None
