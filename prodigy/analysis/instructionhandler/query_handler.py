from __future__ import annotations

import logging
from typing import get_args, Callable, Union, Sequence

from prodigy.pgcl import Instr, Query, ExpectationInstr, VarExpr, NatLitExpr, RealLitExpr, BinopExpr, Binop, \
    UnopExpr, Unop, ProbabilityQueryInstr, PlotInstr, PrintInstr, OptimizationQuery, OptimizationType, Expr

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler import _assume
from prodigy.analysis.instructionhandler.instruction_handler import InstructionHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.result import (QueryResult, describe_distribution, finish_query,
                                     is_monotone_nonnegative, joint_table, sym, univariate_table)
from prodigy.analysis.session import AnalysisSession, current_session
from prodigy.distribution import Distribution, MarginalType
from prodigy.util.logger import log_setup
from prodigy.util.plotter import Plotter

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


class QueryHandler(InstructionHandler):
    """Answers queries. Results are printed (CLI) and booked in the current session, where
    they get sound intervals (see :mod:`prodigy.analysis.result`)."""

    @staticmethod
    def compute(
            instruction: Instr, prog_info: ProgramInfo,
            distribution: Distribution, error_prob: Distribution,
            config: ForwardAnalysisConfig,
            analyzer: Callable[
                [Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution, ForwardAnalysisConfig],
                tuple[Distribution, Distribution]
            ]
    ) -> tuple[Distribution, Distribution]:
        _assume(instruction, get_args(Query), 'QueryHandler')
        session = current_session()

        # User wants to compute an expected value of an expression
        if isinstance(instruction, ExpectationInstr):
            expression = instruction.expr
            if isinstance(expression, (VarExpr, NatLitExpr, RealLitExpr)):
                result = distribution.get_expected_value_of(expression)
            elif isinstance(expression, BinopExpr):
                if expression.operator in (Binop.PLUS, Binop.MINUS,
                                           Binop.TIMES):
                    result = distribution.get_expected_value_of(expression)
                else:
                    raise SyntaxError("Expression cannot be a condition.")
            elif isinstance(expression,
                            UnopExpr) and expression.operator == Unop.NEG:
                result = distribution.get_expected_value_of(expression)
            else:
                raise SyntaxError("Expression has wrong format.")

            print(f"Expected value: {result}")
            session.record_query(finish_query(
                session, QueryResult("expectation", str(expression), value=result),
                monotone=is_monotone_nonnegative(expression, prog_info.program), probability=False))
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
        query = QueryResult("optimization", str(instr.expr),
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
            print(f"Marginal distribution of {expression}: {marginal}")
            query = QueryResult("marginal", str(expression), value=str(marginal), bound=session.bound_kind)
            if session.collect_details:
                query.data = {"marginal": univariate_table(dist, expression.var, session.coefficient_limit,
                                                           kind=session.bound_kind,
                                                           residual=session.residual)}
            session.record_query(query)
        # Probability of condition computation.
        else:
            sat_part = dist.filter(expression)
            prob = sat_part.get_probability_mass()
            print(f"Probability of {expression}: {prob}")
            session.record_query(finish_query(session, QueryResult("probability", str(expression), value=prob),
                                              monotone=True, probability=True))
        return dist
