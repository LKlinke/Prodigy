# pylint: disable=protected-access
from __future__ import annotations

import functools
import operator
from typing import Callable, Generator, List, Set, Union, get_args, Tuple

import sympy
from probably.pgcl import (BernoulliExpr, Binop, BinopExpr,  # type: ignore
                           BoolLitExpr, DistrExpr, DUniformExpr, Expr,
                           GeometricExpr, IidSampleExpr, NatLitExpr,
                           PoissonExpr, RealLitExpr, Unop, UnopExpr, VarExpr,
                           Walk, walk_expr, Program, NatType, RealType)
from probably.pgcl.parser import parse_expr
from probably.pgcl.syntax import check_is_linear_expr  # type: ignore
from probably.util.ref import Mut
from sympy.assumptions.assume import global_assumptions

# TODO Implement these checks in probably
from prodigy.distribution import (CommonDistributionsFactory, Distribution,
                                  DistributionParam, MarginalType, State)
from prodigy.pgcl.pgcl_checks import (check_is_constant_constraint,
                                      check_is_modulus_condition)
from prodigy.util.logger import Style, log_setup, logging

logger = log_setup(__name__, logging.DEBUG, file="GF_operations.log")


def _term_generator(function: sympy.Poly):
    assert isinstance(function,
                      sympy.Poly), "Terms can only be generated for finite GF"
    poly = function
    while poly.as_expr() != 0:
        yield poly.EC(), poly.EM().as_expr()
        poly -= poly.EC() * poly.EM().as_expr()


class GeneratingFunction(Distribution):
    """
    This class represents a generating function. It wraps the sympy library.
    A GeneratingFunction object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """

    use_simplification = False
    use_latex_output = False

    # ==================================== CONSTRUCTORS ====================================

    def __init__(self,
                 function: Union[str, sympy.Expr] = "",
                 *variables: Union[str, sympy.Symbol],
                 preciseness: float = 1.0,
                 closed: bool = None,
                 finite: bool = None):

        # Set the basic information.
        self._function: sympy.Expr = sympy.S(str(function), rational=True)
        self._preciseness = sympy.S(str(preciseness), rational=True)

        # Set variables and parameters
        self._variables: Set[sympy.Symbol] = self._function.free_symbols
        self._parameters: Set[sympy.Symbol] = set()
        if variables:
            self._variables = self._variables.union(
                map(lambda v: sympy.S(str(v)), filter(lambda v: v != "", variables)))
            self._parameters = self._variables.difference(
                map(lambda v: sympy.S(str(v)), filter(lambda v: v != "", variables)))
            self._variables -= self._parameters

        for var in self._variables:
            global_assumptions.add(sympy.Q.nonnegative(var))
        for param in self._parameters:
            global_assumptions.add(sympy.Q.positive(param))

        # Do closed form and finiteness heuristics
        self._is_closed_form = closed if closed else not self._function.is_polynomial(
            *self._variables)
        self._is_finite = finite if finite else self._function.ratsimp(
        ).is_polynomial(*self._variables)

    def _update_modulo_expression(self,
                                  expression: Expr) -> GeneratingFunction:

        assert isinstance(expression, BinopExpr) and expression.operator == Binop.MODULO, \
            f"This function can only be called on Modulo Expressions, got {expression}."

        # Currently only unnested modulo expression are supported. Maybe update this at some time.
        if isinstance(expression.lhs, VarExpr) and isinstance(
                expression.rhs, NatLitExpr):
            result = GeneratingFunction("0",
                                        *self._variables,
                                        preciseness=1,
                                        closed=True,
                                        finite=True)
            progression_list = self._arithmetic_progression(
                str(expression.lhs), str(expression.rhs))
            for i in range(expression.rhs.value):
                func = progression_list[i]
                result += func.marginal(
                    expression.lhs.var, method=MarginalType.EXCLUDE
                ) * GeneratingFunction(f"{expression.lhs}**{i}")
            return result
        else:
            raise NotImplementedError(
                "Nested modulo expressions are currently not supported.")

    def _update_statewise(self, expression: Expr) -> GeneratingFunction:
        """ Updates a finite distribution by evaluating the expression for each encoded state separately. """

        assert self._is_finite, f"Cannot do a statewise update for infinite Generatingfunction {self}."
        assert isinstance(expression, BinopExpr) and isinstance(expression.lhs, VarExpr), \
            f"Expression must be an assignment, was {expression}."

        result = sympy.S(0)
        for prob, state in self:
            updated_state = state.copy()
            updated_state[expression.lhs.var] = int(
                self.evaluate(str(expression.rhs), state))
            monomial = sympy.S(updated_state.to_monomial())
            result += sympy.S(prob) * monomial
        return GeneratingFunction(result,
                                  *self._variables,
                                  preciseness=self._preciseness)

    def _filter_constant_condition(self,
                                   condition: Expr) -> GeneratingFunction:
        """
        Filters out the terms that satisfy a constant condition, i.e, (var <= 5) (var > 5) (var = 5).
        :param condition: The condition to filter.
        :return: The filtered generating function.
        """

        # Normalize the condition into the format _var_ (< | <= | =) const. I.e., having the variable on the lhs.
        if isinstance(condition.rhs,
                      VarExpr) and condition.rhs.var not in self._variables:
            if condition.operator == Binop.LEQ:
                return self.filter(
                    UnopExpr(operator=Unop.NEG,
                             expr=BinopExpr(operator=Binop.LT,
                                            lhs=condition.rhs,
                                            rhs=condition.lhs)))
            elif condition.operator == Binop.LT:
                return self.filter(
                    UnopExpr(operator=Unop.NEG,
                             expr=BinopExpr(operator=Binop.LEQ,
                                            lhs=condition.rhs,
                                            rhs=condition.lhs)))
            elif condition.operator == Binop.GT:
                return self.filter(
                    BinopExpr(operator=Binop.LT,
                              lhs=condition.rhs,
                              rhs=condition.lhs))
            elif condition.operator == Binop.GEQ:
                return self.filter(
                    BinopExpr(operator=Binop.LEQ,
                              lhs=condition.rhs,
                              rhs=condition.lhs))

        if isinstance(condition.lhs,
                      VarExpr) and condition.lhs.var not in self._variables:
            if condition.operator == Binop.GT:
                return self.filter(
                    BinopExpr(operator=Binop.LT,
                              lhs=condition.rhs,
                              rhs=condition.lhs))
            elif condition.operator == Binop.GEQ:
                return self.filter(
                    BinopExpr(operator=Binop.LEQ,
                              lhs=condition.rhs,
                              rhs=condition.lhs))

        # Now we have an expression of the form _var_ (< | <=, =) _const_.
        variable = str(condition.lhs)
        constant = condition.rhs.value
        result = sympy.S(0)
        ranges = {
            Binop.LT: range(constant),
            Binop.LEQ: range(constant + 1),
            Binop.EQ: [constant]
        }

        # Compute the probabilities of the states _var_ = i where i ranges depending on the operator (< , <=, =).
        for i in ranges[condition.operator]:
            result += (self._function.diff(variable, i) / sympy.factorial(i)
                       ).limit(variable, 0, '-') * sympy.S(variable) ** i

        return GeneratingFunction(result,
                                  *self._variables,
                                  preciseness=self._preciseness,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def _explicit_state_unfolding(self, condition: Expr) -> BinopExpr:
        """
        Checks whether one side of the condition has only finitely many valuations and explicitly creates a new
        condition which is the disjunction of each individual evaluations.
        :param condition: The condition to unfold.
        :return: The disjunction condition of explicitly encoded state conditions.
        """
        expr = sympy.S(str(condition.rhs))
        marginal = self.marginal(*expr.free_symbols)

        # Marker to express which side of the equation has only finitely many interpretations.
        left_side_original = True

        # Check whether left hand side has only finitely many interpretations.
        if not marginal.is_finite():
            # Failed, so we have to check the right hand side
            left_side_original = False
            expr = sympy.S(str(condition.lhs))
            marginal = self.marginal(expr.free_symbols)

            if not marginal.is_finite():
                # We are not able to marginalize into a finite amount of states! -> FAIL filtering.
                raise NotImplementedError(
                    f"Instruction {condition} is not computable on infinite generating function"
                    f" {self._function}")

        # Now we know that `expr` can be instantiated with finitely many states.
        # We generate these explicit state.
        state_expressions: List[BinopExpr] = []

        # Compute for all states the explicit condition checking that specific valuation.
        for _, state in marginal:

            # Evaluate the current expression
            evaluated_expr = self.evaluate(str(expr), state)

            # create the equalities for each variable, value pair in a given state
            # i.e., {x:1, y:0, c:3} -> [x=1, y=0, c=3]
            encoded_state = self._state_to_equality_expression(state)

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

    @staticmethod
    def _state_to_equality_expression(state: State) -> BinopExpr:
        equalities: List[Expr] = []
        for var, val in state.items():
            equalities.append(
                BinopExpr(Binop.EQ, lhs=VarExpr(var), rhs=NatLitExpr(val)))
        if(equalities == []):
            return BoolLitExpr(False)
        return functools.reduce(
            lambda expr1, expr2: BinopExpr(Binop.AND, expr1, expr2),
            equalities, BoolLitExpr(value=True))

    def _mult_term_generator(self):
        """
            Generates terms of multivariate generating function in `grlex` order.
        """
        i = 1
        while True:
            # The easiest method is to create all monomials of until total degree `i` and sort them.
            logger.debug("Generating and sorting of new monomials")
            new_monomials = sorted(
                sympy.polys.monomials.itermonomials(self._variables, i),
                key=sympy.polys.orderings.monomial_key("grlex",
                                                       list(self._variables)))
            # instead of repeating all the monomials for higher total degrees, just cut-off the already created ones.
            if i > 1:
                new_monomials = new_monomials[
                                sympy.polys.monomials.
                                    monomial_count(len(self._variables), i - 1):]
            logger.debug("Monomial_generation done")

            # after we have the list of new monomials, create (probability, state) pais and yielding them.
            for monomial in new_monomials:
                state = {}
                if monomial == 1:
                    for var in self._variables:
                        state[var] = 0
                else:
                    state = monomial.as_expr().as_powers_dict()
                yield self._probability_of_state(state), monomial.as_expr()
            logger.debug("\t>Terms generated until total degree of %d", i)
            i += 1

    def _arithmetic_progression(self, variable: str,
                                modulus: str) -> List[GeneratingFunction]:
        """
        Creates a list of subdistributions where at list index i, the `variable` is congruent i modulo `modulus`.
        """
        # pylint: disable=invalid-name
        a = sympy.S(modulus)
        var = sympy.S(variable)
        primitive_uroot = sympy.S(f"exp(2 * {sympy.pi} * {sympy.I}/{a})")
        result = []
        for remainder in range(a):
            psum = 0
            for m in range(a):
                psum += primitive_uroot ** (-m *
                                            remainder) * self._function.subs(
                    var, (primitive_uroot ** m) * var)
            result.append(
                GeneratingFunction(f"(1/{a}) * ({psum})",
                                   *self._variables,
                                   preciseness=self._preciseness,
                                   closed=self._is_closed_form,
                                   finite=self._is_finite))
        return result

    def _limit(self, variable: Union[str, sympy.Symbol],
               value: Union[str, sympy.Expr]) -> GeneratingFunction:
        func = self._function
        if self._is_closed_form:
            print("\rComputing limit...", end="\r", flush=True)
            func = func.limit(sympy.S(variable), sympy.S(value), "-")
            # func = func.subs(sympy.S(variable), sympy.S(value))
        else:
            func = func.subs(sympy.S(variable), sympy.S(value))
        return GeneratingFunction(func,
                                  preciseness=self._preciseness,
                                  closed=self._is_closed_form,
                                  finite=func.ratsimp().is_polynomial())

    @staticmethod
    def _split_addend(addend: sympy.Expr) -> Tuple[sympy.Expr, sympy.Expr]:
        r"""
        This method assumes that the addend is given in terms of :math:`\alpha \times X^{\sigma}`, where
        :math:`\alpha \in [0,1], \sigma \in \mathbb{N}^k`.
        :param addend: the addend to split into its factor and monomial.
        :return: a tuple (factor, monomial)
        """
        if addend.free_symbols == set():
            return addend, sympy.S(1)
        else:
            factor_powers = addend.as_powers_dict()
            result = (sympy.S(1), sympy.S(1))
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
                result[str(var)] = 0
        else:
            variables_and_powers = monomial.as_powers_dict()
            for var in self._variables:
                if var in variables_and_powers.keys():
                    result[str(var)] = int(variables_and_powers[var])
                else:
                    result[str(var)] = 0
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
            if str(s_var) not in state:
                complete_state[str(s_var)] = 0

        # When its a closed form, or not finite, do the maths.
        if self._is_closed_form or not self._is_finite:
            result = self._function
            for variable, value in complete_state.items():
                result = result.diff(sympy.S(variable), value)
                result /= sympy.factorial(value)
                result = result.limit(variable, 0, "-")
            return result
        # Otherwise use poly module and simply extract the correct coefficient from it
        else:
            monomial = sympy.S(1)
            for variable, value in complete_state.items():
                monomial *= sympy.S(f"{variable} ** {value}")
            probability = self._function.as_poly(
                *self._variables).coeff_monomial(monomial)
            return probability if probability else sympy.core.numbers.Zero()

    # Distribution interface implementation

    def update(self, expression: Expr) -> GeneratingFunction:
        """ Updates the current distribution by applying the expression to itself. Currently, we are able to handle the
            following cases:
                * Modulo operations like: <<VAR>> % <<CONSTANT>>
                * Linear transformations: <<VAR>> := f(<<VARS>>) where f is a linear function.
                * Arbitrary Expressions where the current distribution has finite support.
                * Approximations to arbitrary expression where the current distribution has infinite support.
            ATTENTION: The latter two might take a while to compute as it is implemented in a brute force manner.
        """

        assert isinstance(expression, BinopExpr) and isinstance(expression.lhs, VarExpr), \
            f"Expression must be an assignment, was {expression}."

        variable: str = expression.lhs.var

        # rhs is a modulo expression
        if isinstance(expression.rhs,
                      BinopExpr) and expression.rhs.operator == Binop.MODULO:
            return self._update_modulo_expression(expression)

        # rhs is a linear expression
        variables = {}
        parameters = {}
        for var in self.get_variables():
            variables[var] = NatType(bounds=None)
        for param in self.get_parameters():
            parameters[param] = RealType()
        if check_is_linear_expr(
                context=Program(constants={},
                                variables=variables,
                                parameters=parameters,
                                declarations=[],
                                instructions=[]
                                ),
                expr=expression.rhs
        ) is None:
            return self.linear_transformation(variable, expression.rhs)

        # rhs is a non-linear expression, self is finite
        if self.is_finite():
            return self._update_statewise(expression)

        # rhs is non-linear, self is infinite support
        print(f"The assignment {expression} is not computable on {self}")
        error = sympy.S(
            input(
                "Continue with approximation. Enter an allowed relative error (0, 1.0):\t"
            ))
        if 0 < error < 1:
            return list(
                self.approximate(f"{(1 - error) * self.coefficient_sum()}")
            )[-1].update(expression)
        else:
            raise NotImplementedError(
                f"The assignment {expression} is currently not computable on {self}"
            )

    def update_iid(self, sampling_exp: IidSampleExpr,
                   variable: Union[str, VarExpr]) -> Distribution:
        assert isinstance(sampling_exp,
                          IidSampleExpr), "Not an IidSamplingExpression."

        subst_var = sampling_exp.variable
        sampling_dist = sampling_exp.sampling_dist

        if isinstance(sampling_dist, GeometricExpr):
            dist_gf = sympy.S(
                f"({sampling_dist.param}) / (1 - (1-({sampling_dist.param})) * {variable})"
            )
            result = self.marginal(variable, method=MarginalType.EXCLUDE)
            result._function = result._function.subs(
                str(subst_var), f"{subst_var} * {dist_gf}")
            return result
        if isinstance(sampling_dist, BernoulliExpr):
            dist_gf = sympy.S(
                f"{sampling_dist.param} * {variable} + (1 - ({sampling_dist.param}))"
            )
            result = self.marginal(variable, method=MarginalType.EXCLUDE)
            result._function = result._function.subs(
                str(subst_var), f"{subst_var} * ({dist_gf})")
            return result
        if isinstance(sampling_dist, PoissonExpr):
            dist_gf = sympy.S(f"exp({sampling_dist.param} * ({variable} - 1))")
            result = self.marginal(variable, method=MarginalType.EXCLUDE)
            result._function = result._function.subs(
                str(subst_var), f"{subst_var} * ({dist_gf})")
            return result
        if isinstance(sampling_dist, DUniformExpr):
            dist_gf = sympy.S(
                f"1/(({sampling_dist.end}) - ({sampling_dist.start}) + 1) * {variable}**({sampling_dist.start}) * ({variable}**(({sampling_dist.end}) - ({sampling_dist.start}) + 1) - 1) / ({variable} - 1)"
            )
            result = self.marginal(variable, method=MarginalType.EXCLUDE)
            result._function = result._function.subs(
                str(subst_var), f"{subst_var} * ({dist_gf})")
            return result

        if not isinstance(sampling_dist, get_args(DistrExpr)) and isinstance(
                sampling_dist, get_args(Expr)):
            # create distribution in correct variable:
            expr = Mut.alloc(sampling_dist)
            for ref in walk_expr(Walk.DOWN, expr):
                if isinstance(ref.val, VarExpr):
                    ref.val.var = variable
            dist_gf = sympy.S(str(expr.val))
            result = self.marginal(variable, method=MarginalType.EXCLUDE)
            result._function = result._function.subs(
                str(subst_var), f"{subst_var} * {dist_gf}")
            return result

        raise NotImplementedError(
            "Currently only geometric expressions are supported.")

    def get_expected_value_of(self, expression: Union[Expr, str]) -> str:

        expr = sympy.S(str(expression)).ratsimp().expand()
        if not expr.is_polynomial():
            raise NotImplementedError(
                "Expected Value only computable for polynomial expressions.")
        marginal = self.marginal(*expr.free_symbols,
                                 method=MarginalType.INCLUDE)
        gen_func = GeneratingFunction(expr)
        expected_value = GeneratingFunction('0')
        for prob, state in gen_func:
            tmp = marginal.copy()
            for var, val in state.items():
                for _ in range(val):
                    tmp = tmp.diff(var, 1) * GeneratingFunction(var)
                tmp = tmp._limit(var, "1")
            expected_value += GeneratingFunction(prob) * tmp
        if expected_value._function == sympy.S('oo'):
            return str(RealLitExpr.infinity())
        else:
            return str(expected_value._function)

    def copy(self, deep: bool = True) -> GeneratingFunction:
        return GeneratingFunction(str(self._function),
                                  *self._variables,
                                  preciseness=self._preciseness,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def set_variables(self, *variables: str) -> GeneratingFunction:
        if not variables:
            raise ValueError(
                "The free-variables of a distribution cannot be empty!")
        remove_dups = set(variables)
        return GeneratingFunction(self._function,
                                  *remove_dups,
                                  preciseness=self._preciseness,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def set_parameters(self, *parameters: str) -> GeneratingFunction:
        remove_dups = set(parameters)
        if self._variables.intersection(remove_dups):
            raise ValueError("A indeterminate cannot be variable and parameter at the same time.")
        result = self.copy()
        result._parameters = set(sympy.S(param) for param in remove_dups)
        return result

    def _arithmetic(
            self, other, op: Callable
    ) -> GeneratingFunction:
        """
        Computes the basic arithmetic of two generating functions.
        """
        logger.debug("Computing the %s of %s with %s", op, self, other)

        # other object is already a generating function.
        if isinstance(other, GeneratingFunction):
            # Checking for naming clashes in variables and parameters:
            if self._variables.intersection(other._parameters) or self._parameters.intersection(other._variables):
                raise ArithmeticError(
                    f"Name clash for parameters and variables in {self._variables=} {other._variables=} \t {self._parameters=} {other._parameters=}")

            # collect the meta information.
            self_coeff_sum, other_coeff_sum = self.coefficient_sum(
            ), other.coefficient_sum()
            is_closed_form = self._is_closed_form and other._is_closed_form
            is_finite = self._is_finite and other._is_finite
            preciseness = (self_coeff_sum + other_coeff_sum) / (
                    self_coeff_sum / self._preciseness + other_coeff_sum /
                    other._preciseness)  # BUG: Something not working.

            # do the actual operation
            function = op(self._function, other._function)

            # simplify the result if demanded.
            if GeneratingFunction.use_simplification:
                logger.debug("Try simplification but stay in closed form!")
                nominator, denominator = sympy.fraction(function)
                logger.info("Factoring")
                nominator = nominator.factor()
                denominator = denominator.factor()
                function = nominator / denominator
                logger.debug("Closed-form simplification result: %s", function)
                if not is_closed_form:
                    logger.debug("Try to cancel terms")
                    function = function.expand()
                    logger.debug("Canceling result: %s", function)

            # update the occuring variables and parameters
            variables = self._variables.union(other._variables)
            return GeneratingFunction(function,
                                      *variables,
                                      preciseness=preciseness,
                                      closed=is_closed_form,
                                      finite=is_finite)

        # other object is either an expression, or literal
        elif isinstance(other, (str, float, int)):
            # we try to convert this into a Generatingfunction and compute the arithmetic from there on.
            return self._arithmetic(GeneratingFunction(other), op)
        # We don't know how to do arithmetic on other types.
        else:
            raise SyntaxError(
                f"You cannot {str(op)} {type(self)} with {type(other)}.")

    def __add__(self, other) -> GeneratingFunction:
        return self._arithmetic(other, operator.add)

    def __sub__(self, other) -> GeneratingFunction:
        return self._arithmetic(other, operator.sub)

    def __mul__(self, other) -> GeneratingFunction:
        return self._arithmetic(other, operator.mul)

    def __truediv__(self, other) -> GeneratingFunction:
        logger.debug("Isn't it weird to divide Distributions?")
        return self._arithmetic(other, operator.truediv)

    def __str__(self) -> str:
        if GeneratingFunction.use_latex_output:
            return sympy.latex(self._function)
        else:
            output = f"{self._function}"
        return output

    def __repr__(self):
        return repr(self._function)

    # ====== Comparison of Generating Functions ======

    def __le__(self, other) -> bool:
        if not isinstance(other, GeneratingFunction):
            raise TypeError(
                f"Incomparable types {type(self)} and {type(other)}.")

        # if one of the generating functions is finite, we can check the relation coefficient-wise.
        if self._is_finite:
            for prob, state in self:
                if sympy.S(prob) > other._probability_of_state(state):
                    return False
            return True
        if other._is_finite:
            for prob, state in other:
                if self._probability_of_state(state) > sympy.S(prob):
                    return False
            return True
        # when both generating functions have infinite support we can only try to check whether we can eliminate
        # common terms and check whether the result is finite. If so we can do a coefficient-wise pass again.
        difference = (self._function - other._function)
        if difference.is_polynomial():
            return all(
                map(lambda x: x > 0,
                    difference.as_coefficients_dict().values()))

        # We currently do not know what to do otherwise.
        raise RuntimeError(
            "Both objects have infinite support. We cannot determine the order between them."
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeneratingFunction):
            return False
        # We rely on simplification of __sympy__ here. Thus, we cannot guarantee to detect equality when
        # simplification fails.
        return bool(self._function.equals(other._function)) \
               and self._variables == other._variables \
               and self._parameters == other._parameters

    def __iter__(self):
        """ Iterates over the generating function yielding (coefficient, state) pairs.

            For implicit (closed-form) generating functions, we generate terms via tailor expansion.
            If the generating function is even multivariate this might be costly as we try to enumerate all different
            k-tuples, for which the coefficient might be zero (this we do not know upfront).
        """
        logger.debug("iterating over %s...", self)
        if self._is_finite:
            if self._is_closed_form:
                func = self._function.expand().ratsimp().as_poly(
                    *self._variables)
            else:
                if not self._variables:
                    logger.warning(
                        "Empty Polynomial, introducing auxilliary variable to create polynomial."
                    )
                    func = self._function.as_poly(sympy.S("empty_generator"))
                else:
                    func = self._function.as_poly(*self._variables)
            return map(
                lambda term: (str(term[0]), self._monomial_to_state(term[1])),
                _term_generator(func))
        else:
            logger.debug("Multivariate Taylor expansion might take a while...")
            return map(
                lambda term: (str(term[0]), self._monomial_to_state(term[1])),
                self._mult_term_generator())

    def diff(self, variable, k):
        r"""
        Partial `k`-th derivative of the generating function with respect to variable `variable`.
        :param variable: The variable in which the generating function gets differentiated.
        :param k: The order of the partial derivative.
        :return: The `k`-th partial derivative of the generating function in `variable`

        .. math:: \fraction{\delta G^`k`}{\delta `var`^`k`}
        """
        logger.debug("diff Call")
        return GeneratingFunction(sympy.diff(self._function, sympy.S(variable),
                                             k),
                                  *self._variables,
                                  preciseness=self._preciseness)

    # FIXME: Its not nice to have different behaviour depending on the variable type of `threshold`.
    def approximate(
            self,
            threshold: Union[str,
                             int]) -> Generator[GeneratingFunction, None, None]:
        """
            Generate an approximation of a generating function, until `threshold` percent or terms of the probability
            mass is caputred.
        :param threshold: The probability percentage threshold of the probability mass of the distribution.
        :return: A Generating Function generator.
        """
        logger.debug("expand_until() call")
        approx = sympy.S("0")
        precision = sympy.S(0)

        if isinstance(threshold, int):
            assert threshold > 0, "Expanding to less than 0 terms is not valid."
            for n, prob, state in enumerate(self):
                if n >= threshold:
                    break
                s_prob = sympy.S(prob)
                approx += s_prob * sympy.S(state.to_monomial())
                precision += s_prob
                yield GeneratingFunction(approx,
                                         *self._variables,
                                         preciseness=precision,
                                         closed=False,
                                         finite=True)

        elif isinstance(threshold, str):
            s_threshold = sympy.S(threshold)
            assert s_threshold < self.coefficient_sum(), \
                f"Threshold cannot be larger than total coefficient sum! Threshold:" \
                f" {s_threshold}, CSum {self.coefficient_sum()}"
            for prob, state in self:
                if precision >= sympy.S(threshold):
                    break
                s_prob = sympy.S(prob)
                approx += s_prob * sympy.S(state.to_monomial())
                precision += s_prob
                yield GeneratingFunction(str(approx.expand()),
                                         *self._variables,
                                         preciseness=precision,
                                         closed=False,
                                         finite=True)
        else:
            raise TypeError(
                f"Parameter threshold can only be of type str or int,"
                f" not {type(threshold)}.")

    def coefficient_sum(self) -> sympy.Expr:
        logger.debug("coefficient_sum() call")
        coefficient_sum = self._function.simplify(
        ) if GeneratingFunction.use_simplification else self._function
        for var in self._variables:
            coefficient_sum = coefficient_sum.limit(
                var, 1, "-") if self._is_closed_form else coefficient_sum.subs(
                var, 1)
        return coefficient_sum

    def get_probability_mass(self):
        return str(self.coefficient_sum())

    def get_parameters(self) -> Set[str]:
        return set(map(str, self._parameters))

    def get_variables(self) -> Set[str]:
        return set(map(str, self._variables))

    def is_zero_dist(self) -> bool:
        return self._function == 0

    def get_probability_of(self, condition: Union[Expr, str]):
        return parse_expr(str(self.filter(condition).coefficient_sum()))

    def normalize(self) -> GeneratingFunction:
        logger.debug("normalized() call")
        mass = self.coefficient_sum()
        if mass == 0:
            raise ZeroDivisionError
        return GeneratingFunction(self._function / mass,
                                  *self._variables,
                                  preciseness=self._preciseness,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def is_finite(self):
        """
        Checks whether the generating function is finite.
        :return: True if the GF is a polynomial, False otherwise.
        """
        return self._is_finite

    @staticmethod
    def evaluate(expression: str, state: State) -> sympy.Expr:
        """ Evaluates the expression in a given state. """

        s_exp = sympy.S(expression)
        # Iterate over the variable, value pairs in the state
        for var, value in state.items():
            # Convert the variable into a sympy symbol and substitute
            s_var = sympy.S(var)
            s_exp = s_exp.subs(s_var, value)

        # If the state did not interpret all variables in the condition, interpret the remaining variables as 0
        for free_var in s_exp.free_symbols:
            s_exp = s_exp.subs(free_var, 0)
        return s_exp

    @staticmethod
    def evaluate_condition(condition: BinopExpr, state: State) -> bool:
        logger.debug("evaluate_condition() call")
        if not isinstance(condition, BinopExpr):
            raise AssertionError(
                f"Expression must be an (in-)equation, was {condition}")

        lhs = str(condition.lhs)
        rhs = str(condition.rhs)
        op = condition.operator

        if op == Binop.EQ:
            return GeneratingFunction.evaluate(
                lhs, state) == GeneratingFunction.evaluate(rhs, state)
        elif op == Binop.LEQ:
            return GeneratingFunction.evaluate(
                lhs, state) <= GeneratingFunction.evaluate(rhs, state)
        elif op == Binop.LT:
            return GeneratingFunction.evaluate(
                lhs, state) < GeneratingFunction.evaluate(rhs, state)
        elif op == Binop.GT:
            return GeneratingFunction.evaluate(
                lhs, state) > GeneratingFunction.evaluate(rhs, state)
        elif op == Binop.GEQ:
            return GeneratingFunction.evaluate(
                lhs, state) >= GeneratingFunction.evaluate(rhs, state)
        raise AssertionError(f"Unexpected condition type. {condition}")

    def marginal(
            self,
            *variables: Union[str, VarExpr],
            method: MarginalType = MarginalType.INCLUDE
    ) -> GeneratingFunction:
        """
        Computes the marginal distribution in the given variables.
        :param method: The method of marginalization.
        :param variables: a list of variables for which the marginal distribution should be computed
        :return: the marginal distribution.
        """
        logger.debug(
            "Creating marginal for variables %s and joint probability distribution %s",
            variables, self)
        marginal = self.copy()
        if method == MarginalType.INCLUDE:
            for s_var in marginal._variables.difference(
                    map(sympy.S, map(str, filter(lambda v: v != "", variables)))
            ):
                if marginal._is_closed_form:
                    marginal._function = marginal._function.limit(
                        s_var, 1, "-")
                else:
                    marginal._function = marginal._function.subs(s_var, 1)
            marginal._variables = set(map(sympy.S, filter(lambda v: v != "", map(str, variables))))
        else:
            for s_var in variables:
                if marginal._is_closed_form:
                    marginal._function = marginal._function.limit(
                        s_var, 1, "-")
                else:
                    marginal._function = marginal._function.subs(s_var, 1)
            marginal._variables = marginal._variables.difference(
                map(sympy.S, filter(lambda v: v != "", variables)))

        marginal._is_closed_form = not marginal._function.is_polynomial()
        marginal._is_finite = marginal._function.ratsimp().is_polynomial()

        return marginal

    def filter(self, condition: Expr) -> GeneratingFunction:
        """
        Filters out the terms of the generating function that satisfy the expression `expression`.
        :return: The filtered generating function.
        """
        logger.debug("filter(%s) call on %s", condition, self)

        # Boolean literals
        if isinstance(condition, BoolLitExpr):
            return self if condition.value else GeneratingFunction(
                "0", *self._variables)

        # Logical operators
        if condition.operator == Unop.NEG:
            result = self - self.filter(condition.expr)
            return result
        elif condition.operator == Binop.AND:
            result = self.filter(condition.lhs)
            return result.filter(condition.rhs)
        elif condition.operator == Binop.OR:
            filtered = self.filter(condition.lhs)
            return filtered + self.filter(condition.rhs) - filtered.filter(
                condition.rhs)

        # Modulo extractions
        elif check_is_modulus_condition(condition):
            return self._arithmetic_progression(
                str(condition.lhs.lhs),
                str(condition.lhs.rhs))[condition.rhs.value]

        # Constant expressions
        elif check_is_constant_constraint(condition, self):
            return self._filter_constant_condition(condition)

        # all other conditions given that the Generating Function is finite (exhaustive search)
        elif self._is_finite:
            result = sympy.S(0)
            for prob, state in self:
                if self.evaluate_condition(condition, state):
                    result += sympy.S(f"{prob} * {state.to_monomial()}")
            return GeneratingFunction(result,
                                      *self._variables,
                                      preciseness=self._preciseness,
                                      closed=False,
                                      finite=True)

        # Worst case: infinite Generating function and  non-standard condition.
        # Here we try marginalization and hope that the marginal is finite so we can do
        # exhaustive search again. If this is not possible, we raise an NotComputableException
        else:
            expression = self._explicit_state_unfolding(condition)
            return self.filter(expression)

    def linear_transformation(
            self, variable: str,
            expression: Union[Expr, str]) -> GeneratingFunction:
        """
        Computes a distribution where the `variable` is linearly transformed by `expression`.
        """

        logger.debug("linear_transformation() call")
        expr = expression
        if isinstance(expression, str):
            expr = parse_expr(expression)

        # Transform expression into sympy readable format
        rhs = sympy.S(str(expression))
        subst_var = sympy.S(variable)

        parameters: List[sympy.Symbol] = []
        for subexpr in walk_expr(Walk.DOWN, Mut.alloc(expr)):
            if isinstance(subexpr.val, VarExpr) and sympy.S(
                    subexpr.val.var) in self._parameters:
                parameters.append(sympy.S(subexpr.val.var))

        # Check whether the expression contains the substitution variable or not
        terms = rhs.as_coefficients_dict()
        result = self._function
        if subst_var not in terms.keys():
            if self._is_closed_form:
                print("\rComputing limit(linear transformation)...",
                      end='\r',
                      flush=True)
                result = result.limit(subst_var, 1, '-')
            else:
                result = result.subs(subst_var, 1)

        # Do the actual update stepwise
        const_correction_term = 1
        replacements = []
        for var in terms:
            # if there is a constant term, just do a multiplication
            if var == 1:
                const_correction_term = subst_var ** terms[1]
            elif var in parameters:
                const_correction_term = subst_var ** var
            # if the variable is the substitution variable, a different update is necessary
            elif var == subst_var:
                replacements.append((var, subst_var ** terms[var]))
            # otherwise we can collect the substitution in our replacement list
            else:
                replacements.append((var, var * subst_var ** terms[var]))
        res_gf = GeneratingFunction(result.subs(replacements) *
                                    const_correction_term,
                                    *self._variables,
                                    preciseness=self._preciseness,
                                    closed=self._is_closed_form,
                                    finite=self._is_finite)

        if not all(map(lambda x: terms[x] >= 0, terms)):
            summands = sympy.Add.make_args(res_gf._function)
            for summand in summands:
                subbed: sympy.Basic = summand.subs(subst_var, 0)
                for var in summand.free_symbols:
                    subbed = subbed.subs(var, 1)
                if subbed.equals(sympy.S("zoo")):
                    res_gf._function -= summand
                    res_gf._function += summand.subs(subst_var, 1)

        return res_gf


class SympyPGF(CommonDistributionsFactory):
    """Implements PGFs of standard distributions."""

    @staticmethod
    def geometric(var: Union[str, VarExpr],
                  p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 < sympy.S(p) < 1:
            raise ValueError(
                f"parameter of geom distr must be 0 < p <=1, was {p}")
        return GeneratingFunction(f"({p}) / (1 - (1-({p})) * {var})",
                                  var,
                                  closed=True,
                                  finite=False)

    @staticmethod
    def uniform(var: Union[str, VarExpr], lower: DistributionParam,
                upper: DistributionParam) -> Distribution:
        if isinstance(lower, str) and isinstance(
                upper, str) and not 0 <= sympy.S(lower) <= sympy.S(upper):
            raise ValueError(
                "Distribution parameters must satisfy 0 <= a < b < oo")
        return GeneratingFunction(
            f"1/(({upper}) - ({lower}) + 1) * {var}**({lower}) * ({var}**(({upper}) - ({lower}) + 1) - 1) / ({var} - 1)",
            var,
            closed=True,
            finite=True)

    @staticmethod
    def bernoulli(var: Union[str, VarExpr],
                  p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 <= sympy.S(p) <= 1:
            raise ValueError(
                f"Parameter of Bernoulli Distribution must be in [0,1], but was {p}"
            )
        return GeneratingFunction(f"1 - ({p}) + ({p}) * {var}",
                                  var,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def poisson(var: Union[str, VarExpr],
                lam: DistributionParam) -> Distribution:
        if isinstance(lam, str) and sympy.S(lam) < 0:
            raise ValueError(
                f"Parameter of Poisson Distribution must be in [0, oo), but was {lam}"
            )
        return GeneratingFunction(f"exp(({lam}) * ({var} - 1))",
                                  var,
                                  closed=True,
                                  finite=False)

    @staticmethod
    def log(var: Union[str, VarExpr], p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 <= sympy.S(p) <= 1:
            raise ValueError(
                f"Parameter of Logarithmic Distribution must be in [0,1], but was {p}"
            )
        return GeneratingFunction(f"log(1-({p})*{var})/log(1-({p}))",
                                  var,
                                  closed=True,
                                  finite=False)

    @staticmethod
    def binomial(var: Union[str, VarExpr], n: DistributionParam,
                 p: DistributionParam) -> Distribution:
        if isinstance(p, str) and not 0 <= sympy.S(p) <= 1:
            raise ValueError(
                f"Parameter of Binomial Distribution must be in [0,1], but was {p}"
            )
        if isinstance(n, str) and not 0 <= sympy.S(n):
            raise ValueError(
                f"Parameter of Binomial Distribution must be in [0,oo), but was {n}"
            )
        return GeneratingFunction(f"(1-({p})+({p})*{var})**({n})",
                                  var,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def zero(*variables: Union[str, sympy.Symbol]):
        if variables:
            return GeneratingFunction("0",
                                      *variables,
                                      preciseness=1,
                                      closed=True,
                                      finite=True)
        return GeneratingFunction("0", preciseness=1, closed=True, finite=True)

    @staticmethod
    def undefined(*variables: Union[str, VarExpr]) -> Distribution:
        """ A distribution where actually no information about the states is given."""
        raise NotImplementedError(
            "Currently this is unclear how to represent this.")

    @staticmethod
    def one(*variables: Union[str, VarExpr]) -> Distribution:
        return GeneratingFunction("1",
                                  *variables,
                                  preciseness=1,
                                  closed=True,
                                  finite=True)

    @staticmethod
    def from_expr(expression: Union[str, Expr], *variables,
                  **kwargs) -> Distribution:
        return GeneratingFunction(str(expression), *variables, **kwargs)
