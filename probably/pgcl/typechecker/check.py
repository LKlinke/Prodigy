"""
-------------
Type Checking
-------------
"""

from typing import Dict, Optional, Union, get_args, Tuple
import attr
from probably.pgcl.ast import *
from probably.pgcl.ast.expressions import IidSampleExpr
from probably.pgcl.ast.instructions import OptimizationQuery
from probably.util.color import Style

_T = TypeVar('_T')


@attr.s
class CheckFail:
    """
    A check failed with an error message and an optional location.
    """

    location: Optional[Node] = attr.ib()
    message: str = attr.ib()

    @staticmethod
    def expected_type_got(node: Node, expected: Type,
                          got: Type) -> 'CheckFail':
        return CheckFail(node,
                         f'Expected value of type {expected}, got {got}.')

    @staticmethod
    def expected_numeric_got(node: Node, got: Type) -> 'CheckFail':
        return CheckFail(node, f'Expected numeric value, got {got}.')

    @staticmethod
    def fmap(
            source: Iterable[Union[_T,
                                   'CheckFail']]) -> Union['CheckFail', List[_T]]:
        """
        Extract an error from an iterable.

        .. doctest::

            >>> CheckFail.fmap(['a'])
            ['a']
            >>> CheckFail.fmap([CheckFail(None, 'whoops!')])
            CheckFail(location=None, message='whoops!')
        """
        output = []
        for step in source:
            if isinstance(step, CheckFail):
                return step
            output.append(step)
        return output


def _get_variable_type(prog: Program, expr: Expr, check: bool = True) -> Union[Type, CheckFail]:
    assert isinstance(expr, VarExpr), f"Cannot type check variable for expression {expr}."

    # is it a variable?
    variable_type = prog.variables.get(expr.var)
    if variable_type is not None:
        return variable_type

    # maybe a parameter?
    parameter_type = prog.parameters.get(expr.var)
    if parameter_type is not None:
        return parameter_type

    # it must be a constant
    constant_type = prog.constants.get(expr.var)
    if constant_type is None:
        return CheckFail(expr, "variable is not declared")
    return get_type(prog, constant_type)


def _get_unary_operation_type(prog: Program, expr: Expr, check: bool = True) -> Union[Type, CheckFail]:
    assert isinstance(expr, UnopExpr), f"Expression must be of type UnaryExpr, was {type(expr)}."
    if check:
        # Currently, all unary operators take boolean operands
        assert expr.operator in [Unop.NEG, Unop.IVERSON]

        operand_typ = get_type(prog, expr.expr, check=check)
        if isinstance(operand_typ, CheckFail):
            return operand_typ

        if not is_compatible(BoolType(), operand_typ):
            return CheckFail.expected_type_got(expr.expr, BoolType(),
                                               operand_typ)

    if expr.operator == Unop.NEG:
        return BoolType()

    if expr.operator == Unop.IVERSON:
        return RealType()


def _get_binary_operation_type(prog: Program, expr: Expr, check: bool = True) -> Union[Type, CheckFail]:
    assert isinstance(expr, BinopExpr), f"Expression must be of type BinopExpr, was {type(expr)}."

    # binops that take boolean operands and return a boolean
    if expr.operator in [Binop.OR, Binop.AND]:
        if check:
            for operand in [expr.lhs, expr.rhs]:
                operand_typ = get_type(prog, operand, check=check)
                if isinstance(operand_typ, CheckFail):
                    return operand_typ
                if not is_compatible(BoolType(), operand_typ):
                    return CheckFail.expected_type_got(
                        operand, BoolType(), operand_typ)
        return BoolType()

    # the rest of binops take numeric operands, so check those
    lhs_typ = get_type(prog, expr.lhs, check=check)
    if isinstance(lhs_typ, CheckFail):
        return lhs_typ
    if not isinstance(lhs_typ, (NatType, RealType)):
        return CheckFail.expected_numeric_got(expr.lhs, lhs_typ)

    if check and expr.operator in [
        Binop.LEQ, Binop.LE, Binop.GEQ, Binop.GE, Binop.EQ, Binop.PLUS, Binop.MINUS,
        Binop.TIMES, Binop.MODULO, Binop.POWER
    ]:
        rhs_typ = get_type(prog, expr.rhs, check=check)
        if isinstance(rhs_typ, CheckFail):
            return rhs_typ
        if not isinstance(rhs_typ, (NatType, RealType)):
            return CheckFail.expected_numeric_got(expr.rhs, rhs_typ)

        if not is_compatible(lhs_typ, rhs_typ):
            return CheckFail.expected_type_got(expr, lhs_typ, rhs_typ)

    # binops that take numeric operands and return a boolean
    if expr.operator in [Binop.LEQ, Binop.LE, Binop.EQ, Binop.GEQ, Binop.GE]:
        return BoolType()

    # binops that take numeric operands and return a numeric value
    if expr.operator in [Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.MODULO, Binop.POWER]:
        # intentionally lose the bounds on NatType (see NatType documentation)
        if isinstance(lhs_typ, NatType) and lhs_typ.bounds is not None:
            return NatType(bounds=None)
        return lhs_typ


def _check_distribution_arguments(prog: Program, expected_type: Type, *parameters: Tuple[Type, Expr]):
    for expected_param_type, param_expr in parameters:
        param_type = get_type(prog, param_expr)
        if isinstance(param_type, CheckFail):
            return CheckFail
        elif not is_compatible(expected_param_type, param_type):
            return CheckFail.expected_type_got(param_expr, expected_param_type, param_type)
    return expected_type


def get_distribution_type(prog: Program, expr: Expr, check: bool = True) -> Union[Type, CheckFail]:
    assert isinstance(expr, get_args(DistrExpr)), f"Expression must be a distribution expression, was {type(expr)}."

    if isinstance(expr, DUniformExpr):
        return _check_distribution_arguments(prog, NatType(bounds=None), (NatType(bounds=None), expr.start),
                                             (NatType(bounds=None), expr.end)
                                             )
    elif isinstance(expr, CUniformExpr):
        raise NotImplementedError("Currently continuous distributions are not supported.")

    elif isinstance(expr, GeometricExpr):
        return _check_distribution_arguments(prog, NatType(bounds=None), (RealType(), expr.param))

    elif isinstance(expr, BernoulliExpr):
        return _check_distribution_arguments(prog, NatType(bounds=None), (RealType(), expr.param))

    elif isinstance(expr, PoissonExpr):
        return _check_distribution_arguments(prog, NatType(bounds=None), (NatType(bounds=None), expr.param))

    elif isinstance(expr, LogDistExpr):
        return _check_distribution_arguments(prog, NatType(bounds=None), (RealType(), expr.param))

    elif isinstance(expr, BinomialExpr):
        return _check_distribution_arguments(prog, NatType(bounds=None), (NatType(bounds=None), expr.n),
                                             (RealType(), expr.p)
                                             )
    if isinstance(expr, IidSampleExpr):
        if isinstance(expr.sampling_dist, get_args(DistrExpr)):
            return get_distribution_type(prog, expr.sampling_dist, check)
        else:
            print(Style.OKRED + "Type Checking is not accurate anymore." + Style.RESET)
            return check_expression(prog, expr.sampling_dist)


def get_type(program: Program,
             expr: Expr,
             check=True) -> Union[Type, CheckFail]:
    """
    Get the type of the expression or expectation.

    Set `check` to `True` to force checking correctness of all subexpressions, even if that is not necessary
    to determine the given expression's type.
    """

    # Atomic Expressions
    if isinstance(expr, VarExpr):
        return _get_variable_type(program, expr, check)
    elif isinstance(expr, BoolLitExpr):
        return BoolType()
    elif isinstance(expr, NatLitExpr):
        return NatType(bounds=None)
    elif isinstance(expr, RealLitExpr):
        return RealType()

    # Composite Expressions.
    elif isinstance(expr, get_args(DistrExpr)):
        return get_distribution_type(program, expr, check)
    elif isinstance(expr, UnopExpr):
        return _get_unary_operation_type(program, expr, check)
    elif isinstance(expr, BinopExpr):
        return _get_binary_operation_type(program, expr, check)

    # Categorical Expressions are handled differently (mainly important for wp-analysis).
    elif isinstance(expr, CategoricalExpr):
        first_expr = expr.exprs[0][0]
        typ = get_type(program, first_expr, check=check)
        if isinstance(typ, CheckFail):
            return typ
        if check:
            for other_expr, _other_prob in expr.exprs[1:]:
                other_typ = get_type(program, other_expr, check=check)
                if isinstance(other_typ, CheckFail):
                    return other_typ
                if not is_compatible(typ, other_typ):
                    return CheckFail.expected_type_got(expr, typ, other_typ)
        return typ

    # Unknown Expression type.
    raise SyntaxError("Unknown type of expression.")


def is_compatible(lhs: Type, rhs: Type) -> bool:
    """
    Check if the right-hand side type is compatible with the left-hand side type.

    This function essentially defines the implicit type coercions of the pGCL language.
    Compatibility is not only important for which values can be used as the right-hand side of an assignment, but compatibility
    rules are also checked when operators are used.

    The only rules for compatibility currently implemented are:

    * Natural numbers are compatible with variables of every other type of number (bounded or unbounded).
    * Otherwise the types must be exactly equal for values to be compatible.
    """
    if isinstance(lhs, NatType) and isinstance(rhs, NatType):
        return True
    return lhs == rhs


def check_constant_declarations(program: Program) -> Optional[CheckFail]:
    """
    Check that constants are defined before they are used.
    Also check that constants only contain other constants.
    """
    declared = set()
    for name, value in program.constants.items():
        for subexpr_ref in walk_expr(Walk.DOWN, Mut.alloc(value)):
            if isinstance(subexpr_ref.val, VarExpr):
                subexpr_var = subexpr_ref.val.var
                subexpr_def = program.constants.get(subexpr_var)
                if subexpr_def is None:
                    return CheckFail(
                        subexpr_ref.val,
                        "Variable used in constant definition is not a constant"
                    )
                if subexpr_var not in declared:
                    return CheckFail(subexpr_ref.val,
                                     "Constant is not yet defined")
        declared.add(name)

    return None


def check_declaration_list(program: Program) -> Optional[CheckFail]:
    """
    Check that all variables/constants are defined at most once and that real
    variables are only declared if they are allowed in the config.
    """
    declared: Dict[Var, Decl] = dict()
    for decl in program.declarations:
        if declared.get(decl.var) is not None:
            return CheckFail(decl,
                             "Already declared variable/constant before.")
        if not program.config.allow_real_vars:
            if isinstance(decl, VarDecl) and isinstance(decl.typ, RealType):
                return CheckFail(
                    decl,
                    "Real number variables are not allowed by the program config."
                )
        declared[decl.var] = decl
    return None


def check_instrs(program: Program,
                 *instrs: List[Instr]) -> Optional[CheckFail]:
    for instrs_list in instrs:
        for instr in instrs_list:
            res = check_instr(program, instr)
            if isinstance(res, CheckFail):
                return res
    return None


def check_instr(program: Program, instr: Instr) -> Optional[CheckFail]:
    """
    Check a single instruction for type-safety.
    """
    if isinstance(instr, SkipInstr):
        return None

    if isinstance(instr, WhileInstr):
        cond_type = get_type(program, instr.cond)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not is_compatible(BoolType(), cond_type):
            return CheckFail.expected_type_got(instr.cond, BoolType(),
                                               cond_type)
        return check_instrs(program, instr.body)

    if isinstance(instr, IfInstr):
        cond_type = get_type(program, instr.cond)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not is_compatible(BoolType(), cond_type):
            return CheckFail.expected_type_got(instr.cond, BoolType(),
                                               cond_type)
        return check_instrs(program, instr.true, instr.false)

    if isinstance(instr, AsgnInstr):
        lhs_type = program.variables.get(instr.lhs)
        if lhs_type is None:
            return CheckFail(instr, f'{instr.lhs} is not a variable.')
        rhs_type = get_type(program, instr.rhs, check=True)
        if isinstance(rhs_type, CheckFail):
            return rhs_type
        if not is_compatible(lhs_type, rhs_type):
            return CheckFail.expected_type_got(instr, lhs_type, rhs_type)
        return None

    if isinstance(instr, ChoiceInstr):
        prob_type = get_type(program, instr.prob, check=True)
        if isinstance(prob_type, CheckFail):
            return prob_type
        if not is_compatible(RealType(), prob_type):
            return CheckFail.expected_type_got(instr.prob, RealType(),
                                               prob_type)
        return check_instrs(program, instr.lhs, instr.rhs)

    if isinstance(instr, ObserveInstr):
        cond_type = get_type(program, instr.cond)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not is_compatible(BoolType(), cond_type):
            return CheckFail.expected_type_got(instr.cond, BoolType(),
                                               cond_type)
        return None

    if isinstance(instr, ProbabilityQueryInstr):
        cond_type = get_type(program, instr.expr)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not (is_compatible(BoolType(), cond_type) or is_compatible(NatType(None), cond_type)):
            return CheckFail.expected_type_got(instr.expr, BoolType(),
                                               cond_type)
        return None

    if isinstance(instr, ExpectationInstr):
        expr_type = get_type(program, instr.expr, check=True)
        if isinstance(expr_type, CheckFail):
            return expr_type
        return None

    if isinstance(instr, PrintInstr):
        return None

    if isinstance(instr, OptimizationQuery):
        expr_type = get_type(program, instr.expr, check=True)
        if isinstance(expr_type, CheckFail):
            return expr_type
        return None

    if isinstance(instr, PlotInstr):
        var_1_type = get_type(program, instr.var_1, check=True)
        if isinstance(var_1_type, CheckFail):
            return var_1_type
        if instr.var_2:
            var_2_type = get_type(program, instr.var_2, check=True)
            if isinstance(var_2_type, CheckFail):
                return var_2_type
        if instr.prob:
            prob_type = get_type(program, instr.prob, check=True)
            if isinstance(prob_type, CheckFail):
                return prob_type
            if not is_compatible(RealType(), prob_type):
                return CheckFail.expected_type_got(instr.prob, RealType(),
                                                   prob_type)
        if instr.term_count:
            int_type = get_type(program, instr.term_count, check=True)
            if isinstance(int_type, CheckFail):
                return int_type
            if not is_compatible(NatType(bounds=None), int_type):
                return CheckFail.expected_type_got(instr.term_count, NatType(None),
                                                   int_type)
        return None
    if isinstance(instr, LoopInstr):
        int_type = get_type(program, instr.iterations, check=True)
        if isinstance(int_type, CheckFail):
            return int_type
        if not is_compatible(NatType(bounds=None), int_type):
            return CheckFail.expected_type_got(instr.iterations, NatType(None),
                                               int_type)
        return check_instrs(program, instr.body)

    raise Exception("unreachable")


def check_program(program: Program) -> Optional[CheckFail]:
    """Check a program for type-safety."""
    check_result = check_constant_declarations(program)
    if check_result is not None:
        return check_result
    check_result = check_declaration_list(program)
    if check_result is not None:
        return check_result
    return check_instrs(program, program.instructions)


def check_expression(program, expr: Expr) -> Optional[CheckFail]:
    """
    Check an expression for type-safety.
    """
    expr_type = get_type(program, expr, check=True)
    if isinstance(expr_type, CheckFail):
        return expr_type
    if isinstance(expr_type, RealType):
        return CheckFail(expr,
                         "A program expression may not return a probability.")
    return None


def check_expectation(program, expr: Expr) -> Optional[CheckFail]:
    """Check an expectation for type-safety."""
    expr_type = get_type(program, expr, check=True)
    if isinstance(expr_type, CheckFail):
        return expr_type
    return None
