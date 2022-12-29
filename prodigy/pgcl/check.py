"""
-------------
Type Checking
-------------
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, TypeVar, Union

import attr
# mypy complains here, but this isn't actually wrong
# the next version of frozendict will likely fix this issue
# https://github.com/Marco-Sulla/python-frozendict/issues/62
from frozendict import frozendict

from probably.pgcl.ast.declarations import Function, FunctionDecl, VarDecl
from probably.pgcl.ast.expressions import FunctionCallExpr
from probably.util.ref import Mut

from .ast import (AsgnInstr, Binop, BinopExpr, BoolLitExpr, BoolType,
                  CategoricalExpr, ChoiceInstr, Decl, ExpectationInstr, Expr,
                  IfInstr, Instr, LoopInstr, NatLitExpr, NatType, Node,
                  ObserveInstr, OptimizationQuery, PlotInstr, PrintInstr,
                  ProbabilityQueryInstr, Program, RealLitExpr, RealType,
                  SkipInstr, TickExpr, TickInstr, Type, Unop, UnopExpr, Var,
                  VarExpr, WhileInstr)
from .ast.walk import Walk, walk_expr

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


sample_predefined_functions: Dict[Var, List[VarDecl]] = frozendict({
    'unif_d': [VarDecl('start', NatType(None)),
               VarDecl('end', NatType(None))],
    'unif': [VarDecl('start', NatType(None)),
             VarDecl('end', NatType(None))],
    'geometric': [VarDecl('p', RealType())],
    'bernoulli': [VarDecl('p', RealType())],
    'poisson': [VarDecl('p', NatType(None))],
    'logdist': [VarDecl('p', RealType())],
    'binomial': [VarDecl('n', NatType(None)),
                 VarDecl('p', RealType())],
    'iid': [VarDecl('dist', NatType(None))]
})


def _check_function_call(
        program: Program, expr: FunctionCallExpr,
        predefined_functions: Dict[Var, List[VarDecl]]) -> Optional[CheckFail]:
    param_list: List[VarDecl]
    if expr.function in program.functions:
        param_list = program.functions[expr.function].declarations
    else:
        if expr.function not in predefined_functions:
            return CheckFail(expr, f"Unknown function: {expr.function}")
        param_list = predefined_functions[expr.function]

    if len(expr.params[0]) > len(param_list):
        return CheckFail(expr, "Too many parameters")

    covered_names: Set[Var] = set()
    for decl, param_expr in zip(param_list, expr.params[0]):
        typ = get_type(program, param_expr, True, predefined_functions)
        if isinstance(typ, CheckFail):
            return typ
        if not is_compatible(decl.typ, typ):
            return CheckFail.expected_type_got(param_expr, decl.typ, typ)
        covered_names.add(decl.var)

    param_dict = {decl.var: decl.typ for decl in param_list}
    for var, param_expr in expr.params[1].items():
        if var not in param_dict:
            return CheckFail(
                expr, f"Unknown variable in named function parameters: {var}")
        if var in covered_names:
            return CheckFail(
                expr, f"Variable {var} is covered by multiple parameters")
        typ = get_type(program, param_expr, True, predefined_functions)
        if isinstance(typ, CheckFail):
            return typ
        if not is_compatible(param_dict[var], typ):
            return CheckFail.expected_type_got(param_expr, param_dict[var],
                                               typ)
        covered_names.add(var)

    return None


# pylint: disable = too-many-statements
def get_type(
    program: Program,
    expr: Expr,
    check=True,
    predefined_functions: Dict[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Union[Type, CheckFail]:
    """
    Get the type of the expression or expectation.

    Set `check` to `True` to force checking correctness of all subexpressions, even if that is not necessary to determine the given expression's type.

    .. doctest::

        >>> from .ast import *
        >>> nat = NatLitExpr(10)
        >>> program = Program(list(), dict(), dict(), dict(), dict(), list())

        >>> get_type(program, BinopExpr(Binop.TIMES, nat, nat))
        NatType(bounds=None)

        >>> get_type(program, VarExpr('x'))
        CheckFail(location=VarExpr('x'), message='variable is not declared')

        >>> get_type(program, UnopExpr(Unop.IVERSON, BoolLitExpr(True)))
        RealType()
    """

    if isinstance(expr, VarExpr):
        # is it a variable?
        variable_type = program.variables.get(expr.var)
        if variable_type is not None:
            return variable_type

        # maybe a parameter?
        parameter_type = program.parameters.get(expr.var)
        if parameter_type is not None:
            return parameter_type

        # it must be a constant
        constant_type = program.constants.get(expr.var)
        if constant_type is None:
            return CheckFail(expr, "variable is not declared")
        return get_type(program,
                        constant_type,
                        predefined_functions=predefined_functions)

    if isinstance(expr, BoolLitExpr):
        return BoolType()

    if isinstance(expr, NatLitExpr):
        return NatType(bounds=None)

    if isinstance(expr, RealLitExpr):
        return RealType()

    if isinstance(expr, UnopExpr):
        if check:
            # Currently, all unary operators take boolean operands
            assert expr.operator in [Unop.NEG, Unop.IVERSON]

            typ = get_type(program, expr.expr, check, predefined_functions)
            if isinstance(typ, CheckFail):
                return typ

            if not is_compatible(BoolType(), typ):
                return CheckFail.expected_type_got(expr.expr, BoolType(), typ)

        if expr.operator == Unop.NEG:
            return BoolType()

        if expr.operator == Unop.IVERSON:
            return RealType()

    if isinstance(expr, BinopExpr):
        # binops that take boolean operands and return a boolean
        if expr.operator in [Binop.OR, Binop.AND]:
            if check:
                for operand in [expr.lhs, expr.rhs]:
                    typ = get_type(program, operand, check,
                                   predefined_functions)
                    if isinstance(typ, CheckFail):
                        return typ
                    if not is_compatible(BoolType(), typ):
                        return CheckFail.expected_type_got(
                            operand, BoolType(), typ)
            return BoolType()

        # the rest of binops take numeric operands, so check those
        lhs_typ = get_type(program, expr.lhs, check, predefined_functions)
        if isinstance(lhs_typ, CheckFail):
            return lhs_typ
        if not isinstance(lhs_typ, (NatType, RealType)):
            return CheckFail.expected_numeric_got(expr.lhs, lhs_typ)

        if check and expr.operator in [
                Binop.LEQ, Binop.LT, Binop.GEQ, Binop.GT, Binop.EQ, Binop.PLUS,
                Binop.MINUS, Binop.TIMES, Binop.MODULO, Binop.POWER,
                Binop.DIVIDE
        ]:
            rhs_typ = get_type(program, expr.rhs, check, predefined_functions)
            if isinstance(rhs_typ, CheckFail):
                return rhs_typ
            if not isinstance(rhs_typ, (NatType, RealType)):
                return CheckFail.expected_numeric_got(expr.rhs, rhs_typ)

            if not is_compatible(lhs_typ, rhs_typ):
                return CheckFail.expected_type_got(expr, lhs_typ, rhs_typ)

        # binops that take numeric operands and return a boolean
        if expr.operator in [
                Binop.LEQ, Binop.LT, Binop.EQ, Binop.GEQ, Binop.GT
        ]:
            return BoolType()

        # binops that take numeric operands and return a numeric value
        if expr.operator in [
                Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.MODULO,
                Binop.POWER, Binop.DIVIDE
        ]:
            # intentionally lose the bounds on NatType (see NatType documentation)
            if isinstance(lhs_typ, NatType) and lhs_typ.bounds is not None:
                return NatType(bounds=None)
            return lhs_typ

    if isinstance(expr, CategoricalExpr):
        first_expr = expr.exprs[0][0]
        typ = get_type(program, first_expr, check, predefined_functions)
        if isinstance(typ, CheckFail):
            return typ
        if check:
            for other_expr, _other_prob in expr.exprs[1:]:
                other_typ = get_type(program, other_expr, check,
                                     predefined_functions)
                if isinstance(other_typ, CheckFail):
                    return other_typ
                if not is_compatible(typ, other_typ):
                    return CheckFail.expected_type_got(expr, typ, other_typ)
        return typ

    if isinstance(expr, TickExpr):
        typ = get_type(program, expr.expr, check, predefined_functions)
        if isinstance(typ, CheckFail):
            return typ
        if check and not is_compatible(NatType(bounds=None), typ):
            return CheckFail.expected_type_got(expr.expr, NatType(None), typ)
        return typ

    if isinstance(expr, FunctionCallExpr):
        if check:
            valid = _check_function_call(program, expr, predefined_functions)
            if isinstance(valid, CheckFail):
                return valid
        return NatType(bounds=None)

    raise Exception("unreachable")


# pylint: enable = too-many-statements


def is_compatible(lhs: Type, rhs: Type) -> bool:
    """
    Check if the right-hand side type is compatible with the left-hand side type.

    This function essentially defines the implicit type coercions of the pGCL language.
    Compatibility is not only important for which values can be used as the right-hand side of an assignment, but compatibility
    rules are also checked when operators are used.

    The only rules for compatibility currently implemented are:

    * Natural numbers are compatible with variables of every other type of number (bounded or unbounded).
    * Otherwise the types must be exactly equal for values to be compatible.

    .. doctest::

        >>> from .ast import Bounds

        >>> is_compatible(BoolType(), BoolType())
        True
        >>> is_compatible(NatType(bounds=None), NatType(bounds=None))
        True
        >>> is_compatible(NatType(bounds=Bounds(NatLitExpr(1), NatLitExpr(5))), NatType(bounds=Bounds(NatLitExpr(1), NatLitExpr(5))))
        True
        >>> is_compatible(NatType(bounds=Bounds(NatLitExpr(1), NatLitExpr(5))), NatType(bounds=None))
        True

        >>> is_compatible(BoolType(), NatType(bounds=None))
        False
    """
    if isinstance(lhs, NatType) and isinstance(rhs, NatType):
        return True
    return lhs == rhs


def _check_constant_declarations(program: Program) -> Optional[CheckFail]:
    """
    Check that constants are defined before they are used.
    Also check that constants only contain other constants.

    .. doctest::

        >>> from .parser import parse_pgcl
        >>> program = parse_pgcl("const x := y; const y := x")
        >>> _check_constant_declarations(program)
        CheckFail(location=VarExpr('y'), message='Constant is not yet defined')

        >>> program = parse_pgcl("nat x; const y := x")
        >>> _check_constant_declarations(program)
        CheckFail(location=VarExpr('x'), message='Variable used in constant definition is not a constant')
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


def _check_declaration_list(
        program: Program,
        predefined_functions: Dict[Var, List[VarDecl]]) -> Optional[CheckFail]:
    """
    Check that all variables/constants are defined at most once.

    .. doctest::

        >>> from .parser import parse_pgcl
        >>> program = parse_pgcl("const x := 1; const x := 1")
        >>> _check_declaration_list(program, {})
        CheckFail(location=..., message='Already declared variable/constant before.')
    """
    declared: Dict[Var, Decl] = {}
    for decl in program.declarations:
        if declared.get(decl.var) is not None:
            return CheckFail(decl,
                             "Already declared variable/constant before.")
        declared[decl.var] = decl
        if isinstance(decl, FunctionDecl):
            res = check_function(decl.body, program, predefined_functions)
            if isinstance(res, CheckFail):
                return res
    return None


def _check_instrs(
        program: Program, *instrs: List[Instr],
        predefined_functions: Dict[Var, List[VarDecl]]) -> Optional[CheckFail]:
    for instrs_list in instrs:
        for instr in instrs_list:
            res = check_instr(program,
                              instr,
                              predefined_functions=predefined_functions)
            if isinstance(res, CheckFail):
                return res
    return None


# pylint: disable = too-many-statements
def check_instr(
    program: Program,
    instr: Instr,
    predefined_functions: Dict[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """
    Check a single instruction for type-safety.

    .. doctest::

        >>> from .parser import parse_pgcl

        >>> program = parse_pgcl("bool x; x := true")
        >>> check_instr(program, program.instructions[0])

        >>> program = parse_pgcl("nat x; x := true")
        >>> check_instr(program, program.instructions[0])
        CheckFail(location=..., message='Expected value of type NatType(bounds=None), got BoolType().')

        >>> program = parse_pgcl("const x := true; x := true")
        >>> check_instr(program, program.instructions[0])
        CheckFail(location=..., message='x is not a variable.')

        >>> program = parse_pgcl("nat x [1,5]; x := 3")
        >>> check_instr(program, program.instructions[0])

        >>> program = parse_pgcl("nat x [1,5]; nat y; x := x + 3")
        >>> check_instr(program, program.instructions[0])
    """
    if isinstance(instr, SkipInstr):
        return None

    if isinstance(instr, WhileInstr):
        cond_type = get_type(program,
                             instr.cond,
                             predefined_functions=predefined_functions)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not is_compatible(BoolType(), cond_type):
            return CheckFail.expected_type_got(instr.cond, BoolType(),
                                               cond_type)
        return _check_instrs(program,
                             instr.body,
                             predefined_functions=predefined_functions)

    if isinstance(instr, IfInstr):
        cond_type = get_type(program,
                             instr.cond,
                             predefined_functions=predefined_functions)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not is_compatible(BoolType(), cond_type):
            return CheckFail.expected_type_got(instr.cond, BoolType(),
                                               cond_type)
        return _check_instrs(program,
                             instr.true,
                             instr.false,
                             predefined_functions=predefined_functions)

    if isinstance(instr, AsgnInstr):
        lhs_type = program.variables.get(instr.lhs)
        if lhs_type is None:
            return CheckFail(instr, f'{instr.lhs} is not a variable.')
        rhs_type = get_type(program,
                            instr.rhs,
                            check=True,
                            predefined_functions=predefined_functions)
        if isinstance(rhs_type, CheckFail):
            return rhs_type
        if not is_compatible(lhs_type, rhs_type):
            return CheckFail.expected_type_got(instr, lhs_type, rhs_type)
        return None

    if isinstance(instr, ChoiceInstr):
        prob_type = get_type(program,
                             instr.prob,
                             check=True,
                             predefined_functions=predefined_functions)
        if isinstance(prob_type, CheckFail):
            return prob_type
        if not is_compatible(RealType(), prob_type):
            return CheckFail.expected_type_got(instr.prob, RealType(),
                                               prob_type)
        return _check_instrs(program,
                             instr.lhs,
                             instr.rhs,
                             predefined_functions=predefined_functions)

    if isinstance(instr, ObserveInstr):
        cond_type = get_type(program,
                             instr.cond,
                             predefined_functions=predefined_functions)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not is_compatible(BoolType(), cond_type):
            return CheckFail.expected_type_got(instr.cond, BoolType(),
                                               cond_type)
        return None

    if isinstance(instr, ProbabilityQueryInstr):
        cond_type = get_type(program,
                             instr.expr,
                             predefined_functions=predefined_functions)
        if isinstance(cond_type, CheckFail):
            return cond_type
        if not (is_compatible(BoolType(), cond_type)
                or is_compatible(NatType(None), cond_type)):
            return CheckFail.expected_type_got(instr.expr, BoolType(),
                                               cond_type)
        return None

    if isinstance(instr, ExpectationInstr):
        expr_type = get_type(program,
                             instr.expr,
                             check=True,
                             predefined_functions=predefined_functions)
        if isinstance(expr_type, CheckFail):
            return expr_type
        return None

    if isinstance(instr, PrintInstr):
        return None

    if isinstance(instr, OptimizationQuery):
        expr_type = get_type(program,
                             instr.expr,
                             check=True,
                             predefined_functions=predefined_functions)
        if isinstance(expr_type, CheckFail):
            return expr_type
        if instr.parameter not in program.parameters:
            return CheckFail(
                instr,
                f"In optimization queries, the variable can only be a program parameter, was {instr.parameter}."
            )
        return None

    if isinstance(instr, PlotInstr):
        var_1_type = get_type(program,
                              instr.var_1,
                              check=True,
                              predefined_functions=predefined_functions)
        if isinstance(var_1_type, CheckFail):
            return var_1_type
        if instr.var_2:
            var_2_type = get_type(program,
                                  instr.var_2,
                                  check=True,
                                  predefined_functions=predefined_functions)
            if isinstance(var_2_type, CheckFail):
                return var_2_type
        if instr.prob:
            prob_type = get_type(program,
                                 instr.prob,
                                 check=True,
                                 predefined_functions=predefined_functions)
            if isinstance(prob_type, CheckFail):
                return prob_type
            if not is_compatible(RealType(), prob_type):
                return CheckFail.expected_type_got(instr.prob, RealType(),
                                                   prob_type)
        if instr.term_count:
            int_type = get_type(program,
                                instr.term_count,
                                check=True,
                                predefined_functions=predefined_functions)
            if isinstance(int_type, CheckFail):
                return int_type
            if not is_compatible(NatType(bounds=None), int_type):
                return CheckFail.expected_type_got(instr.term_count,
                                                   NatType(None), int_type)
        return None

    if isinstance(instr, LoopInstr):
        int_type = get_type(program,
                            instr.iterations,
                            check=True,
                            predefined_functions=predefined_functions)
        if isinstance(int_type, CheckFail):
            return int_type
        if not is_compatible(NatType(bounds=None), int_type):
            return CheckFail.expected_type_got(instr.iterations, NatType(None),
                                               int_type)
        return _check_instrs(program,
                             instr.body,
                             predefined_functions=predefined_functions)

    if isinstance(instr, TickInstr):
        typ = get_type(program,
                       instr.expr,
                       check=True,
                       predefined_functions=predefined_functions)
        if isinstance(typ, CheckFail):
            return typ
        if not is_compatible(NatType(bounds=None), typ):
            return CheckFail.expected_type_got(instr.expr, NatType(None), typ)
        return None

    raise Exception("unreachable")


# pylint: enable = too-many-statements


def check_program(
    program: Program,
    predefined_functions: Dict[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """Check a program for type-safety."""
    check_result = _check_constant_declarations(program)
    if check_result is not None:
        return check_result
    check_result = _check_declaration_list(program, predefined_functions)
    if check_result is not None:
        return check_result
    return _check_instrs(program,
                         program.instructions,
                         predefined_functions=predefined_functions)


def check_expression(
    program,
    expr: Expr,
    predefined_functions: Dict[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """
    Check an expression for type-safety.

    .. doctest::

        >>> program = Program(list(), dict(), dict(), dict(), dict(), list())
        >>> check_expression(program, RealLitExpr("1.0"))
        CheckFail(location=..., message='A program expression may not return a probability.')
    """
    expr_type = get_type(program,
                         expr,
                         check=True,
                         predefined_functions=predefined_functions)
    if isinstance(expr_type, CheckFail):
        return expr_type
    if isinstance(expr_type, RealType):
        return CheckFail(expr,
                         "A program expression may not return a probability.")
    return None


def check_expectation(
    program,
    expr: Expr,
    predefined_functions: Dict[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """Check an expectation for type-safety."""
    expr_type = get_type(program,
                         expr,
                         check=True,
                         predefined_functions=predefined_functions)
    if isinstance(expr_type, CheckFail):
        return expr_type
    return None


def check_function(
    function: Function,
    context: Program,
    predefined_functions: Dict[Var,
                               List[VarDecl]] = sample_predefined_functions
) -> Optional[CheckFail]:
    """
    Check a function for type-safety. Functions currently may only contain and
    return integers.
    """

    for decl in function.declarations:
        if decl.typ != NatType(bounds=None):
            return CheckFail(
                location=decl,
                message=
                "Only variables of type NatType are allowed in functions")

    as_prog = Program.from_function(function, context)
    res = check_program(as_prog, predefined_functions)
    if isinstance(res, CheckFail):
        return res

    typ = get_type(as_prog,
                   function.returns,
                   check=True,
                   predefined_functions=predefined_functions)
    if isinstance(typ, CheckFail):
        return typ
    if not isinstance(typ, NatType):
        return CheckFail(location=function.returns,
                         message="Functions may only return integers")

    return None
