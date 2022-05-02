"""

A Lark-based parser for pGCL.

For more details on what syntax is accepted for pGCL programs, you can view the :ref:`formal grammar used for the parser <pgcl_grammar>`.

.. rubric:: Notes on the parsing algorithm

For most of the grammar, a simple run of the Lark parser suffices. The only
slightly tricky part is parsing of categorical choice expressions. To avoid
ambiguous grammar, we parse those as normal expressions of `PLUS` operators.
Individual weights attached to probabilities are saved in a temporary
`LikelyExpr` type which associates a single expression with a probability. Later
we collect all `LikelyExpr` and flatten them into a single
:class:`CategoricalExpr`. `LikelyExpr` never occur outside of this parser.
"""
import textwrap
from fractions import Fraction

import attr
import os
from decimal import Decimal
from typing import Optional, Union, Tuple, Dict

from lark import Lark, Tree

from probably.pgcl.analyzer.syntax import has_variable
from probably.pgcl.ast import *
from probably.pgcl.ast.expressions import expr_str_parens, IidSampleExpr
from probably.pgcl.ast.instructions import OptimizationQuery
from probably.util.lark_expr_parser import (atom, build_expr_parser, infixl,
                                            prefix, infixr)
from probably.util.ref import Mut

_PGCL_GRAMMAR = ""


def setup(filename: str) -> Lark:
    # Read the specification grammar
    global _PGCL_GRAMMAR
    with open(filename, 'r') as file:
        _PGCL_GRAMMAR = file.read()

    _OPERATOR_TABLE = [[infixl("or", "||")], [infixl("and", "&")],
                       [infixl("leq", "<="),
                        infixl("le", "<"),
                        infixl("ge", ">"),
                        infixl("geq", ">="),
                        infixl("eq", "=")],
                       [infixl("plus", "+"),
                        infixl("minus", "-")], [infixl("times", "*"), infixl("divide", "/")], [infixr("power", "^")],
                       [infixl("likely", ":")], [infixl("mod", "%")],
                       [
                           prefix("neg", "not "),
                           atom("parens", '"(" expression ")"'),
                           atom("iverson", '"[" expression "]"'),
                           atom("literal", "literal"),
                           atom("var", "var")
                       ]]
    _PGCL_GRAMMAR += "\n" + textwrap.indent(build_expr_parser(_OPERATOR_TABLE, "expression"), '    ')

    return Lark(_PGCL_GRAMMAR)


def _doc_parser_grammar():
    raise Exception(
        "This function only exists for documentation purposes and should never be called"
    )


_doc_parser_grammar.__doc__ = "The Lark grammar for pGCL::\n" + _PGCL_GRAMMAR + "\n\nThis function only exists for " \
                                                                                "documentation purposes and should " \
                                                                                "never be called in code. "

_PARSER = setup(os.path.dirname(os.path.abspath(__file__)) + "/pgcl_grammar.txt")

# Collect parameter information here.
parameters: Dict[Var, Type] = dict()

# All known distribution types. Dictionary entry contains the token name as key.
# Also the value of a given gey is a tuple consisting of the number of parameters and the Class name (constructor call)
distributions: Dict[str, Tuple[int, Callable]] = {
    "duniform": (2, DUniformExpr),
    "cuniform": (2, CUniformExpr),
    "geometric": (1, GeometricExpr),
    "poisson": (1, PoissonExpr),
    "logdist": (1, LogDistExpr),
    "bernoulli": (1, BernoulliExpr),
    "binomial": (2, BinomialExpr)
}


@attr.s
class _LikelyExpr(ExprClass):
    """
    A temporary type of expressions, used to build up a
    :class:`CategoricalExpr`. A single _LikelyExpr is an expression value and a
    probability as a constant. _LikelyExprs are transformed into
    CategoricalExprs later during parsing.

    They do not occur in the public API of probably, but they may occur as part
    of errors emitted by the parser before translation to CategoricalExprs.
    """
    value: Expr = attr.ib()
    prob: RealLitExpr = attr.ib()

    def __str__(self) -> str:
        return f'{expr_str_parens(self.value)} : {expr_str_parens(self.prob)}'


def _as_tree(t: Union[str, Tree]) -> Tree:
    assert isinstance(t, Tree)
    return t


def _child_tree(t: Tree, index: int) -> Tree:
    return _as_tree(t.children[index])


def _child_str(t: Tree, index: int) -> str:
    res = t.children[index]
    assert isinstance(res, str)
    return res


def _parse_var(t: Tree) -> Var:
    assert t.data == 'var'
    assert t.children[0].type == 'CNAME'  # type: ignore
    return str(_child_str(t, 0))


def _parse_bounds(t: Optional[Tree]) -> Optional[Bounds]:
    if t is None:
        return None
    assert isinstance(t, Tree) and t.data == "bounds"
    return Bounds(_parse_expr(_child_tree(t, 0)),
                  _parse_expr(_child_tree(t, 1)))


def _parse_declaration(t: Tree) -> Decl:
    def var0():
        return _parse_var(_child_tree(t, 0))

    def opt_child1():
        if len(t.children) <= 1:
            return None
        return _child_tree(t, 1)

    if t.data == "bool":
        return VarDecl(var0(), BoolType())
    elif t.data == "nat":
        return VarDecl(var0(), NatType(_parse_bounds(opt_child1())))
    elif t.data == "real":
        return VarDecl(var0(), RealType())
    elif t.data == "const":
        return ConstDecl(var0(), _parse_expr(_child_tree(t, 1)))
    elif t.data == "rparam":
        parameters[var0()] = RealType()
        return ParameterDecl(var0(), RealType())
    elif t.data == "nparam":
        parameters[var0()] = NatType(bounds=None)
        return ParameterDecl(var0(), NatType(bounds=None))
    else:
        raise Exception(f'invalid AST: {t.data}')


def _parse_declarations(t: Tree) -> List[Decl]:
    assert t.data == "declarations"
    return [_parse_declaration(_as_tree(d)) for d in t.children]


def _parse_expr(t: Tree) -> Expr:
    def expr0() -> Expr:
        return _parse_expr(_child_tree(t, 0))

    def expr1() -> Expr:
        return _parse_expr(_child_tree(t, 1))

    if t.data == 'literal':
        return _parse_literal(_child_tree(t, 0))
    elif t.data == 'var':
        name = _parse_var(_child_tree(t, 0))
        return VarExpr(name, name in parameters)
    elif t.data == 'or':
        return BinopExpr(Binop.OR, expr0(), expr1())
    elif t.data == 'and':
        return BinopExpr(Binop.AND, expr0(), expr1())
    elif t.data == 'leq':
        return BinopExpr(Binop.LEQ, expr0(), expr1())
    elif t.data == 'le':
        return BinopExpr(Binop.LE, expr0(), expr1())
    elif t.data == 'geq':
        return BinopExpr(Binop.GEQ, expr0(), expr1())
    elif t.data == 'ge':
        return BinopExpr(Binop.GE, expr0(), expr1())
    elif t.data == 'eq':
        return BinopExpr(Binop.EQ, expr0(), expr1())
    elif t.data == 'plus':
        return BinopExpr(Binop.PLUS, expr0(), expr1())
    elif t.data == 'minus':
        return BinopExpr(Binop.MINUS, expr0(), expr1())
    elif t.data == 'times':
        return BinopExpr(Binop.TIMES, expr0(), expr1())
    elif t.data == 'power':
        return BinopExpr(Binop.POWER, expr0(), expr1())
    elif t.data == 'mod':
        return BinopExpr(Binop.MODULO, expr0(), expr1())
    elif t.data == 'divide':
        return _parse_fraction(expr0(), expr1())
    elif t.data == 'likely':
        prob_expr = expr1()
        if not isinstance(prob_expr, RealLitExpr):
            raise Exception(
                f"Probability annotation must be a probability literal: {t}")
        # We return a _LikelyExpr here, which is not in the Expr union type, but
        # we'll remove all occurrences later, so just ignore the types here.
        # It's a bit nasty, but gets the job done.
        return _LikelyExpr(expr0(), prob_expr)  # type:ignore
    elif t.data == 'neg':
        return UnopExpr(Unop.NEG, expr0())
    elif t.data == 'iverson':
        return UnopExpr(Unop.IVERSON, expr0())
    elif t.data == 'parens':
        return _parse_expr(_child_tree(t, 0))
    else:
        raise Exception(f'invalid AST: {t.data}')


def _parse_fraction(num: Expr, denom: Expr) -> Union[RealLitExpr, BinopExpr]:
    if isinstance(num, NatLitExpr) and isinstance(denom, NatLitExpr):
        return RealLitExpr(Fraction(num.value, denom.value))
    return BinopExpr(Binop.DIVIDE, num, denom)


def _parse_literal(t: Tree) -> Expr:
    if t.data == 'true':
        return BoolLitExpr(True)
    elif t.data == 'false':
        return BoolLitExpr(False)
    elif t.data == 'nat':
        return NatLitExpr(int(_child_str(t, 0)))
    elif t.data == 'real':
        return RealLitExpr(Decimal(_child_str(t, 0)))
    elif t.data == 'infinity':
        return RealLitExpr.infinity()
    else:
        raise Exception(f'invalid AST: {t.data}')


def _parse_distribution(t: Tree) -> Expr:
    assert t.data in distributions
    param_count, constructor = distributions[t.data]
    params = []
    for i in range(param_count):
        param = _parse_expr(_child_tree(t, i))
        if has_variable(param):
            raise SyntaxError(
                "In distribution parameter expressions, no variables are allowed. - Forgot parameter declaration?")
        params.append(param)
    return constructor(*params)


def _parse_rvalue(t: Tree) -> Expr:
    if t.data in distributions:
        return _parse_distribution(t)

    elif t.data == "iid":
        return IidSampleExpr(_parse_rvalue(_child_tree(t, 0)), VarExpr(_parse_var(_child_tree(t, 1))))

    # otherwise we have an expression, but it may contain _LikelyExprs, which we
    # need to parse.
    expr = _parse_expr(_child_tree(t, 0))
    if isinstance(expr, BinopExpr) and expr.operator == Binop.PLUS:
        operands = expr.flatten()
        likely_operands: List[_LikelyExpr] = [
            operand for operand in operands
            if isinstance(operand, _LikelyExpr)
        ]
        if len(likely_operands) > 0:
            if len(likely_operands) != len(operands):
                raise Exception(
                    f"Failed to parse categorical expression: each term in {t} must have an associated probability"
                )
            categories: List[Tuple[Expr, RealLitExpr]] = [
                (operand.value, operand.prob) for operand in likely_operands
            ]
            return CategoricalExpr(categories)

    # We didn't find a summation of _LikelyExprs, make sure there are no
    # _LikelyExprs nested somewhere.
    for nested_expr_ref in walk_expr(Walk.DOWN, Mut.alloc(expr)):
        if isinstance(nested_expr_ref.val, _LikelyExpr):
            raise Exception(
                f"Illegal place for a probability annotation: {expr}")

    # There are no _LikelyExprs anywhere, just return the expression as-is.
    return expr


def _parse_instr(t: Tree) -> Instr:
    if t.data == 'skip':
        return SkipInstr()
    elif t.data == 'while':
        return WhileInstr(_parse_expr(_child_tree(t, 0)),
                          _parse_instrs(_child_tree(t, 1)))
    elif t.data == 'if':
        return IfInstr(_parse_expr(_child_tree(t, 0)),
                       _parse_instrs(_child_tree(t, 1)),
                       _parse_instrs(_child_tree(t, 2)))
    elif t.data == 'assign':
        variable = _parse_var(_child_tree(t, 0))
        if variable in parameters:
            raise SyntaxError("Parameters must not be assigned a new value.")
        return AsgnInstr(_parse_var(_child_tree(t, 0)),
                         _parse_rvalue(_child_tree(t, 1)))
    elif t.data == 'choice':
        return ChoiceInstr(_parse_expr(_child_tree(t, 1)),
                           _parse_instrs(_child_tree(t, 0)),
                           _parse_instrs(_child_tree(t, 2)))
    elif t.data == 'tick':
        return TickInstr(_parse_expr(_child_tree(t, 0)))
    elif t.data == 'observe':
        return ObserveInstr(_parse_expr(_child_tree(t, 0)))
    elif t.data == 'loop':
        return LoopInstr(NatLitExpr(value=int(t.children[0])), _parse_instrs(_child_tree(t, 1)))
    else:
        raise Exception(f'invalid AST: {t.data}')


def _parse_instrs(t: Tree) -> List[Instr]:
    assert t.data in ["instructions", "block"]
    return [_parse_instr(_as_tree(t)) for t in t.children]


def _parse_queries(t: Tree) -> List[Instr]:
    assert t.data == "queries"
    return [_parse_query(_as_tree(t)) for t in t.children]


def _parse_query(t: Tree):
    if t.data == 'expectation':
        return ExpectationInstr(_parse_expr(_child_tree(t, 0)))
    elif t.data == 'prquery':
        return ProbabilityQueryInstr(_parse_expr(_child_tree(t, 0)))
    elif t.data == 'print':
        return PrintInstr()
    elif t.data == 'optimize':
        mode = _parse_var(_child_tree(t, 2))
        if mode == "MAX":
            opt_type = OptimizationType.MAXIMIZE
        elif mode == "MIN":
            opt_type = OptimizationType.MINIMIZE
        else:
            raise SyntaxError(f"The Optimization can either be 'MAX' or 'MIN', but not {mode}")
        parameter = _parse_var(_child_tree(t, 1))
        if parameter not in parameters:
            raise SyntaxError(
                f"In Optimization queries, the variable can only be a program parameter, was {parameter}.")
        return OptimizationQuery(_parse_expr(_child_tree(t, 0)), parameter, opt_type)
    elif t.data == "plot":
        if len(t.children) == 3:
            lit = _parse_literal(_child_tree(t, 2))
            if isinstance(lit, BoolLitExpr):
                raise SyntaxError("Plot instructions cannot handle boolean literals as arguments")
            if t.children[2].data == 'real' or t.children[2].data == 'infinity':
                return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))),
                                 VarExpr(_parse_var(_child_tree(t, 1))),
                                 prob=lit)
            else:
                return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))),
                                 VarExpr(_parse_var(_child_tree(t, 1))),
                                 term_count=lit)
        elif len(t.children) == 2:
            if t.children[1].data == 'real':
                lit = _parse_literal(_child_tree(t, 1))
                return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))),
                                 prob=lit)
            elif t.children[1].data == 'nat':
                lit = _parse_literal(_child_tree(t, 1))
                return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))),
                                 term_count=lit)
            elif t.children[1].data == 'infinity':
                lit = _parse_literal(_child_tree(t, 1))
                return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))),
                                 prob=lit)
            elif t.children[1].data == 'true' or t.children[1].data == 'false':
                raise SyntaxError("Plot instruction does not support boolean operators")
            else:
                return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))),
                                 VarExpr(_parse_var(_child_tree(t, 1))),
                                 )
        else:
            return PlotInstr(VarExpr(_parse_var(_child_tree(t, 0))))
    else:
        raise Exception(f'invalid AST: {t.data}')


def _parse_program(config: ProgramConfig, t: Tree) -> Program:
    assert t.data == 'start'
    declarations = _parse_declarations(_child_tree(t, 0))
    instructions = _parse_instrs(_child_tree(t, 1))
    instructions.extend(_parse_queries(_child_tree(t, 2)))
    return Program.from_parse(config, declarations, parameters, instructions)


def parse_pgcl(code: str, config: ProgramConfig = ProgramConfig()) -> Program:
    """
    Parse a pGCL program with an optional :py:class:`probably.pgcl.ast.ProgramConfig`.

    .. doctest::

        >>> parse_pgcl("x := y")
        Program(variables={}, constants={}, instructions=[AsgnInstr(lhs='x', rhs=VarExpr('y'))])

        >>> parse_pgcl("x := unif(5, 17)").instructions[0]
        AsgnInstr(lhs='x', rhs=DUniformExpr(start=NatLitExpr(5), end=NatLitExpr(17)))

        >>> parse_pgcl("x := x : 1/3 + y : 2/3").instructions[0]
        AsgnInstr(lhs='x', rhs=CategoricalExpr(exprs=[(VarExpr('x'), RealLitExpr("1/3")), (VarExpr('y'), RealLitExpr("2/3"))]))
    """
    tree = _PARSER.parse(code)
    return _parse_program(config, tree)


def parse_expr(code: str) -> Expr:
    """
    Parse a pGCL expression.

    As a program expression, it may not contain Iverson bracket expressions.

    .. doctest::

        >>> parse_expr("x < y & z")
        BinopExpr(operator=Binop.AND, lhs=BinopExpr(operator=Binop.LE, lhs=VarExpr('x'), rhs=VarExpr('y')), rhs=VarExpr('z'))

        >>> parse_expr("[x]")
        Traceback (most recent call last):
            ...
        Exception: parse_expr: Expression may not contain an Iverson bracket expression.
    """
    tree = _PARSER.parse(code, start="expression")
    expr = _parse_expr(tree)

    for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expr)):
        if isinstance(sub_expr.val, UnopExpr):
            if sub_expr.val.operator == Unop.IVERSON:
                raise Exception(
                    "parse_expr: Expression may not contain an Iverson bracket expression."
                )

    return expr


def parse_expectation(code: str) -> Expr:
    """
    Parse a pGCL expectation. This allows all kind of expressions.

    .. doctest::

        >>> parse_expectation("[x]")
        UnopExpr(operator=Unop.IVERSON, expr=VarExpr('x'))

        >>> parse_expectation("0.2")
        RealLitExpr("0.2")

        >>> parse_expectation("1/3")
        RealLitExpr("1/3")

        >>> parse_expectation("∞")
        RealLitExpr("Infinity")
    """
    tree = _PARSER.parse(code, start="expression")
    return _parse_expr(tree)
