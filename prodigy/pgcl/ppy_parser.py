"""
----------------
Parser for .ppy
----------------

A Lark-based parser for the Python-like surface syntax ``.ppy`` (see the design
in ``docs/source/language-design.md``). It produces the same AST as
:mod:`prodigy.pgcl.parser` for everything the old syntax can express, so the type
checker, the compiler and the analysis do not change.

Sugar is translated into the existing AST:

- a missing ``else`` gives an empty ``else`` branch, ``else if`` a nested
  :class:`~prodigy.pgcl.ast.IfInstr`;
- ``switch (c) { case 1: {…} case 2, 3: {…} default: {…} }`` gives nested
  ``if (c == 1) … else if (c == 2 or c == 3) … else …``;
- ``categorical({v: p, …})`` and ``categorical(p0, …, pn)`` give a
  :class:`~prodigy.pgcl.ast.CategoricalExpr`;
- declarations at the top level of the program or of a function are collected
  into the declarations, an initial value becomes an assignment at the place of
  the declaration;
- ``loop(N)`` with a constant ``N`` uses the constant's literal value.

Two constructs have no counterpart in the old syntax. A declaration inside a
block is parsed into :class:`~prodigy.pgcl.ast.LocalDeclInstr`; its scope is
checked here (see :mod:`prodigy.pgcl.scope`), and
:func:`prodigy.pgcl.parser.parse_pgcl` lowers it into plain assignments.
Floor division ``//`` is ``Binop.FLOOR_DIVIDE``.

Errors are raised as :class:`PpySyntaxError` with line and column.

.. doctest::

    >>> parse_ppy("x: nat; x = unif(1, 6);").instructions
    [AsgnInstr(lhs='x', rhs=FunctionCallExpr(function='unif', params=([NatLitExpr(1), NatLitExpr(6)], {})))]
"""
from __future__ import annotations

import copy
import functools
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import attr
from lark import Lark, Token, Transformer, v_args
from lark.exceptions import (UnexpectedCharacters, UnexpectedEOF,
                             UnexpectedInput, UnexpectedToken, VisitError)
from lark.tree import Meta

from prodigy.pgcl.ast import (AbortInstr, AsgnInstr, Binop, BinopExpr,
                              BoolLitExpr, BoolType, Bounds, CategoricalExpr,
                              ChoiceInstr, ConstDecl, Decl, ExpectationExpr,
                              CovarianceInstr, ExpectationInstr, Expr, Function, FunctionCallExpr, FunctionDecl,
                              VarianceInstr,
                              IfInstr, InferExpr, Instr, LocalDeclInstr,
                              LoopInstr, NatLitExpr, NatType, ObserveInstr,
                              OptimizationQuery, OptimizationType,
                              ParameterDecl, PlotInstr, PrintInstr,
                              ProbabilityExpr,
                              ProbabilityQueryInstr, Program, QueryInstr,
                              RealLitExpr, RealType, SampleExpr, SkipInstr,
                              Type, Unop, UnopExpr, VarDecl,
                              VarExpr, WhileInstr)
from prodigy.pgcl.ast.types import DistributionType
from prodigy.pgcl.ast.walk import Walk, walk_expr, walk_instr_exprs, walk_instrs
from prodigy.pgcl.positions import set_span, span_from_meta
from prodigy.pgcl.scope import ScopeError, lower_block_scope
from prodigy.pgcl.syntax_errors import (ProgramSyntaxError, position,
                                        removed_construct)
from prodigy.pgcl.util.ref import Mut

PPY_GRAMMAR = r"""
    start: item*

    ?item: statement
         | "const" NAME (":" type)? "=" expression ";"     -> const_decl
         | "param" NAME ":" type ";"                       -> param_decl
         | "def" NAME "(" [def_params] ")" ("->" type)? "{" statement* "return" expression ";" "}" -> def_decl
         | "query" NAME "[" expression ("," expression)? "]" ";" -> query
         | "print" ";"                                     -> print_query
         | "plot" "(" NAME ("," NAME)? ("," plot_literal)? ")" ";" -> plot
         | "optimize" "(" expression "," NAME "," NAME ")" ";" -> optimize

    def_params: def_param ("," def_param)*
    def_param: NAME ":" type

    ?statement: "skip" ";"                                 -> skip
              | "abort" ";"                                -> abort
              | NAME "=" expression ";"                    -> assign
              | names ":" type ("=" expression)? ";"       -> var_decl
              | "observe" "(" expression ")" ";"           -> observe
              | "while" "(" expression ")" block           -> while_
              | if_statement
              | block "[" expression "]" block             -> choice
              | "loop" "(" (INT | NAME) ")" block          -> loop
              | "switch" "(" NAME ")" "{" case+ default? "}" -> switch
              | "query" block                              -> query_block
              | ";"                                        -> empty

    if_statement: "if" "(" expression ")" block ("else" (block | if_statement))?

    case: "case" expression ("," expression)* ":" block
    default: "default" ":" block

    block: "{" statement* "}"

    names: NAME ("," NAME)*

    type: "nat" bounds?  -> nat_type
        | "real"         -> real_type
        | "bool"         -> bool_type
        | "dist"         -> dist_type

    bounds: "[" expression "," expression "]"

    plot_literal: INT    -> nat_lit
                | FLOAT  -> real_lit
                | INFINITY -> infinity

    ?expression: or_expr

    ?or_expr: and_expr
            | or_expr "or" and_expr      -> or_
            | or_expr "||" and_expr      -> or_

    ?and_expr: not_expr
             | and_expr "and" not_expr   -> and_
             | and_expr "&&" not_expr    -> and_

    ?not_expr: comparison
             | "not" not_expr            -> not_
             | "!" not_expr              -> not_

    ?comparison: sum
               | sum "<=" sum            -> leq
               | sum "<" sum             -> lt
               | sum ">" sum             -> gt
               | sum ">=" sum            -> geq
               | sum "==" sum            -> eq

    ?sum: term
        | sum "+" term                   -> plus
        | sum "-" term                   -> minus

    ?term: factor
         | term "*" factor               -> times
         | term "/" factor               -> divide
         | term "//" factor              -> floor_divide
         | term "%" factor               -> mod

    ?factor: power
           | "-" factor                  -> uminus

    ?power: atom
          | atom "**" factor             -> power

    ?atom: INT                           -> nat_lit
         | FLOAT                         -> real_lit
         | INFINITY                      -> infinity
         | "true"                        -> true
         | "false"                       -> false
         | NAME                          -> var
         | NAME "(" [arguments] ")"      -> call
         | "categorical" "(" categories ")" -> categorical
         | "infer" "(" NAME ("," arguments)? ")" -> infer
         | "sample" "(" NAME ")"         -> sample
         | NAME "[" expression "]"       -> dist_query
         | "(" expression ")"

    arguments: argument ("," argument)*
    ?argument: expression
             | NAME "=" expression       -> named_argument

    ?categories: "{" category ("," category)* "}" -> category_dict
               | expression ("," expression)*     -> category_list
    category: expression ":" expression

    INFINITY: "∞" | "\\infty"

    %import common.CNAME -> NAME
    %import common.INT
    %import common.FLOAT
    %import common.WS

    COMMENT: /#[^\n]*/
    BLOCK_COMMENT: /\/\*(.|\n)*?\*\//
    %ignore COMMENT
    %ignore BLOCK_COMMENT
    %ignore WS
"""
"""The Lark grammar of ``.ppy``. Precedences follow Python: ``or`` < ``and`` <
``not`` < comparisons < ``+ -`` < ``* / // %`` < unary ``-`` < ``**``."""

PPY_KEYWORDS = frozenset({
    "abort", "and", "bool", "case", "categorical", "const", "def", "default",
    "dist", "else", "false", "if", "infer", "loop", "nat", "not", "observe",
    "optimize", "or", "param", "plot", "print", "query", "real", "return",
    "sample", "skip", "switch", "true", "while"
})
"""Words that cannot be used as names in ``.ppy``."""

_ILLEGAL_FUNCTION_NAMES = {
    'unif_d', 'unif_c', 'unif', 'geometric', 'poisson', 'logdist', 'binomial',
    'negative_binomial', 'bernoulli', 'iid', 'sample_pgf'
}

_PARSER = Lark(PPY_GRAMMAR,
               parser="lalr",
               propagate_positions=True,
               start=["start", "expression"])

_HINTS = {
    ":=": "assignments in .ppy are written 'x = e'",
    "//": "comments in .ppy start with '#'",
    "&": "use 'and' or '&&'",
    "^": "powers in .ppy are written 'x ** 2'",
    "?": "queries in .ppy are written 'query Pr[…]', 'query Ex[…]', 'query Var[…]' or 'query Cov[…, …]'",
}


class PpySyntaxError(ProgramSyntaxError):
    """A syntax error in a ``.ppy`` program, with line and column (1-based)."""


def _error(where: Union[Meta, Token, None], message: str) -> PpySyntaxError:
    if where is None or getattr(where, "line", None) is None:
        return PpySyntaxError(message, None, None)
    return PpySyntaxError(message, where.line, where.column)


@attr.s
class _VarDeclStatement:
    """Parser-internal: one or more variable declarations of one statement."""
    decls: List[VarDecl] = attr.ib()
    value: Optional[Expr] = attr.ib()
    meta: Meta = attr.ib()


@attr.s
class _TopDecl:
    """Parser-internal: a declaration only allowed at the top level."""
    decl: Decl = attr.ib()
    meta: Meta = attr.ib()


@attr.s
class _Query:
    """Parser-internal: a query statement, only allowed at the end."""
    instr: Instr = attr.ib()
    meta: Meta = attr.ib()


def _spanned(method):
    """Attaches the source span of the statement to the instruction (or query) it returns
    (see :mod:`prodigy.pgcl.positions`)."""
    @functools.wraps(method)
    def wrapper(self, meta: Meta, children: List[Any]) -> Any:
        result = method(self, meta, children)
        set_span(result.instr if isinstance(result, _Query) else result, span_from_meta(meta))
        return result
    return wrapper


_MONADIC = (CategoricalExpr, InferExpr, SampleExpr)
"""Expressions that denote a distribution, allowed only where the old syntax has an rvalue."""


def _zero_for(typ: Type) -> Optional[Expr]:
    if isinstance(typ, NatType):
        return NatLitExpr(0)
    if isinstance(typ, BoolType):
        return BoolLitExpr(False)
    return None


def _fraction(num: Expr, denom: Expr, meta: Meta) -> Expr:
    # same as the old parser: a quotient of two literals is a probability literal
    if isinstance(num, NatLitExpr) and isinstance(denom, NatLitExpr):
        if denom.value == 0:
            raise _error(meta, f"Division by zero in {num.value}/0")
        return RealLitExpr(Fraction(num.value, denom.value))
    return BinopExpr(Binop.DIVIDE, num, denom)


def _probability(expr: Expr, meta: Meta) -> RealLitExpr:
    if isinstance(expr, RealLitExpr) and not expr.is_infinite():
        return expr
    if isinstance(expr, NatLitExpr):
        return RealLitExpr(Fraction(expr.value))
    raise _error(
        meta, f"A probability of categorical(…) must be a literal such as 1/2 "
        f"or 0.5, got '{expr}' (symbolic probabilities are not implemented yet).")


# The callbacks are named after the grammar rules, and many of them get the
# same arguments; that is what Lark expects.
# pylint: disable=missing-function-docstring,too-many-public-methods,unused-argument
@v_args(meta=True)
class _ToAst(Transformer):
    """Turn a Lark parse tree of ``.ppy`` into the pGCL AST."""
    def __init__(self) -> None:
        super().__init__()
        self.pending_loops: List[Tuple[LoopInstr, str, Meta]] = []
        self.calls: List[Tuple[FunctionCallExpr, Meta]] = []
        self.parameter_counts: Dict[str, int] = {}
        self.function_decls: List[FunctionDecl] = []
        self.local_decls: Dict[int, Meta] = {}

    # --- names and checks -------------------------------------------------

    @staticmethod
    def _name(token: Token) -> str:
        name = str(token)
        if name in PPY_KEYWORDS:
            raise _error(token, f"'{name}' is a reserved word and cannot be used as a name.")
        return name

    @staticmethod
    def _plain(expr: Expr, meta: Optional[Meta]) -> Expr:
        """Reject distributions (categorical, infer, sample) inside expressions."""
        if isinstance(expr, _MONADIC):
            kind = {CategoricalExpr: "categorical(…)", InferExpr: "infer(…)",
                    SampleExpr: "sample(…)"}[type(expr)]
            raise _error(
                meta, f"{kind} is only allowed as the whole right-hand side of an "
                "assignment, an initial value, a return value or a function argument.")
        return expr

    # --- expressions ------------------------------------------------------

    def nat_lit(self, meta: Meta, children: List[Any]) -> Any:
        return NatLitExpr(int(children[0]))

    def real_lit(self, meta: Meta, children: List[Any]) -> Any:
        return RealLitExpr(Decimal(str(children[0])))

    def infinity(self, meta: Meta, children: List[Any]) -> Any:
        return RealLitExpr.infinity()

    def true(self, meta: Meta, children: List[Any]) -> Any:
        return BoolLitExpr(True)

    def false(self, meta: Meta, children: List[Any]) -> Any:
        return BoolLitExpr(False)

    def var(self, meta: Meta, children: List[Any]) -> Any:
        return VarExpr(self._name(children[0]))

    def _binop(self, operator: Binop, meta: Meta, children: List[Any]) -> Expr:
        lhs, rhs = (self._plain(child, meta) for child in children)
        return BinopExpr(operator, lhs, rhs)

    def or_(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.OR, meta, children)

    def and_(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.AND, meta, children)

    def not_(self, meta: Meta, children: List[Any]) -> Any:
        return UnopExpr(Unop.NEG, self._plain(children[0], meta))

    def leq(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.LEQ, meta, children)

    def lt(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.LT, meta, children)

    def gt(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.GT, meta, children)

    def geq(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.GEQ, meta, children)

    def eq(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.EQ, meta, children)

    def plus(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.PLUS, meta, children)

    def minus(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.MINUS, meta, children)

    def times(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.TIMES, meta, children)

    def divide(self, meta: Meta, children: List[Any]) -> Any:
        lhs, rhs = (self._plain(child, meta) for child in children)
        return _fraction(lhs, rhs, meta)

    def floor_divide(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.FLOOR_DIVIDE, meta, children)

    def mod(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.MODULO, meta, children)

    def power(self, meta: Meta, children: List[Any]) -> Any:
        return self._binop(Binop.POWER, meta, children)

    def uminus(self, meta: Meta, children: List[Any]) -> Any:
        # the same representation as the old parser: -e is (-1) * e
        return BinopExpr(Binop.TIMES, RealLitExpr(Fraction(-1)),
                         self._plain(children[0], meta))

    def arguments(self, meta: Meta, children: List[Any]) -> Any:
        positional: List[Expr] = []
        named: Dict[str, Expr] = {}
        for child in children:
            if isinstance(child, tuple):
                name, value = child
                if name in named:
                    raise _error(meta, f"The argument '{name}' is given twice.")
                named[name] = value
            elif named:
                raise _error(meta, "Positional arguments must come before named arguments.")
            else:
                positional.append(child)
        return positional, named

    def named_argument(self, meta: Meta, children: List[Any]) -> Any:
        return self._name(children[0]), children[1]

    def call(self, meta: Meta, children: List[Any]) -> Any:
        name = self._name(children[0])
        params = children[1] if children[1] is not None else ([], {})
        expr = FunctionCallExpr(name, params)
        self.calls.append((expr, meta))
        return expr

    def infer(self, meta: Meta, children: List[Any]) -> Any:
        name = self._name(children[0])
        params = children[1] if len(children) > 1 else ([], {})
        call = FunctionCallExpr(name, params)
        self.calls.append((call, meta))
        return InferExpr(call)

    def sample(self, meta: Meta, children: List[Any]) -> Any:
        return SampleExpr(self._name(children[0]))

    def dist_query(self, meta: Meta, children: List[Any]) -> Any:
        kind = str(children[0])
        inner = self._plain(children[1], meta)
        if kind == "Pr":
            return ProbabilityExpr(inner)
        if kind == "Ex":
            return ExpectationExpr(inner)
        raise _error(children[0], f"Unknown expression '{kind}[…]': use 'Pr[d == k]' or 'Ex[d]' "
                     "for a variable d of type dist.")

    def category(self, meta: Meta, children: List[Any]) -> Any:
        value, prob = children
        return self._plain(value, meta), _probability(self._plain(prob, meta), meta)

    def category_dict(self, meta: Meta, children: List[Any]) -> Any:
        return list(children)

    def category_list(self, meta: Meta, children: List[Any]) -> Any:
        return [(NatLitExpr(index), _probability(self._plain(prob, meta), meta))
                for index, prob in enumerate(children)]

    def categorical(self, meta: Meta, children: List[Any]) -> Any:
        categories = children[0]
        total = sum(prob.to_fraction() for _, prob in categories)
        if total != 1:
            raise _error(
                meta, f"The probabilities of categorical(…) must sum to 1, they sum to {total}.")
        return CategoricalExpr(categories)

    # --- types and declarations -------------------------------------------

    def bounds(self, meta: Meta, children: List[Any]) -> Any:
        return Bounds(self._plain(children[0], meta), self._plain(children[1], meta))

    def nat_type(self, meta: Meta, children: List[Any]) -> Any:
        return NatType(children[0] if children else None)

    def real_type(self, meta: Meta, children: List[Any]) -> Any:
        return RealType()

    def bool_type(self, meta: Meta, children: List[Any]) -> Any:
        return BoolType()

    def dist_type(self, meta: Meta, children: List[Any]) -> Any:
        return DistributionType()

    def names(self, meta: Meta, children: List[Any]) -> Any:
        return [self._name(child) for child in children]

    def var_decl(self, meta: Meta, children: List[Any]) -> Any:
        names, typ = children[0], children[1]
        value = children[2] if len(children) > 2 else None
        if len(set(names)) != len(names):
            raise _error(meta, "A name is declared twice in one declaration.")
        if value is not None and len(names) > 1:
            raise _error(meta, "An initial value needs a single variable; declare the variables separately.")
        return _VarDeclStatement([VarDecl(name, copy.deepcopy(typ)) for name in names],
                                 value, meta)

    def const_decl(self, meta: Meta, children: List[Any]) -> Any:
        name = self._name(children[0])
        typ: Optional[Type] = children[1] if len(children) == 3 else None
        value = self._plain(children[-1], meta)
        if typ is not None:
            self._check_constant_type(name, typ, value, meta)
        return _TopDecl(ConstDecl(name, value), meta)

    @staticmethod
    def _check_constant_type(name: str, typ: Type, value: Expr, meta: Meta) -> None:
        # the AST keeps no type for constants; reject what clearly contradicts it
        if isinstance(typ, NatType) and typ.bounds is not None:
            raise _error(meta, f"The constant '{name}' cannot have bounds.")
        if isinstance(typ, DistributionType):
            raise _error(meta, f"The constant '{name}' cannot be of type dist.")
        literal = isinstance(value, (NatLitExpr, RealLitExpr, BoolLitExpr))
        fits = {
            NatType: (NatLitExpr,),
            RealType: (NatLitExpr, RealLitExpr),
            BoolType: (BoolLitExpr,),
        }[type(typ)]
        if literal and not isinstance(value, fits):
            raise _error(meta, f"The value '{value}' of the constant '{name}' does not have the declared type.")

    def param_decl(self, meta: Meta, children: List[Any]) -> Any:
        name = self._name(children[0])
        typ = children[1]
        if isinstance(typ, NatType):
            if typ.bounds is not None:
                raise _error(meta, f"The parameter '{name}' cannot have bounds.")
        elif not isinstance(typ, RealType):
            raise _error(meta, f"The parameter '{name}' must be of type real or nat.")
        return _TopDecl(ParameterDecl(name, typ), meta)

    def def_param(self, meta: Meta, children: List[Any]) -> Any:
        return VarDecl(self._name(children[0]), children[1])

    def def_params(self, meta: Meta, children: List[Any]) -> Any:
        return list(children)

    def def_decl(self, meta: Meta, children: List[Any]) -> Any:
        name = self._name(children[0])
        if name in _ILLEGAL_FUNCTION_NAMES:
            raise _error(meta, f"'{name}' is a built-in distribution and cannot be redefined.")
        params: List[VarDecl] = children[1] or []
        index = 2
        if isinstance(children[index], (NatType, RealType, BoolType, DistributionType)):
            if children[index] != NatType(None):
                raise _error(meta, "Functions can only return nat for now (not implemented yet).")
            index += 1
        body = [child for child in children[index:-1] if child is not None]
        returns = children[-1]
        declarations: List[Decl] = list(params)
        self.parameter_counts[name] = len(params)
        instructions = _hoist(body, declarations, zero_init_all=True)
        names = [decl.var for decl in declarations]
        duplicates = {var for var in names if names.count(var) > 1}
        if duplicates:
            raise _error(meta, f"'{sorted(duplicates)[0]}' is declared twice in the function '{name}'.")
        variables = [decl for decl in declarations if isinstance(decl, VarDecl)]
        decl = FunctionDecl(name, Function.from_parse(variables, instructions, returns))
        self.function_decls.append(decl)
        return _TopDecl(decl, meta)

    # --- statements -------------------------------------------------------

    @_spanned
    def skip(self, meta: Meta, children: List[Any]) -> Any:
        return SkipInstr()

    @_spanned
    def abort(self, meta: Meta, children: List[Any]) -> Any:
        return AbortInstr()

    def empty(self, meta: Meta, children: List[Any]) -> Any:
        return None

    @_spanned
    def assign(self, meta: Meta, children: List[Any]) -> Any:
        return AsgnInstr(self._name(children[0]), children[1])

    @_spanned
    def observe(self, meta: Meta, children: List[Any]) -> Any:
        return ObserveInstr(self._plain(children[0], meta))

    def block(self, meta: Meta, children: List[Any]) -> Any:
        instrs: List[Instr] = []
        for child in children:
            if child is None:
                continue
            if isinstance(child, _VarDeclStatement):
                for decl in child.decls:
                    local = set_span(LocalDeclInstr(decl, child.value), span_from_meta(child.meta))
                    self.local_decls[id(local)] = child.meta
                    instrs.append(local)
            else:
                instrs.append(child)
        return instrs

    @_spanned
    def while_(self, meta: Meta, children: List[Any]) -> Any:
        return WhileInstr(self._plain(children[0], meta), children[1])

    @_spanned
    def if_statement(self, meta: Meta, children: List[Any]) -> Any:
        cond = self._plain(children[0], meta)
        if len(children) < 3:
            return IfInstr(cond, children[1], [])
        other = children[2]
        return IfInstr(cond, children[1], other if isinstance(other, list) else [other])

    @_spanned
    def choice(self, meta: Meta, children: List[Any]) -> Any:
        return ChoiceInstr(self._plain(children[1], meta), children[0], children[2])

    @_spanned
    def loop(self, meta: Meta, children: List[Any]) -> Any:
        count, body = children
        if count.type == "INT":
            return LoopInstr(NatLitExpr(int(count)), body)
        instr = LoopInstr(NatLitExpr(0), body)
        self.pending_loops.append((instr, self._name(count), meta))
        return instr

    def case(self, meta: Meta, children: List[Any]) -> Any:
        values = [self._plain(value, meta) for value in children[:-1]]
        return values, children[-1]

    def default(self, meta: Meta, children: List[Any]) -> Any:
        return children[0]

    @_spanned
    def switch(self, meta: Meta, children: List[Any]) -> Any:
        var = self._name(children[0])
        rest = children[1:]
        otherwise: List[Instr] = []
        if rest and not isinstance(rest[-1], tuple):
            otherwise = rest[-1]
            rest = rest[:-1]
        for values, body in reversed(rest):
            conditions = [BinopExpr(Binop.EQ, VarExpr(var), value) for value in values]
            cond = BinopExpr.reduce(Binop.OR, conditions)
            otherwise = [IfInstr(cond, body, otherwise)]
        return otherwise[0]

    @_spanned
    def query_block(self, meta: Meta, children: List[Any]) -> Any:
        return QueryInstr(children[0])

    # --- queries ----------------------------------------------------------

    @_spanned
    def query(self, meta: Meta, children: List[Any]) -> Any:
        kind = str(children[0])
        expr = self._plain(children[1], meta)
        if kind == "Cov":
            if len(children) < 3:
                raise _error(children[0], "'query Cov[…]' takes two expressions: 'query Cov[x, y]'.")
            return _Query(CovarianceInstr(expr, self._plain(children[2], meta)), meta)
        if len(children) > 2:
            raise _error(children[0], f"'query {kind}[…]' takes one expression.")
        if kind == "Pr":
            return _Query(ProbabilityQueryInstr(expr), meta)
        if kind == "Ex":
            return _Query(ExpectationInstr(expr), meta)
        if kind == "Var":
            return _Query(VarianceInstr(expr), meta)
        raise _error(children[0], f"Unknown query '{kind}': use 'query Pr[…]', 'query Ex[…]', "
                     "'query Var[…]' or 'query Cov[…, …]'.")

    @_spanned
    def print_query(self, meta: Meta, children: List[Any]) -> Any:
        return _Query(PrintInstr(), meta)

    @_spanned
    def plot(self, meta: Meta, children: List[Any]) -> Any:
        variables = [VarExpr(self._name(child)) for child in children if isinstance(child, Token)]
        literals = [child for child in children if not isinstance(child, Token)]
        kwargs: Dict[str, Any] = {}
        if literals:
            literal = literals[0]
            if isinstance(literal, NatLitExpr):
                kwargs["term_count"] = literal
            else:
                kwargs["prob"] = literal
        if len(variables) > 1:
            kwargs["var_2"] = variables[1]
        return _Query(PlotInstr(variables[0], **kwargs), meta)

    @_spanned
    def optimize(self, meta: Meta, children: List[Any]) -> Any:
        expr = self._plain(children[0], meta)
        parameter = self._name(children[1])
        mode = str(children[2])
        if mode not in ("MAX", "MIN"):
            raise _error(children[2], f"The optimization is either MAX or MIN, not {mode}.")
        kind = OptimizationType.MAXIMIZE if mode == "MAX" else OptimizationType.MINIMIZE
        return _Query(OptimizationQuery(expr, parameter, kind), meta)

    # --- program ----------------------------------------------------------

    def start(self, meta: Meta, children: List[Any]) -> Any:
        declarations: List[Decl] = []
        instructions: List[Instr] = []
        queries: List[Instr] = []
        statements: List[Any] = []
        first_query: Optional[_Query] = None
        for child in children:
            if child is None:
                continue
            if isinstance(child, _Query):
                queries.append(child.instr)
                first_query = first_query or child
                continue
            if first_query is not None:
                raise _error(first_query.meta,
                             "Queries must come at the end of the program, but a statement follows this one.")
            statements.append(child)
        instructions = _hoist(statements, declarations, zero_init_all=False)
        constants = {decl.var: decl.value for decl in declarations if isinstance(decl, ConstDecl)}
        self._resolve_loops(constants)
        self._check_calls()
        program = Program.from_parse(declarations, instructions + queries)
        try:
            lower_block_scope(program)
        except ScopeError as exc:
            where = None if exc.declaration is None else self.local_decls.get(id(exc.declaration))
            raise _error(where, exc.message) from None
        return program

    def _resolve_loops(self, constants: Dict[str, Expr]) -> None:
        for instr, name, meta in self.pending_loops:
            value = constants.get(name)
            if not isinstance(value, NatLitExpr):
                raise _error(meta, f"loop({name}) needs a constant with a literal nat value.")
            instr.iterations = NatLitExpr(value.value)

    def _check_calls(self) -> None:
        # A function's own variables follow its parameters in its declarations
        # (the old AST has no separate list); only the parameters may be passed.
        functions = {decl.var: decl.body for decl in self.function_decls}
        for call, meta in self.calls:
            function = functions.get(call.function)
            if function is None:
                continue  # built-in distributions are checked by the type checker
            positional, named = call.params
            parameters = [decl.var for decl in function.declarations]
            count = self.parameter_counts[call.function]
            if len(positional) > count:
                raise _error(meta, f"'{call.function}' takes {count} argument(s), got {len(positional)}.")
            allowed = parameters[:count]
            for name in named:
                if name not in allowed:
                    raise _error(meta, f"'{call.function}' has no parameter '{name}'.")
                if name in allowed[:len(positional)]:
                    raise _error(meta, f"The argument '{name}' of '{call.function}' is given twice.")


def _hoist(items: List[Any], declarations: List[Decl], *, zero_init_all: bool) -> List[Instr]:
    """
    Move the declarations of a program or function body into ``declarations``
    and return the instructions. An initial value becomes an assignment at the
    place of the declaration; a variable declared after the first instruction
    (or, in a function, any local variable) starts at 0.
    """
    instructions: List[Instr] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, _TopDecl):
            _check_not_used_before(item.decl.var, instructions, None, item.meta)
            declarations.append(item.decl)
        elif isinstance(item, _VarDeclStatement):
            for decl in item.decls:
                _check_not_used_before(decl.var, instructions, item.value, item.meta)
                declarations.append(decl)
                value = item.value
                if value is None and (zero_init_all or instructions):
                    value = _zero_for(decl.typ)
                if value is not None:
                    instructions.append(set_span(AsgnInstr(decl.var, copy.deepcopy(value)),
                                                 span_from_meta(item.meta)))
        else:
            instructions.append(item)
    return instructions


def _check_not_used_before(name: str, instructions: List[Instr],
                           value: Optional[Expr], meta: Meta) -> None:
    if name in _names_in(instructions, value):
        raise _error(meta, f"'{name}' is used before its declaration.")


def _names_in(instructions: List[Instr], value: Optional[Expr]) -> Set[str]:
    names: Set[str] = set()
    exprs: List[Expr] = [] if value is None else [value]
    for ref in walk_instrs(Walk.DOWN, instructions):
        instr = ref.val
        if isinstance(instr, AsgnInstr):
            names.add(instr.lhs)
        if isinstance(instr, OptimizationQuery):
            names.add(instr.parameter)
        exprs.extend(sub.val for sub in walk_instr_exprs(Walk.DOWN, instr))
    for expr in exprs:
        for sub in walk_expr(Walk.DOWN, Mut.alloc(expr)):
            if isinstance(sub.val, VarExpr):
                names.add(sub.val.var)
            elif isinstance(sub.val, SampleExpr):
                names.add(sub.val.dist)
    return names


def _describe_terminal(name: str) -> str:
    try:
        pattern = _PARSER.get_terminal(name).pattern
    except KeyError:
        return name
    if pattern.type == "str":
        return f"'{pattern.value}'"
    return {"NAME": "a name", "INT": "a number", "FLOAT": "a number",
            "INFINITY": "'∞'"}.get(name, name.lower())


def _describe_expected(name: str) -> str:
    if name == "$END":
        return "the end of the program"
    return _describe_terminal(name)


def _expected(names: Set[str]) -> str:
    described = sorted({_describe_expected(name) for name in names})
    if not described:
        return ""
    if len(described) > 8:
        return ", expected one of " + ", ".join(described[:8]) + ", …"
    return ", expected " + " or ".join(described)


_COMMENTS = re.compile(r"#[^\n]*|/\*[\s\S]*?\*/")


def _unterminated_comment(source: str) -> Optional[PpySyntaxError]:
    """A '/*' outside of comments that is never closed (the parse fails on it)."""
    start = _COMMENTS.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), source).find("/*")
    if start < 0:
        return None
    line = source.count("\n", 0, start) + 1
    return PpySyntaxError("The comment '/*' is never closed with '*/'.", line,
                          start - (source.rfind("\n", 0, start) + 1) + 1)


def _lark_error(source: str, exc: UnexpectedInput) -> PpySyntaxError:
    unterminated = _unterminated_comment(source)
    if unterminated is not None:
        return unterminated
    if isinstance(exc, UnexpectedToken) and exc.token.type == "$END":
        return PpySyntaxError("The program ends unexpectedly" + _expected(exc.expected) + ".",
                              *_end_position(source))
    if isinstance(exc, UnexpectedEOF):
        return PpySyntaxError("The program ends unexpectedly" + _expected(set(exc.expected)) + ".",
                              *_end_position(source))
    removed = _removed_construct(source, exc)
    if removed is not None:
        message, start = removed
        return PpySyntaxError(message + ".", *position(source, start))
    if isinstance(exc, UnexpectedToken):
        text = str(exc.token)
        message = f"Unexpected '{text}'" + _expected(exc.expected)
        start = exc.token.start_pos
        if text == "=" and start is not None and source[start - 1:start] == ":":
            text = ":="
    elif isinstance(exc, UnexpectedCharacters):
        offset = exc.pos_in_stream or 0
        text = source[offset:offset + 2]
        if text not in _HINTS:
            text = text[:1]
        message = f"Unexpected '{text}'" + _expected(set(exc.allowed or ()))
    else:
        text = ""
        message = str(exc)
    hint = _HINTS.get(text)
    if hint is not None:
        message += f" ({hint})"
    return PpySyntaxError(message + ".", exc.line, exc.column)


def _removed_construct(source: str, exc: UnexpectedInput) -> Optional[Tuple[str, int]]:
    if isinstance(exc, UnexpectedToken):
        return removed_construct(source, exc.token.start_pos, "INT" in exc.expected)
    if isinstance(exc, UnexpectedCharacters):
        return removed_construct(source, exc.pos_in_stream, "INT" in (exc.allowed or ()))
    return None


def _end_position(source: str) -> Tuple[int, int]:
    lines = source.split("\n")
    return len(lines), len(lines[-1]) + 1


def _transform(source: str, start: str) -> Any:
    try:
        tree = _PARSER.parse(source, start=start)
    except UnexpectedInput as exc:
        raise _lark_error(source, exc) from None
    transformer = _ToAst()
    try:
        return transformer.transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, PpySyntaxError):
            raise exc.orig_exc from None
        raise


def parse_ppy(code: str) -> Program:
    """
    Parse a program in the ``.ppy`` syntax.

    .. doctest::

        >>> parse_ppy("x: nat = 3; if (x == 3) { skip; }").instructions[1]
        IfInstr(cond=BinopExpr(operator=Binop.EQ, lhs=VarExpr('x'), rhs=NatLitExpr(3)), true=[SkipInstr()], false=[])

        >>> parse_ppy("x: nat; x = 1 +;")
        Traceback (most recent call last):
            ...
        prodigy.pgcl.ppy_parser.PpySyntaxError: line 1, column 16: Unexpected ';', expected ... or a number.
    """
    program = _transform(code, "start")
    assert isinstance(program, Program)
    return program


def parse_ppy_expr(code: str) -> Expr:
    """
    Parse an expression in the ``.ppy`` syntax.

    .. doctest::

        >>> parse_ppy_expr("not z or x")
        BinopExpr(operator=Binop.OR, lhs=UnopExpr(operator=Unop.NEG, expr=VarExpr('z')), rhs=VarExpr('x'))
    """
    expr = _transform(code, "expression")
    return _ToAst._plain(expr, None)  # pylint: disable=protected-access
