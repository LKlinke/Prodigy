"""
------------------------
Pretty printer for .ppy
------------------------

Prints a pGCL AST in the ``.ppy`` syntax, with the canonical spelling ``and``,
``or`` and ``not``. Parsing the output with :func:`prodigy.pgcl.ppy_parser.parse_ppy`
gives back an equal AST; the converter from ``.pgcl`` is built on this.

Comments are not part of the AST; :mod:`prodigy.pgcl.converter` carries them over.

.. doctest::

    >>> from prodigy.pgcl.parser import parse_pgcl
    >>> print(program_to_ppy(parse_pgcl("nat x; rparam p; {x := x + 1} [p] {x := 0}; ?Pr[x = 1]")), end="")
    x: nat;
    param p: real;
    {
        x = x + 1;
    } [p] {
        x = 0;
    }
    query Pr[x == 1];
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import List, Tuple

from prodigy.pgcl.ast import (AbortInstr, AsgnInstr, Binop, BinopExpr,
                              BoolLitExpr, BoolType, CategoricalExpr,
                              ChoiceInstr, ConstDecl, Decl, ExpectationInstr,
                              Expr, FunctionCallExpr, FunctionDecl, IfInstr,
                              InferExpr, Instr, LocalDeclInstr, LoopInstr,
                              NatLitExpr, NatType, ObserveInstr,
                              OptimizationQuery, OptimizationType,
                              ParameterDecl, PlotInstr, PrintInstr,
                              ProbabilityQueryInstr, Program, QueryInstr,
                              RealLitExpr, RealType, SampleExpr, SkipInstr,
                              Type, Unop, UnopExpr, VarDecl,
                              VarExpr, WhileInstr)
from prodigy.pgcl.ast.types import DistributionType
from prodigy.pgcl.ppy_parser import PPY_KEYWORDS

_INDENT = "    "


class PpyPrintError(ValueError):
    """The AST cannot be written in ``.ppy`` (e.g. a name that is a keyword there)."""


# Precedence levels, as in the grammar of prodigy.pgcl.ppy_parser
_OR, _AND, _NOT, _COMPARE, _SUM, _PRODUCT, _UNARY, _POWER, _ATOM = range(1, 10)

_BINOPS = {
    Binop.OR: ("or", _OR),
    Binop.AND: ("and", _AND),
    Binop.LEQ: ("<=", _COMPARE),
    Binop.LT: ("<", _COMPARE),
    Binop.GT: (">", _COMPARE),
    Binop.GEQ: (">=", _COMPARE),
    Binop.EQ: ("==", _COMPARE),
    Binop.PLUS: ("+", _SUM),
    Binop.MINUS: ("-", _SUM),
    Binop.TIMES: ("*", _PRODUCT),
    Binop.DIVIDE: ("/", _PRODUCT),
    Binop.FLOOR_DIVIDE: ("//", _PRODUCT),
    Binop.MODULO: ("%", _PRODUCT),
    Binop.POWER: ("**", _POWER),
}


def _name(name: str) -> str:
    if name in PPY_KEYWORDS:
        raise PpyPrintError(
            f"'{name}' is a reserved word in .ppy and cannot be used as a name; rename it first.")
    return name


def _is_negation(expr: BinopExpr) -> bool:
    # both parsers represent -e as (-1) * e
    lhs = expr.lhs
    return (expr.operator == Binop.TIMES and isinstance(lhs, RealLitExpr)
            and isinstance(lhs.value, Fraction) and lhs.value == -1)


def _real(expr: RealLitExpr) -> Tuple[str, int]:
    if expr.is_infinite():
        if expr.value < 0:
            raise PpyPrintError("Negative infinity cannot be written in .ppy.")
        return "∞", _ATOM
    value = expr.value
    if value < 0:
        raise PpyPrintError(f"The negative literal {value} cannot be written in .ppy.")
    if isinstance(value, Fraction):
        # a quotient of two nat literals is parsed back into this Fraction
        return f"{value.numerator}/{value.denominator}", _PRODUCT
    assert isinstance(value, Decimal)
    text = str(value)
    if not any(char in text for char in ".eE"):
        text += "."  # Decimal("5.") prints as "5", which would parse as nat
    return text, _ATOM


def _wrap(expr: Expr, minimum: int) -> str:
    text, level = _expr(expr)
    return text if level >= minimum else f"({text})"


def _call_arguments(positional: List[Expr], named: dict) -> List[str]:
    return [expr_to_ppy(arg) for arg in positional] + \
        [f"{_name(var)} = {expr_to_ppy(arg)}" for var, arg in named.items()]


# pylint: disable=too-many-return-statements
def _expr(expr: Expr) -> Tuple[str, int]:
    """Print an expression and return it with its precedence level."""
    if isinstance(expr, VarExpr):
        return _name(expr.var), _ATOM
    if isinstance(expr, BoolLitExpr):
        return ("true" if expr.value else "false"), _ATOM
    if isinstance(expr, NatLitExpr):
        if expr.value < 0:
            raise PpyPrintError(f"The negative literal {expr.value} cannot be written in .ppy.")
        return str(expr.value), _ATOM
    if isinstance(expr, RealLitExpr):
        return _real(expr)
    if isinstance(expr, UnopExpr):
        if expr.operator == Unop.NEG:
            return f"not {_wrap(expr.expr, _NOT)}", _NOT
        raise PpyPrintError(f"The unary operator {expr.operator} has no .ppy syntax.")
    if isinstance(expr, BinopExpr):
        if _is_negation(expr):
            return f"-{_wrap(expr.rhs, _UNARY)}", _UNARY
        symbol, level = _BINOPS[expr.operator]
        if level == _POWER:
            # right associative; the exponent may be a unary minus
            return f"{_wrap(expr.lhs, _ATOM)} ** {_wrap(expr.rhs, _UNARY)}", level
        if level == _COMPARE:
            # comparisons do not chain
            return f"{_wrap(expr.lhs, _SUM)} {symbol} {_wrap(expr.rhs, _SUM)}", level
        # left associative
        return f"{_wrap(expr.lhs, level)} {symbol} {_wrap(expr.rhs, level + 1)}", level
    if isinstance(expr, CategoricalExpr):
        pairs = ", ".join(f"{expr_to_ppy(value)}: {expr_to_ppy(prob)}" for value, prob in expr.exprs)
        return f"categorical({{{pairs}}})", _ATOM
    if isinstance(expr, FunctionCallExpr):
        return f"{_name(expr.function)}({', '.join(_call_arguments(*expr.params))})", _ATOM
    if isinstance(expr, InferExpr):
        call = expr.function
        arguments = [_name(call.function)] + _call_arguments(*call.params)
        return f"infer({', '.join(arguments)})", _ATOM
    if isinstance(expr, SampleExpr):
        return f"sample({_name(expr.dist)})", _ATOM
    raise PpyPrintError(f"{type(expr).__name__} has no .ppy syntax.")


def expr_to_ppy(expr: Expr) -> str:
    """
    Print an expression in the ``.ppy`` syntax.

    .. doctest::

        >>> from prodigy.pgcl.parser import parse_expectation
        >>> expr_to_ppy(parse_expectation("not (x = 1) & y^2 < -x + 1"))
        'not x == 1 and y ** 2 < -(x + 1)'
    """
    return _expr(expr)[0]


def _type(typ: Type) -> str:
    if isinstance(typ, NatType):
        if typ.bounds is None:
            return "nat"
        return f"nat[{expr_to_ppy(typ.bounds.lower)}, {expr_to_ppy(typ.bounds.upper)}]"
    if isinstance(typ, BoolType):
        return "bool"
    if isinstance(typ, RealType):
        return "real"
    if isinstance(typ, DistributionType):
        return "dist"
    raise PpyPrintError(f"The type {typ} has no .ppy syntax.")


def _constant_type(value: Expr) -> str:
    if isinstance(value, NatLitExpr):
        return ": nat"
    if isinstance(value, RealLitExpr):
        return ": real"
    if isinstance(value, BoolLitExpr):
        return ": bool"
    return ""  # the type annotation is optional


def _indent(text: str) -> str:
    return "\n".join(_INDENT + line if line else line for line in text.split("\n"))


def _plot(instr: PlotInstr) -> str:
    arguments = [_name(instr.var_1.var)]
    if instr.var_2:
        arguments.append(_name(instr.var_2.var))
    if instr.prob:
        arguments.append(_real(instr.prob)[0])
    if instr.term_count:
        arguments.append(str(instr.term_count.value))
    return f"plot({', '.join(arguments)});"


class PpyPrinter:
    """
    Prints instructions, declarations and programs in the ``.ppy`` syntax.

    Subclasses can decorate the text of single instructions and declarations
    (the converter adds comments this way) by overriding :meth:`instr` and
    :meth:`decl`; blocks and function bodies are printed through them.
    """

    def instr(self, instr: Instr) -> str:
        """Print an instruction (or query)."""
        return self._instr(instr)

    def decl(self, decl: Decl) -> str:
        """Print a declaration."""
        return self._decl(decl)

    def program(self, program: Program) -> str:
        """Print a program, declarations first."""
        lines = [self.decl(decl) for decl in program.declarations]
        lines.extend(self.instr(instr) for instr in program.instructions)
        return "\n".join(lines) + "\n"

    def block(self, instrs: List[Instr]) -> str:
        """Print a block in curly braces."""
        if not instrs:
            return "{ }"
        lines = "\n".join(self.instr(instr) for instr in instrs)
        return "{\n" + _indent(lines) + "\n}"

    def _else(self, instrs: List[Instr]) -> str:
        if not instrs:
            return ""
        if len(instrs) == 1 and isinstance(instrs[0], IfInstr):
            text = self.instr(instrs[0])
            if text.startswith("if "):  # nothing printed before the nested if
                return f" else {text}"
        return f" else {self.block(instrs)}"

    # pylint: disable=too-many-return-statements,too-many-branches
    def _instr(self, instr: Instr) -> str:
        if isinstance(instr, SkipInstr):
            return "skip;"
        if isinstance(instr, AbortInstr):
            return "abort;"
        if isinstance(instr, AsgnInstr):
            return f"{_name(instr.lhs)} = {expr_to_ppy(instr.rhs)};"
        if isinstance(instr, WhileInstr):
            return f"while ({expr_to_ppy(instr.cond)}) {self.block(instr.body)}"
        if isinstance(instr, IfInstr):
            return f"if ({expr_to_ppy(instr.cond)}) {self.block(instr.true)}{self._else(instr.false)}"
        if isinstance(instr, ChoiceInstr):
            return f"{self.block(instr.lhs)} [{expr_to_ppy(instr.prob)}] {self.block(instr.rhs)}"
        if isinstance(instr, LoopInstr):
            return f"loop({instr.iterations.value}) {self.block(instr.body)}"
        if isinstance(instr, ObserveInstr):
            return f"observe({expr_to_ppy(instr.cond)});"
        if isinstance(instr, QueryInstr):
            return f"query {self.block(instr.instrs)}"
        if isinstance(instr, LocalDeclInstr):
            value = "" if instr.value is None else f" = {expr_to_ppy(instr.value)}"
            return f"{_name(instr.decl.var)}: {_type(instr.decl.typ)}{value};"
        if isinstance(instr, ProbabilityQueryInstr):
            return f"query Pr[{expr_to_ppy(instr.expr)}];"
        if isinstance(instr, ExpectationInstr):
            return f"query Ex[{expr_to_ppy(instr.expr)}];"
        if isinstance(instr, PrintInstr):
            return "print;"
        if isinstance(instr, PlotInstr):
            return _plot(instr)
        if isinstance(instr, OptimizationQuery):
            mode = "MAX" if instr.type == OptimizationType.MAXIMIZE else "MIN"
            return f"optimize({expr_to_ppy(instr.expr)}, {_name(instr.parameter)}, {mode});"
        raise PpyPrintError(f"{type(instr).__name__} has no .ppy syntax.")

    def _decl(self, decl: Decl) -> str:
        if isinstance(decl, VarDecl):
            return f"{_name(decl.var)}: {_type(decl.typ)};"
        if isinstance(decl, ConstDecl):
            return f"const {_name(decl.var)}{_constant_type(decl.value)} = {expr_to_ppy(decl.value)};"
        if isinstance(decl, ParameterDecl):
            if isinstance(decl.typ, NatType) and decl.typ.bounds is not None:
                raise PpyPrintError("A parameter with bounds has no .ppy syntax.")
            return f"param {_name(decl.var)}: {_type(decl.typ)};"
        if isinstance(decl, FunctionDecl):
            function = decl.body
            # all declarations of a function are its parameters in the old syntax
            params = ", ".join(f"{_name(var.var)}: {_type(var.typ)}" for var in function.declarations)
            body = [self.instr(instr) for instr in function.instructions]
            body.append(f"return {expr_to_ppy(function.returns)};")
            return f"def {_name(decl.var)}({params}) -> nat {{\n{_indent(chr(10).join(body))}\n}}"
        raise PpyPrintError(f"{type(decl).__name__} has no .ppy syntax.")


_PRINTER = PpyPrinter()


def instr_to_ppy(instr: Instr) -> str:
    """Print an instruction (or query) in the ``.ppy`` syntax."""
    return _PRINTER.instr(instr)


def decl_to_ppy(decl: Decl) -> str:
    """Print a declaration in the ``.ppy`` syntax."""
    return _PRINTER.decl(decl)


def program_to_ppy(program: Program) -> str:
    """Print a program in the ``.ppy`` syntax, declarations first."""
    return _PRINTER.program(program)
