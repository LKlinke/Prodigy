# pylint: disable=invalid-name
# Names follow the mathematics (H, Phi, A, b, ...); scipy and FLINT are imported where they are needed.
"""
Translation of pGCL programs (the AST of prodigy.pgcl) into the small intermediate representation used by the
exact generating-function calculus of :mod:`prodigy.analysis.lowerbounds.egcalc`.

Statements::

    ("seq", [stmt, ...]) | ("skip",) | ("abort",) | ("observe", cond)
    ("choice", prob, stmt, stmt) | ("if", cond, stmt, stmt) | ("assign", var, expr)

Expressions::

    ("num", Fraction) | ("var", name) | ("add"|"sub"|"mul"|"div"|"mod", e, e)
    ("call", name, [e, ...]) | ("cmp", op, e, e) | ("and"|"or", e, e) | ("not", e)
    ("true",) | ("false",)

``abort`` is interpreted as divergence, i.e. its forward semantics maps every measure to the zero measure
(the run never terminates), which is the semantics of ``diverge`` in the paper.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple, Union

from prodigy.pgcl import (AbortInstr, AsgnInstr, Binop, BinopExpr, BoolLitExpr, ChoiceInstr, ExpectationInstr,
                           FunctionCallExpr, IfInstr, Instr, NatLitExpr, ObserveInstr, ProbabilityQueryInstr,
                           Program, RealLitExpr, SkipInstr, Unop, UnopExpr, VarExpr, WhileInstr)

IR = tuple


class UnsupportedProgram(Exception):
    """The program (or loop) is outside the fragment supported by the exact template calculus."""


_BINOPS = {
    Binop.PLUS: "add", Binop.MINUS: "sub", Binop.TIMES: "mul", Binop.DIVIDE: "div", Binop.MODULO: "mod",
    Binop.AND: "and", Binop.OR: "or",
}
_CMPS = {Binop.LEQ: "<=", Binop.LT: "<", Binop.GT: ">", Binop.GEQ: ">=", Binop.EQ: "="}


def expr_to_ir(expr) -> IR:
    """Translate a pGCL expression."""
    if isinstance(expr, NatLitExpr):
        return "num", Fraction(int(expr.value))
    if isinstance(expr, RealLitExpr):
        return "num", Fraction(str(expr.value))
    if isinstance(expr, BoolLitExpr):
        return ("true",) if expr.value else ("false",)
    if isinstance(expr, VarExpr):
        return "var", str(expr.var)
    if isinstance(expr, UnopExpr):
        if expr.operator == Unop.NEG:
            return "not", expr_to_ir(expr.expr)
        raise UnsupportedProgram(f"unsupported unary operator in {expr}")
    if isinstance(expr, BinopExpr):
        if expr.operator in _CMPS:
            return "cmp", _CMPS[expr.operator], expr_to_ir(expr.lhs), expr_to_ir(expr.rhs)
        if expr.operator in _BINOPS:
            return _BINOPS[expr.operator], expr_to_ir(expr.lhs), expr_to_ir(expr.rhs)
        raise UnsupportedProgram(f"unsupported binary operator in {expr}")
    if isinstance(expr, FunctionCallExpr):
        return "call", str(expr.function), [expr_to_ir(e) for e in expr.params[0]]
    raise UnsupportedProgram(f"unsupported expression {expr} ({type(expr).__name__})")


def instr_to_ir(instr: Union[Instr, Sequence[Instr]]) -> IR:
    """Translate a pGCL instruction or a list of instructions."""
    if isinstance(instr, list):
        return "seq", [instr_to_ir(i) for i in instr]
    if isinstance(instr, SkipInstr):
        return ("skip",)
    if isinstance(instr, AbortInstr):
        return ("abort",)
    if isinstance(instr, ObserveInstr):
        return "observe", expr_to_ir(instr.cond)
    if isinstance(instr, ChoiceInstr):
        return "choice", expr_to_ir(instr.prob), instr_to_ir(instr.lhs), instr_to_ir(instr.rhs)
    if isinstance(instr, IfInstr):
        return "if", expr_to_ir(instr.cond), instr_to_ir(instr.true), instr_to_ir(instr.false)
    if isinstance(instr, AsgnInstr):
        return "assign", str(instr.lhs), expr_to_ir(instr.rhs)
    if isinstance(instr, WhileInstr):
        raise UnsupportedProgram("nested loops are not supported by the exact template calculus")
    raise UnsupportedProgram(f"unsupported instruction {instr} ({type(instr).__name__})")


def eval_number(e: IR) -> Fraction:
    """Evaluate a closed arithmetic expression (e.g. a probability such as 1/3) to a fraction."""
    t = e[0]
    if t == "num":
        return Fraction(e[1])
    if t in ("add", "sub", "mul", "div"):
        a, b = eval_number(e[1]), eval_number(e[2])
        return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[t]
    raise UnsupportedProgram(f"not a constant: {e}")


def split_program(program: Program) -> Tuple[List[Instr], WhileInstr, List[Instr]]:
    """Split a program into the loop-free prefix, the (single, top-level) loop, and the instructions after it."""
    loops = [i for i, ins in enumerate(program.instructions) if isinstance(ins, WhileInstr)]
    if len(loops) != 1:
        raise UnsupportedProgram(f"expected exactly one top-level loop, found {len(loops)}")
    k = loops[0]
    return list(program.instructions[:k]), program.instructions[k], list(program.instructions[k + 1:])


def queries_of(instructions: Sequence[Instr]) -> List[Tuple[str, IR, str]]:
    """The ``?Pr[...]`` and ``?Ex[...]`` queries among the given instructions as (kind, expression, text)."""
    out = []
    for ins in instructions:
        if isinstance(ins, ProbabilityQueryInstr):
            out.append(("?Pr", expr_to_ir(ins.expr), str(ins.expr)))
        elif isinstance(ins, ExpectationInstr):
            out.append(("?Ex", expr_to_ir(ins.expr), str(ins.expr)))
    return out


def eval_expr(e: IR, state: dict):
    """Evaluate an IR expression in a concrete state (used by the explicit-state reference semantics)."""
    t = e[0]
    if t == "num":
        return e[1]
    if t == "var":
        return state[e[1]]
    if t == "true":
        return True
    if t == "false":
        return False
    if t == "not":
        return not eval_expr(e[1], state)
    if t == "and":
        return eval_expr(e[1], state) and eval_expr(e[2], state)
    if t == "or":
        return eval_expr(e[1], state) or eval_expr(e[2], state)
    if t == "cmp":
        a, b = eval_expr(e[2], state), eval_expr(e[3], state)
        return {"<=": a <= b, "<": a < b, ">": a > b, ">=": a >= b, "=": a == b}[e[1]]
    if t in ("add", "sub", "mul", "div", "mod"):
        a, b = eval_expr(e[1], state), eval_expr(e[2], state)
        if t == "add":
            return a + b
        if t == "sub":
            return max(a - b, 0)          # truncated subtraction on the naturals
        if t == "mul":
            return a * b
        if t == "div":
            return Fraction(a) / b
        return a % b
    raise UnsupportedProgram(f"cannot evaluate {e}")
