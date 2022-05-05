r"""
--------------
Program Shapes
--------------

The ``syntax`` module provides functions to analyze a program's syntax: Whether it
is *linear*, or whether it only consists of one big loop.

.. _linearity:

^^^^^^^^^
Linearity
^^^^^^^^^

*Linear arithmetic expressions* :math:`e \in \mathsf{AE}_\text{lin}` adhere to the grammar

.. math::

    \begin{aligned}
        e \rightarrow &\quad n \in \mathbb{N} \quad && \text{\small{}(constants)} \\
                \mid  &\quad x \in \mathsf{Vars} && \text{\small{}(variables)} \\
                \mid  &\quad e + e && \text{\small{}(addition)} \\
                \mid  &\quad e \dot{-} e && \text{\small{}(monus)} \\
                \mid  &\quad n \cdot e && \text{\small{}(multiplication by constants)}
    \end{aligned}

*Monus* means subtraction truncated at zero [#monus]_. In Prodigy, we just use
:py:data:`prodigy.pgcl.ast.Binop.MINUS` (and do not distinguish between minus
and monus).

*Linear Boolean expressions* :math:`\varphi \in \mathsf{BE}_\text{lin}` adhere to the grammar

.. math::

    \begin{aligned}
        \varphi \rightarrow &\quad e < e \quad && \text{\small{}(strict inequality of arithmetic expressions)} \\
                       \mid &\quad e \leq e && \text{\small{}(inequality of arithmetic expressions)} \\
                       \mid &\quad e = e && \text{\small{}(equality of arithmetic expressions)} \\
                       \mid &\quad \varphi \land \varphi && \text{\small{}(conjunction)} \\
                       \mid &\quad \varphi \lor \varphi && \text{\small{}(disjunction)} \\
                       \mid &\quad \neg \varphi && \text{\small{}(negation)} \\
                       \mid &\quad \mathsf{true} \\
                       \mid &\quad \mathsf{false}
    \end{aligned}

*Linear pGCL programs* consist of statements which only ever use linear
arithmetic and linear Boolean expressions. Probabilistic choices must be done
with a constant probability expression.

The weakest pre-expectations of linear programs (see :py:mod:`prodigy.pgcl.backward`)
are also linear in the sense that they can be easily rewritten as *linear
expectations*. You can use for example :py:func:`prodigy.pgcl.simplify.normalize_expectation`.

The set :math:`\mathsf{Exp}_\text{lin}` of *linear expectations* is given by the grammar

.. math::

    \begin{aligned}
        f \rightarrow &\quad e\quad && \text{\small{}(arithmetic expression)} \\
                 \mid &\quad \infty && \text{\small{}(infinity)} \\
                 \mid &\quad r \cdot f && \text{\small{}(scaling)} \\
                 \mid &\quad [\varphi] \cdot f && \text{\small{}(guarding)} \\
                 \mid &\quad f + f && \text{\small{}(addition)}
    \end{aligned}

where :math:`e \in \mathsf{AE}_\text{lin}` is a linear arithmetic expression,
:math:`r \in \mathbb{Q}_{\geq 0}` is a non-negative rational, and where
:math:`\varphi \in \mathsf{BE}_\text{lin}` is a linear Boolean expression.

.. [#monus] See the `Monus Wikipedia page <https://en.wikipedia.org/wiki/Monus>`_ for the general mathematical term.

.. autofunction:: prodigy.pgcl.analyzer.syntax.check_is_linear_program
.. autofunction:: prodigy.pgcl.analyzer.syntax.check_is_linear_expr

.. _one_big_loop:

^^^^^^^^^^^^
One Big Loop
^^^^^^^^^^^^

Check whether the program consists of one big while loop with optional
assignment statements before the loop.

Every program can be converted into a program with just one big loop and a bunch
of initialization assignments before the loop using
:py:func:`prodigy.pgcl.cfg.program_one_big_loop`.

.. autofunction:: prodigy.pgcl.analyzer.syntax.check_is_one_big_loop

"""

from typing import Optional, Sequence

from prodigy.util.ref import Mut

from prodigy.pgcl.ast import (AsgnInstr, Binop, BinopExpr, Expr, Instr, Program, Unop,
                              UnopExpr, VarExpr, WhileInstr, NatLitExpr)
from prodigy.pgcl.typechecker.check import CheckFail
from prodigy.pgcl.ast.walk import Walk, instr_exprs, mut_expr_children, walk_expr, walk_instrs


def check_is_linear_program(program: Program) -> Optional[CheckFail]:
    """
    Check if the given (well-typed) program is linear.

    .. doctest::

        >>> from .parser import parse_pgcl

        >>> program = parse_pgcl("bool b; while (true) { }")
        >>> check_is_linear_program(program)

        >>> program = parse_pgcl("while (true) { x := 2 * x }")
        >>> check_is_linear_program(program)

        >>> program = parse_pgcl("while (true) { x := x * x }")
        >>> check_is_linear_program(program)
        CheckFail(location=..., message='Is not a linear expression')
    """
    for instr_ref in walk_instrs(Walk.DOWN, program.instructions):
        for expr in instr_exprs(instr_ref.val):
            res = check_is_linear_expr(expr)
            if isinstance(res, CheckFail):
                return res
    return None


def has_variable(expr: Expr) -> bool:
    if isinstance(expr, UnopExpr) and expr.operator == Unop.IVERSON:
        return False
    if isinstance(expr, VarExpr) and not expr.is_parameter:
        return True
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        if has_variable(child_ref.val):
            return True
    return False


def check_is_linear_expr(expr: Expr) -> Optional[CheckFail]:
    """
    Linear expressions do not multiply variables with each other.
    However, they may contain multiplication with constants or Iverson brackets.
    Division is also not allowed in linear expressions.

    .. doctest::

        >>> from .ast import *
        >>> from .parser import parse_expectation
        >>> nat = NatLitExpr(10)
        >>> check_is_linear_expr(BinopExpr(Binop.TIMES, nat, nat))
        >>> check_is_linear_expr(BinopExpr(Binop.TIMES, nat, VarExpr('x')))
        >>> check_is_linear_expr(BinopExpr(Binop.TIMES, VarExpr('x'), VarExpr('x')))
        CheckFail(location=..., message='Is not a linear expression')
        >>> check_is_linear_expr(parse_expectation("[x < 6] * x"))
        >>> check_is_linear_expr(parse_expectation("[x * x]"))
        CheckFail(location=..., message='Is not a linear expression')
        >>> check_is_linear_expr(parse_expectation("x/x"))
        CheckFail(location=..., message='General division is not linear (division of constants is)')
    """

    for node_ref in walk_expr(Walk.DOWN, Mut.alloc(expr)):
        node = node_ref.val
        if isinstance(node, BinopExpr):
            if node.operator == Binop.MODULO or \
                    (node.operator == Binop.TIMES and has_variable(node.lhs) and has_variable(node.rhs)):
                return CheckFail(node, "Is not a linear expression")
            if node.operator == Binop.DIVIDE:
                return CheckFail(
                    node,
                    "General division is not linear (division of constants is)"
                )

    return None


def check_is_modulus_condition(expression) -> bool:
    if isinstance(expression, BinopExpr) \
            and expression.operator == Binop.EQ \
            and isinstance(expression.rhs, NatLitExpr) \
            and isinstance(expression.lhs, BinopExpr) \
            and expression.lhs.operator == Binop.MODULO:
        mod_expr = expression.lhs
        if isinstance(mod_expr.lhs, VarExpr) and isinstance(mod_expr.rhs, NatLitExpr):
            return True
    return False


def check_is_constant_constraint(expression: Expr) -> bool:
    if not isinstance(expression, BinopExpr):
        return False
    if expression.operator not in (Binop.EQ, Binop.LEQ, Binop.LE, Binop.GE, Binop.GEQ):
        return False
    if isinstance(expression.lhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.rhs)):
            if has_variable(sub_expr.val):
                return False
        return True
    elif isinstance(expression.rhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.lhs)):
            if has_variable(sub_expr.val):
                return False
        return True
    else:
        return False


def check_is_one_big_loop(instrs: Sequence[Instr],
                          *,
                          allow_init=True) -> Optional[CheckFail]:
    """
    Check whether this program consists of only one big loop with optional
    assignments before that.

    Args:
        instrs: List of instructions to check
        allow_init: Whether to allow an optional sequence of assignments before the loop.

    .. doctest::

        >>> from .parser import parse_pgcl

        >>> program = parse_pgcl("")
        >>> check_is_one_big_loop(program.instructions)
        CheckFail(location=..., message='Program must contain exactly one loop')

        >>> program = parse_pgcl("while (true) { while (true) { } }")
        >>> check_is_one_big_loop(program.instructions)
        CheckFail(location=..., message='Program must contain exactly one loop')

        >>> program = parse_pgcl("x := 5; while (true) { x := 2 * x }")
        >>> check_is_one_big_loop(program.instructions) is None
        True

        >>> program = parse_pgcl("x := 5; while (true) { x := 2 * x }")
        >>> check_is_one_big_loop(program.instructions, allow_init=False)
        CheckFail(location=..., message='Program must contain exactly one loop')
    """
    instrs = list(instrs)

    while allow_init and len(instrs) > 0 and isinstance(instrs[0], AsgnInstr):
        instrs.pop(0)

    err = "Program must contain exactly one loop"
    if len(instrs) != 1 or not isinstance(instrs[0], WhileInstr):
        return CheckFail(None, err)

    loop = instrs[0]
    for instr in walk_instrs(Walk.DOWN, loop.body):
        if isinstance(instr.val, WhileInstr):
            return CheckFail(instr.val, err)
    return None


def is_loopfree(instrs: Sequence[Instr]) -> bool:
    if not instrs:
        return True
    return all(map(lambda x: not isinstance(x.val, WhileInstr), walk_instrs(Walk.DOWN, list(instrs))))


def check_for_nested_loops(instrs: Sequence[Instr]) -> bool:
    assert isinstance(instrs[0], WhileInstr)
    return len(get_nested_while_loops(instrs)) > 1


def _skip_to_while(instrs: Sequence[Instr]) -> Mut[WhileInstr]:
    assert not is_loopfree(instrs)
    for instr in walk_instrs(Walk.DOWN, list(instrs)):
        if isinstance(instr.val, WhileInstr):
            return instr


def get_nested_while_loops(instrs: Sequence[Instr]) -> [Mut[WhileInstr]]:
    assert not is_loopfree(instrs)
    loops = []
    loop = _skip_to_while(instrs)
    loops.append(loop)
    while not is_loopfree(loop.val.body):
        loop = _skip_to_while(loop.val.body)
        loops.append(loop)
    loops.reverse()
    return loops