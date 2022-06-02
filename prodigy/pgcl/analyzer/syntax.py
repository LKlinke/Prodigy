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

from typing import Sequence

from prodigy.util.ref import Mut

from probably.pgcl import (Binop, BinopExpr, Expr, Instr, Unop,
                           UnopExpr, VarExpr, WhileInstr, NatLitExpr, Program)
from probably.pgcl.ast.walk import Walk, mut_expr_children, walk_expr, walk_instrs


# FIXME: These functions should be implemented in probably.
def has_variable(expr: Expr, prog: Program) -> bool:
    if isinstance(expr, UnopExpr) and expr.operator == Unop.IVERSON:
        return False
    if isinstance(expr, VarExpr) and expr.var not in prog.parameters.keys():
        return True
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        if has_variable(child_ref.val, prog):
            return True
    return False


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


def check_is_constant_constraint(expression: Expr, prog: Program) -> bool:
    if not isinstance(expression, BinopExpr):
        return False
    if expression.operator not in (Binop.EQ, Binop.LEQ, Binop.LE, Binop.GE, Binop.GEQ):
        return False
    if isinstance(expression.lhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.rhs)):
            if has_variable(sub_expr.val, prog):
                return False
        return True
    elif isinstance(expression.rhs, VarExpr):
        for sub_expr in walk_expr(Walk.DOWN, Mut.alloc(expression.lhs)):
            if has_variable(sub_expr.val, prog):
                return False
        return True
    else:
        return False


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