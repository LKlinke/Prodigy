"""
---------------------
Variable Substitution
---------------------

This module does variable substitution for expressions, expectations, and programs.
The purpose of this module is elimination of :class:`SubstExpr` by applying the corresponding substitutions.

.. note::

    The program must be well-typed.
    In particular, variables must be defined before they are used and substitutions must be type-safe.

.. warning::

    The input AST **must be a tree** and objects may not occur multiple times. This is a limitation of the current implementation.
    General DAGs are not supported yet.

.. warning::

    **Substitutions reuse AST objects**, and do not copy unnecessarily.
    Thus, AST objects may turn up multiple times in the AST.
    For later mutation of the AST, you may want to pass `deepcopy = True` to prevent spooky action action at a distance.

.. _substitute_symbolic_variables:

.. rubric:: Symbolic variables

Each substitution function takes a parameter `symbolic` parameter that is a set of variables to be taken symbolic.
Symbolic variables are placeholders for arbitrary expressions, and thus it is not known which variables they contain.
Therefore the substitution cannot be done at this time.
Instead, a substitution applied to a symbolic variable will wrap the variable in a :class:`SubstExpr` that contains all substitutions applied to this variable.

.. doctest::

    >>> from prodigy.util.ref import Mut
    >>> from .ast import SubstExpr, NatLitExpr, VarExpr
    >>> from .parser import parse_expr

    Simple substitution without symbolic variables:
    >>> expr_ref = Mut.alloc(SubstExpr({"x": NatLitExpr(2)}, parse_expr("x + y")))
    >>> substitute_expr(expr_ref)
    >>> str(expr_ref.val)
    '2 + y'

    The same substitution with `y` being symbolic:
    >>> expr_ref = Mut.alloc(SubstExpr({"x": NatLitExpr(2)}, parse_expr("x + y")))
    >>> substitute_expr(expr_ref, symbolic=set("y"))
    >>> str(expr_ref.val)
    '2 + ((y)[x/2])'

.. rubric:: Implementation notes

.. seealso::

    `Documentation of internal classes and functions <substitute_internals.html>`_ can be found on another page.

The implementation is much more sophisticated than the simple approach of doubly-recursive bottom-up substition application.
For comparison, the simple approach is available in `tests/test_substitution.py` and used to test this implementation.
In this module, we implement a different approach using generated local variables for each substitution.

First the `binding` phase, top-down (implemented in :func:`_Binder.bind`):

1. For each substitution expression and each substitution `x/y`, generate a new local variable `t`.
2. Run all current substitutions on `y`, generating `y'`.
3. Now remember two substitutions: `x/t` and `t/y'`.
4. Recurse into the substitutions' children.

Then the `resolve` phase (direction is irrelevant, implemented in :func:`_Binder.resolve`):

1. On the generated substitutions for local variables, replace all simple ones of the form `t1/t2`, resolving `t2` recursively. Now the object references to be inserted in the AST are final (not necessarily nested objects).
2. Now we update the objects to be inserted in the AST, i.e. all replacements of the form `t1/r` for general expressions `r` are updated by replacing all locals `t_i` occuring in `r`. All substitutions are now final.
3. Replace all locals in the expression itself.

Except for step 1 of the `resolve` phase, all phases clearly take linear time.
Step 1 of `resolve` is only run for simple chains of substitutions with no nested locals (e.g. generated from ``x[x/y][y/z]``), and the step also amortizes lookups.
So basically we have linear runtime.

This implementation was very loosely inspired by an article [#locallynameless]_ on "locally nameless" substitution based on the awesomely titled paper "I am not a Number — I am a Free Variable" [#haskell2004]_.


.. rubric:: Footnotes

.. [#locallynameless] https://boarders.github.io/posts/locally-nameless/
.. [#haskell2004] "I am not a number: I am a free variable" (2004) by McBride and McKinna.
"""

import copy as copylib
from typing import Dict, FrozenSet, List, Optional, Union

import attr

from prodigy.util.ref import Mut

from .ast import Expr, ExprClass, Program, SubstExpr, Var, VarExpr
from prodigy.pgcl.ast.walk import (Walk, mut_expr_children, mut_instr_exprs, walk_expr,
                                   walk_instrs)


@attr.s(hash=True)
class _BoundExpr(ExprClass):
    index: int = attr.ib()

    def __str__(self) -> str:
        return f'BoundExpr({self.index})'


class _Binder:
    """
    All substitutions (the `values` in the `subst` dict) are assigned a numeric identifier.
    The binder manages those.
    """
    _bound: List[Expr]
    _deepcopy: bool

    def __init__(self, *, deepcopy: bool):
        self._bound = list()
        self._deepcopy = deepcopy

    def bind(self, expr: Expr) -> _BoundExpr:
        """
        Add the given expression to the binder, return a new :class:`_BoundExpr`
        that points to the expression in the binder.
        """
        bound_index = len(self._bound)
        self._bound.append(expr)
        return _BoundExpr(bound_index)

    def resolve(self):
        """
        In the internal list of bound expressions, replace all
        :class:`_BoundExpr` by the the bound expressions.

        Cycles are handled using a two-step procedure (see module docs).
        """
        # first resolve all top-level bindings
        for expr_ref in Mut.list(self._bound):
            while isinstance(expr_ref.val, _BoundExpr):
                expr_ref.val = self.lookup(expr_ref.val)

        # now that all top-level references in self.bound are final,
        # we can iterate over all expressions and insert them one by one.
        # since the references are stable, changes to an inserted expression
        # will occur everywhere else in the AST it appears as well.
        for expr in self._bound:
            expr_ref = Mut.alloc(expr)
            for subexpr_ref in walk_expr(Walk.DOWN, expr_ref):
                if isinstance(subexpr_ref.val, _BoundExpr):
                    subexpr_ref.val = self.lookup(subexpr_ref.val)
                    assert not isinstance(subexpr_ref.val, _BoundExpr)
            assert expr is expr_ref.val, "expression refs must be stable"

    def lookup(self, bound: _BoundExpr) -> Expr:
        """
        Given a :class:`_BoundExpr`, return the associated expression.

        _Deepcopies_ the expression if ``deepcopy`` is set to ``True``.
        """
        res = self._bound[bound.index]
        if self._deepcopy:
            res = copylib.deepcopy(res)
        return res


class _Subst:
    """
    Data structure for current substitutions to be applied.

    * It holds a mapping for the substitutions itself (Var/BoundExpr to BoundExpr),
    * and the set of symbolic variables that cannot be substituted (see section :ref:`substitute_symbolic_variables` for more information).
    """
    _subst: Dict[Union[Var, _BoundExpr], _BoundExpr]
    _symbolic: FrozenSet[Var]

    def __init__(self,
                 *,
                 subst: Optional[Dict[Union[Var, _BoundExpr],
                                      _BoundExpr]] = None,
                 symbolic: FrozenSet[Var]):
        self._subst = dict() if subst is None else subst.copy()
        self._symbolic = symbolic

    def add_bound(self, binder: _Binder, var: Union[Var, _BoundExpr],
                  expr: Expr):
        """
        Bind the expression using ``binder``, then add a new substitution that
        replaces ``var`` by the newly created bound expression.
        """
        bound_expr = binder.bind(expr)
        self._subst[var] = bound_expr

    def apply(self, var_expr: VarExpr) -> Expr:
        """
        Apply the substitutions to the given variable expression. Symbolic
        variables are replaced by new substitution expressions.
        """
        var = var_expr.var
        if var in self._symbolic:
            return self._mk_subst_expr(var_expr)
        elif var in self._subst:
            return self._subst[var]  # type:ignore
        else:
            return var_expr

    def _mk_subst_expr(self, expr: Expr) -> SubstExpr:
        """
        Create a substitution expression around `body`.
        Used for symbolic variables.
        """
        # cheating with types here a bit
        subst_cast: Dict[Var, Expr] = self._subst  # type:ignore
        return SubstExpr(subst_cast, expr)

    def copy(self) -> "_Subst":
        """
        Make a new copy of this substitution dictionary, for use in branching
        expressions.
        """
        return _Subst(subst=self._subst, symbolic=self._symbolic)


def _bind_substs(binder: _Binder, subst: _Subst, expr_ref: Mut[Expr]):
    """
    Walk the `expr_ref` and replace all right-hand sides of each :class:`SubstExpr` by a locally bound :class:`_BoundExpr`.


    This test asserts the order in which substitutions are applied: First inner
    ones, then outer.

    .. doctest::

        >>> binder = _Binder(deepcopy=False)
        >>> subst = _Subst(symbolic=set())
        >>> expr = SubstExpr({'x': VarExpr('x1')}, SubstExpr({'x': VarExpr('x2')}, VarExpr('x')))
        >>> expr_ref = Mut.alloc(expr)
        >>> _bind_substs(binder, subst, expr_ref)
        >>> str(expr_ref.val)
        'BoundExpr(1)'
        >>> binder.lookup(expr_ref.val)
        VarExpr('x2')
    """
    expr = expr_ref.val

    # Apply the substitution on variables, inserting either the replacement
    # or wrapping the variable with a SubstExpr in case of a symbolic variable.
    if isinstance(expr, VarExpr):
        expr_ref.val = subst.apply(expr)
        return

    # Substitutions have their right-hand side replaced by locally-bound variables.
    if isinstance(expr, SubstExpr):
        assert expr.subst is not None, "Substitution expression must not be visited twice"
        for subst_expr_ref in Mut.dict_values(expr.subst):
            # Assert that we have not visited this SubstExpr before (this can happen when objects
            # occur multiple times in the AST).
            assert not isinstance(subst_expr_ref.val, _BoundExpr)

            _bind_substs(binder, subst.copy(), subst_expr_ref)

        subst = subst.copy()  # open a new subscope
        for var, subst_expr in expr.subst.items():
            subst.add_bound(binder, var, subst_expr)

        # as a safety guard, invalidate the consumed SubstExpr
        expr.subst = None  # type: ignore

        # replace this SubstExpr by its body
        expr_ref.val = expr.expr
        _bind_substs(binder, subst, expr_ref)
        return

    # the default case: apply to children
    for child_expr_ref in mut_expr_children(expr_ref):
        _bind_substs(binder, subst, child_expr_ref)


def _resolve_bound_exprs(binder: _Binder, expr_ref: Mut[Expr]):
    for child_expr_ref in walk_expr(Walk.UP, expr_ref):
        if isinstance(child_expr_ref.val, _BoundExpr):
            child_expr_ref.val = binder.lookup(child_expr_ref.val)
        elif isinstance(child_expr_ref.val, SubstExpr):
            for subst_lhs in child_expr_ref.val.subst.keys():
                assert not isinstance(subst_lhs, _BoundExpr)

            for subst_expr_ref in Mut.dict_values(child_expr_ref.val.subst):
                _resolve_bound_exprs(binder, subst_expr_ref)


def substitute_expr(expr_ref: Mut[Expr],
                    *,
                    symbolic: Optional[FrozenSet[Var]] = None,
                    deepcopy: bool = False):
    """
    Apply substitutions in this expression/expectation.
    After execution, the `expr_ref` contains no :class:`SubstExpr` anywhere except directly around `symbolic` variables.

    Args:
        expr_ref: The reference to the expression to be modified.
        symbolic: The set of symbolic variables. See the module description.
        deepcopy: Whether to call `deepcopy` on each replacement, i.e. not reuse any AST objects in substitutions.
    """
    binder = _Binder(deepcopy=deepcopy)
    subst = _Subst(
        symbolic=frozenset(symbolic) if symbolic is not None else frozenset())
    _bind_substs(binder, subst, expr_ref)
    binder.resolve()
    _resolve_bound_exprs(binder, expr_ref)


def substitute_constants_expr(program: Program,
                              expr_ref: Mut[Expr],
                              *,
                              symbolic: Optional[FrozenSet[Var]] = None):
    """
    Substitute constants in this expression/expectation.

    Args:
        program: The program with variable declarations.
        expr_ref: The reference to the expression to be modified.
        symbolic: The set of symbolic variables. See the module description.
    """
    expr_ref.val = SubstExpr(program.constants.copy(), expr_ref.val)
    substitute_expr(expr_ref, symbolic=symbolic, deepcopy=True)


def substitute_constants(program: Program,
                         *,
                         symbolic: Optional[FrozenSet[Var]] = None):
    """
    Substitute constants in this program.

    .. doctest::

        >>> from prodigy.pgcl.parser import parse_pgcl
        >>> program = parse_pgcl("const x := 23; nat y; nat z; y := x; z := y")
        >>> substitute_constants(program)
        >>> program.instructions
        [AsgnInstr(lhs='y', rhs=NatLitExpr(23)), AsgnInstr(lhs='z', rhs=VarExpr('y'))]

    Args:
        program: The program to be modified.
        symbolic: The set of symbolic variables. See the module description.
    """
    for const_ref in Mut.dict_values(program.constants):
        substitute_constants_expr(program, const_ref)

    for instr_ref in walk_instrs(Walk.DOWN, program.instructions):
        for expr_ref in mut_instr_exprs(instr_ref.val):
            substitute_constants_expr(program, expr_ref, symbolic=symbolic)
