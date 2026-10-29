from copy import deepcopy
from typing import Dict, List, Tuple, Union

import hypothesis.strategies as st
from hypothesis import given

from prodigy.pgcl.ast import Expr, SubstExpr, Unop, UnopExpr, VarExpr
from prodigy.pgcl.ast.walk import Walk, walk_expr
from prodigy.pgcl.parser import parse_pgcl
from prodigy.pgcl.substitute import substitute_expr
from prodigy.pgcl.util.ref import Mut


def make_closure(substs: List[Tuple[int, int]], var: int,
                 wrap: bool) -> Union[VarExpr, SubstExpr]:
    """
    From a specification, generate a closure expression.

    .. doctest::

        >>> str(make_closure([(1,2), (2,3)], 3, False))
        '((3)[1/2])[2/3]'

        >>> str(make_closure([(1,2), (2,3)], 3, True))
        '((3)[1/not 2])[2/not 3]'
    """
    if len(substs) == 0:
        return VarExpr(str(var))
    else:
        cur_subst = substs.pop()
        rhs: Expr = VarExpr(str(cur_subst[1]))
        if wrap:
            rhs = UnopExpr(Unop.NEG, rhs)
        subst: Dict[str, Expr] = {str(cur_subst[0]): rhs}
        return SubstExpr(subst, make_closure(substs, var, wrap))


@st.composite
def st_closure(draw) -> Union[VarExpr, SubstExpr]:
    wrap = draw(st.booleans())
    size = draw(st.integers(min_value=0, max_value=5))
    substs = [(draw(st.integers(min_value=0, max_value=size)),
               draw(st.integers(min_value=0, max_value=size)))
              for i in range(size)]
    var = draw(st.integers(min_value=0, max_value=size))
    return make_closure(substs, var, wrap)


def dumb_substitute(expr_ref: Mut[Expr]):
    """
    Do variable substitution the dumb way with quadratic runtime.

    .. doctest::

        >>> expr = make_closure([(1,2), (2,3)], 1, True)
        >>> str(expr)
        '((1)[1/not 2])[2/not 3]'
        >>> expr_ref = Mut.alloc(expr)
        >>> dumb_substitute(expr_ref)
        >>> str(expr_ref.val)
        'not not 3'
    """
    for ref in walk_expr(Walk.UP, expr_ref):
        if isinstance(ref.val, SubstExpr):
            subst = ref.val.subst
            ref.val = ref.val.expr
            for inner_ref in walk_expr(Walk.UP, ref):
                if isinstance(inner_ref.val, VarExpr):
                    inner_ref.val = subst.get(inner_ref.val.var, inner_ref.val)


@given(st_closure())
def test_generated_substitution(expr: Expr):
    expr_copy = deepcopy(expr)

    smart_res = Mut.alloc(expr)
    substitute_expr(smart_res)

    dumb_res = Mut.alloc(expr_copy)
    dumb_substitute(dumb_res)

    assert smart_res.val == dumb_res.val
