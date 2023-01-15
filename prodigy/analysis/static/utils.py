from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from probably.pgcl import (AsgnInstr, Expr, Instr, Unop, UnopExpr, Var, VarExpr, Walk, WhileInstr, mut_expr_children,
                           walk_instrs)
from probably.util.ref import Mut


def _vars_of_expr(expr: Expr) -> List[Var]:
    """Generates a list of variables an expression "reads" from; also known as "free variables" when having binding."""
    if isinstance(expr, UnopExpr) and expr.operator == Unop.IVERSON:
        return []
    if isinstance(expr, VarExpr):
        return [expr.var]
    buffer: List[Var] = []
    for child_ref in mut_expr_children(Mut.alloc(expr)):
        buffer.extend(_vars_of_expr(child_ref.val))
    return buffer


def _written_vars(instrs: AsgnInstr | List[Instr]) -> List[Var]:
    """Generates a list of "written" vars in a list of statements: left hand-side of assignments and guards of loops."""
    buffer: List[Var] = []
    if isinstance(instrs, AsgnInstr):
        buffer.append(instrs.lhs)
    else:
        for instr_ref in walk_instrs(Walk.DOWN, instrs):
            if isinstance(instr_ref.val, AsgnInstr):
                buffer.append(instr_ref.val.lhs)
            elif isinstance(instr_ref.val, WhileInstr):
                # Termination of while loops determines the value of or a relationship between variables of the guard as
                # the guard is false then.
                buffer.extend(_vars_of_expr(instr_ref.val.cond))
    return buffer


def _ancestors_and_descendants_every_node(graph: Iterable[Tuple[Var, Var]],
                                          variables: Iterable[Var]):
    """Finding ancestors and descendants of every node by going through the edge set and by caching."""
    ancestors: Dict[Var, Set[Var]] = {var: {var} for var in variables}
    descendants: Dict[Var, Set[Var]] = {var: {var} for var in variables}

    # no problems with cycles when iterating through edges
    for from_var, to_var in set(graph):
        # Redundant since ancestors and descendants are defined reflective, so will be done later
        #var_info[from_var][1].add(to_var)
        #var_info[to_var][0].add(from_var)

        # going through the already known descendants of to_var
        for to_var_descendant in descendants[to_var]:
            # and giving them from_var as ancestor
            ancestors[to_var_descendant].add(from_var)
            # also update from_var that these are descendants
            descendants[from_var].add(to_var_descendant)

        # going through the already known ancestors of from_var
        for from_var_ancestor in ancestors[from_var]:
            # and giving them to_var as descendant
            descendants[from_var_ancestor].add(to_var)
            # also update to_var that these are ancestors
            ancestors[to_var].add(from_var_ancestor)

    return ancestors, descendants
