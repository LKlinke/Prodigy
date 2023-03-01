from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx  # type: ignore
from probably.pgcl import (AsgnInstr, Expr, Instr, Unop, UnopExpr, Var,
                           VarExpr, Walk, WhileInstr, mut_expr_children,
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


def _ancestors_every_node(graph: Iterable[Tuple[Var, Var]],
                          variables: Iterable[Var]):
    """Finds ancestors of every node."""
    ancestors: Dict[Var, Set[Var]] = {var: {var} for var in variables}

    nx_g = nx.DiGraph(graph)
    for var in variables:
        ancestors[var].update(
            x[0] for x in nx.edge_bfs(nx_g, var, orientation='reverse'))

    return ancestors
