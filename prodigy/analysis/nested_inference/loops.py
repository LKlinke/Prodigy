"""
Loops that read ``dist`` variables: split by their values, bind them while the loop runs, so that the loop strategies
never see them. Assigning a ``dist`` variable inside a loop is not implemented.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple, Union

from frozendict import frozendict

from prodigy.analysis.exceptions import UnsupportedFeatureError
from prodigy.analysis.context import Frame, ProgramState, Semantics
from prodigy.distribution import Distribution
from prodigy.language import AsgnInstr, Expr, Instr, LoopInstr, VarExpr, WhileInstr
from prodigy.language.ast.expressions import SampleExpr
from prodigy.language.ast.walk import Walk, walk_expr, walk_instr_exprs, walk_instrs
from prodigy.language.util.ref import Mut
from prodigy.analysis.nested_inference.sites import (_set_site, _sum, dist_variables, encoding_variables,
                                                     split_by_values, without)

def _read_dist_variables(instrs: List[Instr], exprs: List[Expr], dist_vars: Set[str]) -> Set[str]:
    found: Set[str] = set()

    def visit(expr: Expr) -> None:
        for ref in walk_expr(Walk.DOWN, Mut.alloc(expr)):
            node = ref.val
            if isinstance(node, VarExpr) and node.var in dist_vars:
                found.add(node.var)
            elif isinstance(node, SampleExpr) and node.dist in dist_vars:
                found.add(node.dist)

    for expr in exprs:
        visit(expr)
    for ref in walk_instrs(Walk.DOWN, instrs):
        for expr_ref in walk_instr_exprs(Walk.DOWN, ref.val):
            visit(expr_ref.val)
    return found


def assigned_dist_variables(instrs: List[Instr], dist_vars: Set[str]) -> List[str]:
    """The ``dist`` variables that ``instrs`` assign (also in nested blocks)."""
    found = []
    for ref in walk_instrs(Walk.DOWN, instrs):
        if isinstance(ref.val, AsgnInstr) and ref.val.lhs in dist_vars and ref.val.lhs not in found:
            found.append(ref.val.lhs)
    return found


def run_loop(frame: Frame, instr: Union[WhileInstr, LoopInstr], state: ProgramState,
             untraced: Semantics) -> Optional[ProgramState]:
    """Runs a loop with the ``dist`` variables bound to their values (see the module
    documentation); ``None`` if there is nothing to bind. The loop itself runs with ``untraced`` in a frame with
    the bindings (its step is traced already)."""
    prog_info, (distribution, error_prob) = frame.program, state
    reg = frame.context.dist_registry
    dist_vars = dist_variables(prog_info) - set(prog_info.dist_bindings)
    if not dist_vars:
        return None
    assigned = assigned_dist_variables(instr.body, dist_vars | set(prog_info.dist_bindings))
    if assigned:
        raise UnsupportedFeatureError(
            f"The loop assigns the dist variable {assigned[0]}: dist values in the state of a loop "
            "are not implemented (a generating function cannot carry infinitely many different "
            "distributions as states). Assign it before the loop.")
    exprs = [instr.cond] if isinstance(instr, WhileInstr) else []
    read = _read_dist_variables(instr.body, exprs, dist_vars)
    present = sorted(var for var in dist_vars if var in distribution.get_variables())
    if not present:
        return None
    results: List[Tuple[Distribution, Distribution]] = []
    for part, sites in split_by_values(distribution, present, prog_info, reg):
        bindings = dict(prog_info.dist_bindings)
        removed: List[str] = []
        restore: Dict[str, int] = {}
        for var, site in sites.items():
            if site is not None and not site.closed:
                if var in read:
                    raise UnsupportedFeatureError(
                        f"The loop reads the dist variable {var}, whose value depends on the program "
                        f"state with infinite support ({site.describe()}); this is not implemented.")
                continue  # stays in the distribution, the loop does not touch it
            removed.extend(encoding_variables(part, var))
            if site is not None:
                bindings[var] = site
                restore[var] = reg.site_id(site)
        inner = frame.for_program(replace(prog_info, dist_bindings=frozendict(bindings)))
        out, error = untraced(inner, instr, ProgramState(without(part, removed), error_prob * "0"))
        for var, site_id in restore.items():
            out = _set_site(out, var, site_id)
        results.append((out, error))
    return _sum(results, distribution, error_prob)
