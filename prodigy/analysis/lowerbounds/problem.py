"""
A single-loop program split into the parts used by the lower-bound modes: the loop-free prefix (which
determines the initial measure ``g`` of the loop), the loop, and the loop-free epilogue with its queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from prodigy.pgcl import ExpectationInstr, PlotInstr, PrintInstr, ProbabilityQueryInstr, Program, WhileInstr

from prodigy.analysis.lowerbounds import explicit
from prodigy.analysis.lowerbounds.ir import (IR, UnsupportedProgram, eval_number, expr_to_ir, instr_to_ir,
                                             split_program)


@dataclass
class Query:
    kind: str          # "?Pr" or "?Ex"
    expr: IR
    text: str


@dataclass
class LoopProblem:
    variables: List[str]
    prefix: IR
    guard: IR
    body: IR
    epilogue: IR
    queries: List[Query] = field(default_factory=list)
    forced_rates: Dict[str, Tuple[Fraction, int]] = field(default_factory=dict)   # variable -> (rate, period)
    initial_state: Optional[Tuple[int, ...]] = None
    init_form: Optional[tuple] = None          # (numerator, Denominator) of an explicitly given initial measure

    @property
    def d(self) -> int:
        return len(self.variables)

    def initial_measure(self, cap: int = 400) -> explicit.Measure:
        """The initial measure ``g`` of the loop as an explicit measure (exact; infinite tails are truncated after
        ``cap`` values per variable)."""
        if self.init_form is not None:
            from prodigy.analysis.lowerbounds.egcalc import GFCalculus
            numer, den = self.init_form
            calc = GFCalculus(self.variables, den)
            box = [max(e[i] for e in numer) + (cap if den.r[i] != 0 else 0) for i in range(self.d)] if numer else \
                [0] * self.d
            arr = calc.expand(numer, box)
            import numpy as np
            meas = {tuple(int(v) for v in idx): arr[idx] for idx in np.ndindex(*arr.shape) if arr[idx] != 0}
            return explicit.apply(self.prefix, meas, self.variables)
        start = self.initial_state or tuple([0] * self.d)
        return explicit.apply(self.prefix, {start: Fraction(1)}, self.variables)

    def describe(self) -> str:
        return f"while {self.guard} with {len(self.queries)} queries over {self.variables}"


def _contains(stmt: IR, tag: str) -> bool:
    if not isinstance(stmt, tuple) or not stmt:
        return False
    if stmt[0] == tag:
        return True
    for sub in stmt[1:]:
        if isinstance(sub, tuple) and _contains(sub, tag):
            return True
        if isinstance(sub, list) and any(_contains(s, tag) for s in sub):
            return True
    return False


def _geometric_samples(stmt: IR, acc: Dict[str, Fraction]):
    if not isinstance(stmt, tuple) or not stmt:
        return acc
    if stmt[0] == "assign" and stmt[2][0] == "call" and stmt[2][1] == "geometric":
        q = eval_number(stmt[2][2][0])
        if stmt[1] in acc and acc[stmt[1]] != (1 - q, 1):
            raise UnsupportedProgram(f"variable {stmt[1]} is sampled from geometric distributions with different "
                                     f"parameters")
        acc[stmt[1]] = (1 - q, 1)
    for sub in stmt[1:]:
        if isinstance(sub, tuple):
            _geometric_samples(sub, acc)
        elif isinstance(sub, list):
            for s in sub:
                _geometric_samples(s, acc)
    return acc


def load_problem(program: Program) -> LoopProblem:
    """Split a program with exactly one top-level loop (and no nested loops) into a :class:`LoopProblem`."""
    prefix, loop, rest = split_program(program)
    assert isinstance(loop, WhileInstr)
    variables = [str(v) for v in program.variables.keys()]
    queries, epi = [], []
    for ins in rest:
        if isinstance(ins, ProbabilityQueryInstr):
            queries.append(Query("?Pr", expr_to_ir(ins.expr), str(ins.expr)))
        elif isinstance(ins, ExpectationInstr):
            queries.append(Query("?Ex", expr_to_ir(ins.expr), str(ins.expr)))
        elif isinstance(ins, WhileInstr):
            raise UnsupportedProgram("only programs with a single loop are supported")
        elif isinstance(ins, (PlotInstr, PrintInstr)):
            continue
        else:
            epi.append(ins)
    prefix_ir = instr_to_ir(prefix)
    body_ir = instr_to_ir(loop.body)
    epi_ir = instr_to_ir(epi)
    forced = _geometric_samples(prefix_ir, {})
    if _geometric_samples(body_ir, {}) or _geometric_samples(epi_ir, {}):
        raise UnsupportedProgram("geometric sampling is only supported before the loop")
    if any(_contains(s, "observe") for s in (prefix_ir, body_ir, epi_ir)):
        raise UnsupportedProgram("observe is not supported by the lower-bound modes: conditioning has to be distinguished from non-termination (Klinkenberg et al., OOPSLA 2024), which these modes do not do yet")
    return LoopProblem(variables=variables, prefix=prefix_ir, guard=expr_to_ir(loop.cond), body=body_ir,
                       epilogue=epi_ir, queries=queries, forced_rates=forced)


def problem_from_loop(loop: WhileInstr, variables: List[str], init_expr: str) -> LoopProblem:
    """A :class:`LoopProblem` for a loop whose input distribution is given as a generating function (used by the
    interactive modes of the while handler)."""
    from prodigy.analysis.lowerbounds.subinvariant import to_product_form
    form = to_product_form(init_expr, variables)
    if form is None:
        raise UnsupportedProgram(f"the input distribution {init_expr} is not a rational generating function with "
                                 f"a product denominator")
    numer, den = form
    body_ir = instr_to_ir(loop.body)
    if _geometric_samples(body_ir, {}):
        raise UnsupportedProgram("geometric sampling is only supported before the loop")
    if _contains(body_ir, "observe"):
        raise UnsupportedProgram("observe is not supported by the lower-bound modes: conditioning has to be distinguished from non-termination (Klinkenberg et al., OOPSLA 2024), which these modes do not do yet")
    forced = {v: (r, p) for v, r, p in zip(variables, den.r, den.p) if r != 0}
    return LoopProblem(variables=list(variables), prefix=("seq", []), guard=expr_to_ir(loop.cond), body=body_ir,
                       epilogue=("seq", []), queries=[], forced_rates=forced, init_form=(numer, den))
