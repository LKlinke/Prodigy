from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Set, Tuple

from probably.pgcl import (AsgnInstr, BernoulliExpr, Binop, BinopExpr,
                           BoolLitExpr, ChoiceInstr, ExpectationInstr, IfInstr,
                           Instr, LoopInstr, NatLitExpr, NatType, ObserveInstr,
                           Program, SkipInstr, TickInstr, Var, VarExpr,
                           WhileInstr, parse_pgcl)
from probably.pgcl.ast.walk import Walk, walk_instrs

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)

def _preprocess(program: Program) -> Program:
    """Preprocesses the program code to be compatible with further methods."""
    # deep copy program, takes some milliseconds, sorry for the runtime and this dumb solution
    original_program_code = str(program)
    new_program = parse_pgcl(original_program_code)

    # transform choice to sampling and if-then-else
    # Due to list of lines instead of fully inductive structure, tedious to replace one line with two lines in-place
    # As we did not want to implement our own walk, we add a dummy if-then-else
    for instr_ref in walk_instrs(Walk.DOWN, new_program.instructions):
        if isinstance(instr_ref.val, ChoiceInstr):
            # generate fresh var name
            i = 0
            while 'choicetmp' + str(i) in new_program.variables.keys():
                i+=1
            fresh_var_name = 'choicetmp' + str(i)

            # add it
            new_program.add_variable(fresh_var_name, NatType(None))

            sampling = AsgnInstr(lhs=fresh_var_name,
                                 rhs=BernoulliExpr(param=instr_ref.val.prob))

            cond = BinopExpr(lhs=VarExpr(fresh_var_name),
                             operator=Binop.EQ,
                             rhs=NatLitExpr(0))
            inner_branching = IfInstr(cond=cond,
                                      true=instr_ref.val.lhs,
                                      false=instr_ref.val.rhs)

            # We need to pack two instructions in one statement to replace it
            # So, we put it in another if-then-else with guard always true
            dummy_if = IfInstr(cond=BoolLitExpr(True),
                               true=[sampling, inner_branching],
                               false=[SkipInstr()])

            instr_ref.val = dummy_if

    return new_program


def independent_vars(program: Program,
                     config: ForwardAnalysisConfig) -> Set[Set[Var, Var]]:
    """
    This method under-approximates the pairwise stochastic independence relation using the d-separation on a simple
    dependence graph.

    .. param config: Some configuration.
    .. param program: The program
    .. returns: Set of variable pairs which are surely independent.
    """

    # preprocessing
    # choice transformation is here to have fresh variable in the set of program.variables
    mod_program = _preprocess(program)

    logger.debug("start.")

    logger.debug(" result:\t%s", "")

    return {set()}
