from __future__ import annotations

import logging
from typing import Iterable, List, Set, Tuple

from probably.pgcl import (AsgnInstr, BernoulliExpr, Binop, BinopExpr,
                           BoolLitExpr, ChoiceInstr, ExpectationInstr, IfInstr,
                           Instr, LoopInstr, NatLitExpr, NatType, ObserveInstr,
                           Program, SkipInstr, TickInstr, Var, VarExpr,
                           WhileInstr, parse_pgcl)
from probably.pgcl.ast.walk import Walk, walk_instrs

from prodigy.analysis.static.utils import _ancestors_and_descendants_every_node, _vars_of_expr, _written_vars
from prodigy.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


def _statement_handling(statement: Instr,
                        graph: List[Tuple[Var, Var]]) -> List[Tuple[Var, Var]]:
    """Together with _statement_list_handling, inductively looks at program statements for dependencies."""
    if isinstance(statement, SkipInstr):
        return graph
    elif isinstance(statement, TickInstr):
        return graph
    elif isinstance(statement, AsgnInstr):
        expr_vars = _vars_of_expr(statement.rhs)
        local_graph: List[Tuple[Var, Var]] = [(var, statement.lhs)
                                              for var in expr_vars]
        return graph + local_graph
    elif isinstance(statement, IfInstr):
        # note, list of instructions here, different typing for sequencing than in theory
        graph_then = _statement_list_handling(statement.true, [])
        graph_else = _statement_list_handling(statement.false, [])
        # control deps
        cond_vars = _vars_of_expr(statement.cond)
        written_vars_then = _written_vars(statement.true)
        written_vars_else = _written_vars(statement.false)
        control_then: List[Tuple[Var, Var]] = [(x, y) for x in cond_vars
                                               for y in written_vars_then]
        control_else: List[Tuple[Var, Var]] = [(x, y) for x in cond_vars
                                               for y in written_vars_else]
        return graph + graph_then + graph_else + control_then + control_else
    elif isinstance(statement, WhileInstr):
        # list of instructions again
        graph_then = _statement_list_handling(statement.body, graph)
        # control deps
        cond_vars = _vars_of_expr(statement.cond)
        written_vars = _written_vars(statement.body)
        control: List[Tuple[Var, Var]] = [(x, y) for x in cond_vars
                                          for y in written_vars]
        return graph_then + control
    elif isinstance(statement, ChoiceInstr):
        # we would need a connection of the form <--> between both sides, but it should not be simply undirected
        # quite difficult without fresh variable and adding a variable at this point is not propagating
        logger.error("choice not supported here")
    elif isinstance(statement, LoopInstr):  # no theory for it
        logger.error("loop not supported")
    elif isinstance(statement, ObserveInstr):  # not possible for pairwise
        logger.error("observe not supported")
    elif isinstance(statement, ExpectationInstr):  # not possible for pairwise
        logger.error("observe not supported")
    else:
        logger.error("%s not supported atm", type(statement))
    return graph


def _statement_list_handling(
        stat_list: List[Instr],
        graph: List[Tuple[Var, Var]]) -> List[Tuple[Var, Var]]:
    """Together with _statement_handling, inductively analyses program statements for dependencies."""
    for statement in stat_list:
        graph = _statement_handling(statement, graph)
    return graph


def build_dependence_graph(program: Program) -> Set[Tuple[Var, Var]]:
    """Builds the variable dependence graph for a stand-alone program."""
    return set(_statement_list_handling(program.instructions, []))


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
                i += 1
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


def independent_vars(program: Program) -> Set[Set[Var, Var]]:
    """
    This method under-approximates the pairwise stochastic independence relation using the d-separation on a simple
    dependence graph.

    .. param program: The program
    .. returns: Set of variable pairs which are surely independent.
    """

    # preprocessing
    # choice transformation is here to have fresh variable in the set of program.variables
    logger.debug("preprocessing")
    mod_program = _preprocess(program)

    logger.debug("building graph")
    # build edge set
    graph = build_dependence_graph(mod_program)
    logger.debug(" graph: %s", str(graph))

    result = _dsep(graph, mod_program.variables)
    logger.debug(" result: %s", str(result))

    return result


def _dsep(graph: Iterable[Tuple[Var, Var]],
          program_variables: Iterable[Var]) -> Set[Set[Var, Var]]:
    """Does the pairwise d-separation of the graph (directed) by intersecting the ancestors."""
    ancestors, _ = _ancestors_and_descendants_every_node(
        graph, program_variables)
    return {
        frozenset({x, y})
        for x in program_variables for y in program_variables
        if len(ancestors[x].intersection(ancestors[y])) == 0
    }
