"""
------------------
Control-Flow Graph
------------------

This module can build up *control-flow graphs* (CFGs) of pGCL programs and
*decompile* them back to programs again. The main purpose of this module is the
transformation of a program with arbitrarily nested loops into a program with
one big loop (:func:`program_one_big_loop`).


.. doctest::

    >>> from prodigy.pgcl.compiler import compile_pgcl
    >>> program = compile_pgcl("nat x; while (x <= 3) { while (x = 0) { x := x + 1; }; if (x = 0) { x := x + 1; x := x + 2; x := x + 3; } {} }")
    >>> graph = ControlFlowGraph.from_instructions(program.instructions)
    >>> graph.render_dot().source
    'digraph...'

You can see the control-flow graph for this program visualized using GraphViz
below. It was created with the :meth:`ControlFlowGraph.render_dot` method. See
its documentation for more information on how to visualize a control-flow graph.

.. exec::

    from prodigy.pgcl.compiler import compile_pgcl
    from prodigy.pgcl.cfg import ControlFlowGraph, _write_docs_graphviz
    program = compile_pgcl("nat x; while (x <= 3) { while (x = 0) { x := x + 1; }; if (x = 0) { x := x + 1; x := x + 2; x := x + 3; } {} }")
    dot = ControlFlowGraph.from_instructions(program.instructions).render_dot()
    _write_docs_graphviz(dot, "cfg")


.. image:: _generated/cfg.svg
    :align: center
"""

from enum import Enum, auto
from functools import reduce
from typing import Any, Dict, List, NewType, Optional, Set, Tuple, cast

import attr
import graphviz as gv
import networkx as nx
from networkx.algorithms.dominance import dominance_frontiers

from prodigy.util import dedup_list

from .ast import (AsgnInstr, Binop, BinopExpr, BoolLitExpr, ChoiceInstr, Expr,
                  IfInstr, Instr, NatLitExpr, NatType, Program, SkipInstr, Var,
                  VarExpr, WhileInstr)
from .compiler import parse_pgcl
from prodigy.analysis.backward.simplify import simplifying_neg

BasicBlockId = NewType('BasicBlockId', int)


def _write_docs_graphviz(dot: gv.Digraph, name: str):
    dot.format = "svg"
    dot.attr("graph", bgcolor="transparent", rankdir="LR")
    dot.render(f"source/_generated/{name}")


class TerminatorKind(Enum):
    """
    Whether the :class:`Terminator` branches on a Boolean condition or throws a
    die with a probability given by an expression.
    """
    BOOLEAN = auto()
    PROBABILISTIC = auto()


@attr.s
class Terminator:
    """
    A :class:`BasicBlock` is terminated by a *terminator*, which says where to
    go next after executing all assignment statements.

    If a successor block id is `None`, it indicates program termination.
    """

    LEAF: 'Terminator'

    @staticmethod
    def goto(ident: Optional[BasicBlockId]) -> 'Terminator':
        """
        Always go to the given block.
        """
        return Terminator(TerminatorKind.BOOLEAN, BoolLitExpr(True), ident,
                          ident)

    @staticmethod
    def branch(cond: Expr, if_true: Optional[BasicBlockId],
               if_false: Optional[BasicBlockId]) -> 'Terminator':
        """
        Construct a Boolean branching terminator.
        """
        return Terminator(TerminatorKind.BOOLEAN, cond, if_true, if_false)

    @staticmethod
    def choice(expr: Expr, lhs: Optional[BasicBlockId],
               rhs: Optional[BasicBlockId]) -> 'Terminator':
        """
        Construct a probabilistic choice terminator.
        """
        return Terminator(TerminatorKind.PROBABILISTIC, expr, lhs, rhs)

    kind: TerminatorKind = attr.ib()
    condition: Expr = attr.ib()
    if_true: Optional[BasicBlockId] = attr.ib()
    if_false: Optional[BasicBlockId] = attr.ib()

    @property
    def successors(self) -> List[Optional[BasicBlockId]]:
        """
        Return a list of successors of terminator.
        """
        return list(set([self.if_true, self.if_false]))

    def is_goto(self) -> bool:
        """
        Whether the condition is constant `True` or both successors are the
        same.
        """
        return self.condition == BoolLitExpr(
            True) or self.if_true == self.if_false

    def substitute(self, subst: Dict[Optional[BasicBlockId],
                                     Optional[BasicBlockId]]):
        """
        Apply the substitution to this terminator's successors.
        """
        assert self is not Terminator.LEAF
        if self.if_true in subst:
            self.if_true = subst[self.if_true]
        if self.if_false in subst:
            self.if_false = subst[self.if_false]

    def flip(self):
        """
        Flip the condition (by negation using
        :func:`prodigy.pgcl.simplify.simplifying_neg`) and switch both branches
        of this terminator. The operation does not change the semantics of the
        terminator.
        """
        assert self is not Terminator.LEAF
        self.condition = simplifying_neg(self.condition)
        self.if_true, self.if_false = self.if_false, self.if_true

    def __str__(self) -> str:
        if self.kind == TerminatorKind.BOOLEAN:
            if self.if_true == self.if_false:
                return f"goto {self.if_true}"
            else:
                return f"br ({self.condition}) {self.if_true} {self.if_false}"
        else:
            return f"choice ({self.condition}) {self.if_true} {self.if_false}"


Terminator.LEAF = Terminator(TerminatorKind.BOOLEAN, BoolLitExpr(True), None,
                             None)


class BlockType(Enum):
    """
    A :class:`BasicBlock` has an attached marker type that is used for
    back-conversion from a control-flow graph to a structured program, for
    pretty-printing the graph, and for debugging.

    All successor branches of `forward` blocks eventually converge on the same
    block. These are created from any statement except loops. `Loop head` nodes
    are the basic blocks created for the loop condition. These are the first
    entry into a loop from the program start. `Trampoline` blocks are created by
    the :py:func:`one_big_loop` transformation only.
    """
    FORWARD = auto()
    LOOP_HEAD = auto()
    TRAMPOLINE = auto()


@attr.s
class BasicBlock:
    """
    A *basic block* in the control-flow graph consists of a series of
    (unconditional) assignments that are executed in sequence, followed by a
    :class:`Terminator` that says where to go next (based on some condition).
    """
    typ: BlockType = attr.ib()
    ident: BasicBlockId = attr.ib()
    assignments: List[Tuple[Var, Expr]] = attr.ib()
    terminator: Terminator = attr.ib()

    def is_trap(self) -> bool:
        return len(self.assignments) == 0 and self.terminator.is_goto(
        ) and self.terminator.if_true == self.ident

    def __str__(self) -> str:
        return "\n".join([f"{var} := {val}" for var, val in self.assignments] +
                         [str(self.terminator)])


@attr.s
class ControlFlowGraph:
    """
    The control-flow graph consists of an indexed list of :class:`BasicBlock`.

    The :meth:`render_dot` and :meth:`to_networkx` methods can be very useful
    for debugging and visualization of control-flow graphs.
    """
    @staticmethod
    def from_instructions(instructions: List[Instr]) -> 'ControlFlowGraph':
        """
        Convert the given list of instructions to the corresponding control-flow
        graph. The successor of the final generated basic blocks is set to ``None``.

        If the list of instructions is empty, a graph with a single basic block
        will be returned.

        .. doctest::

            >>> graph = ControlFlowGraph.from_instructions([])
            >>> len(graph)
            1
            >>> list(graph)
            [BasicBlock(typ=..., ident=0, assignments=[], terminator=Terminator(kind=..., condition=BoolLitExpr(True), if_true=None, if_false=None))]
        """
        graph = ControlFlowGraph(BasicBlockId(0), list())
        if len(instructions) == 0:
            block = graph.fresh_block()
            start_id = cast(Optional[BasicBlockId], block.ident)
            assert start_id == BasicBlockId(0)
        else:
            start_id = _cfg_add_instructions(graph, instructions, None)
            assert start_id is not None
        graph.entry_id = start_id
        return graph

    entry_id: BasicBlockId = attr.ib()
    _basic_blocks: List[BasicBlock] = attr.ib()

    def fresh_block(self) -> BasicBlock:
        """
        Add a new, empty basic block to the graph of type `forward`.
        """
        ident = BasicBlockId(len(self._basic_blocks))
        block = BasicBlock(BlockType.FORWARD, ident, [], Terminator.LEAF)
        self._basic_blocks.append(block)
        return block

    def __iter__(self):
        return iter(self._basic_blocks)

    def __getitem__(self, key: BasicBlockId) -> BasicBlock:
        return self._basic_blocks[int(key)]

    def __len__(self) -> int:
        return len(self._basic_blocks)

    def substitute(self,
                   subst: Dict[Optional[BasicBlockId], Optional[BasicBlockId]],
                   *,
                   blocks: Optional[Set[BasicBlockId]] = None):
        """
        Apply the substitutions to all blocks in `blocks` (or all blocks if
        `blocks` is `None`).

        Throws an exception if the entry point is replaced with `None`.
        """
        if blocks is None:
            blocks = set((block.ident for block in self._basic_blocks))
        for block_id in blocks:
            self[block_id].terminator.substitute(subst)
        if self.entry_id in blocks and self.entry_id in subst:
            new_entry_id = subst[self.entry_id]
            assert new_entry_id is not None
            self.entry_id = new_entry_id

    def to_networkx(self) -> nx.DiGraph:
        """
        Create a NetworkX :py:class:`networkx.DiGraph` from this control-flow
        graph for further analysis.
        """
        g = nx.DiGraph()
        for block in self:
            g.add_node(block.ident)
        for block in self:
            for succ_id in block.terminator.successors:
                g.add_edge(block.ident, succ_id)
        g.add_edge("start", self.entry_id)
        return g

    def render_dot(self) -> gv.Digraph:
        """
        Render this control-flow graph as a GraphViz
        :py:class:`graphviz.Digraph`.

        The result can be written into a file using
        :py:meth:`graphviz.Digraph.render`:

        .. code::

            graph.render_dot().render(filename='graph.svg')

        The :py:meth:`graphviz.Digraph.view` method can be used to quickly view
        a graph:

        .. code::

            graph.render_dot().view(filename='graph.svg')
        """
        dot = gv.Digraph()
        dot.attr("graph", font="monospace")

        dot.node("start", shape="none", height="0", width="0")
        dot.edge("start", str(self.entry_id))
        dot.node("end", shape="none", height="0", width="0")

        for block in self._basic_blocks:
            styles = {
                BlockType.FORWARD: dict(),
                BlockType.LOOP_HEAD: {
                    "style": "rounded"
                },
                BlockType.TRAMPOLINE: {
                    "color": "#e9e9e9",
                    "style": "filled"
                }
            }[block.typ]
            dot.node(str(block.ident), str(block), shape="box", **styles)
            term = block.terminator
            successors = term.successors
            for succ in successors:
                succ_ident = str(succ) if succ is not None else "end"
                edge_color = None
                if len(successors) == 2:
                    edge_color = "#35C9A2" if succ == term.if_true else "#cf0000"
                dot.edge(str(block.ident), succ_ident, color=edge_color)
        return dot


def _cfg_add_instruction(graph: ControlFlowGraph, instruction: Instr,
                         next_block: Optional[BasicBlockId]) -> BasicBlockId:
    block = graph.fresh_block()

    if isinstance(instruction, SkipInstr):
        block.terminator = Terminator.goto(next_block)
    elif isinstance(instruction, WhileInstr):
        block.typ = BlockType.LOOP_HEAD
        body_block_id = _cfg_add_instructions(graph, instruction.body,
                                              block.ident)
        block.terminator = Terminator.branch(instruction.cond, body_block_id,
                                             next_block)
    elif isinstance(instruction, IfInstr):
        true_block_id = _cfg_add_instructions(graph, instruction.true,
                                              next_block)
        false_block_id = _cfg_add_instructions(graph, instruction.false,
                                               next_block)
        block.terminator = Terminator.branch(instruction.cond, true_block_id,
                                             false_block_id)
    elif isinstance(instruction, AsgnInstr):
        block.assignments = [(instruction.lhs, instruction.rhs)]
        block.terminator = Terminator.goto(next_block)
    elif isinstance(instruction, ChoiceInstr):
        lhs_block_id = _cfg_add_instructions(graph, instruction.lhs,
                                             next_block)
        rhs_block_id = _cfg_add_instructions(graph, instruction.rhs,
                                             next_block)
        block.terminator = Terminator.choice(instruction.prob, lhs_block_id,
                                             rhs_block_id)

    return block.ident


def _cfg_add_instructions(
        graph: ControlFlowGraph, instructions: List[Instr],
        next_block: Optional[BasicBlockId]) -> Optional[BasicBlockId]:
    """
    Add a list of instructiosn to the graph and return the ID of the next block.
    """
    return reduce(
        lambda next_id, next_instr: _cfg_add_instruction(
            graph, next_instr, next_id), reversed(instructions), next_block)


def _write_docs_jumptable_images():
    graph = ControlFlowGraph(0, [])
    target1 = graph.fresh_block()
    target1.assignments = [("x", NatLitExpr("42"))]
    target2 = graph.fresh_block()
    target2.assignments = [("y", NatLitExpr("11"))]
    target1.terminator = Terminator.goto(target2.ident)

    _write_docs_graphviz(graph.render_dot(), "cfg-jumptable1")

    jump_table = JumpTable(graph, "pc")
    graph.entry_id = jump_table.trampoline(target1.ident)

    target2_trampoline = jump_table.trampoline(target2.ident)
    target1.terminator = Terminator.goto(target2_trampoline)

    jump_table.finalize()
    _write_docs_graphviz(graph.render_dot(), "cfg-jumptable2")


@attr.s(init=False)
class JumpTable:
    """
    To implement :func:`one_big_loop`, we cannot allow non-local gotos. Instead,
    we build a jump table that branches on a variable.

    .. exec::

        from prodigy.pgcl.cfg import _write_docs_jumptable_images
        _write_docs_jumptable_images()

    Let's look at a simple example. We start with two blocks: The ``x := 42``
    block (call it `B1`), and the ``y := 11`` block (call it `B2`).

    .. image:: _generated/cfg-jumptable1.svg
        :align: center
        :width: 330px

    We'll connect `B1` and `B2` with the jump table so that `B2` executes after
    `B1`. We add one trampoline for `B1` and let `start` enter at that
    trampoline. And then let `B1` continue to a new trampoline to `B2`. Note
    that the generated jump table is linear in the number of branches. See
    :meth:`finalize` as to why.

    .. image:: _generated/cfg-jumptable2.svg
        :align: center
        :width: 600px

    The ``goto 3`` block is unreachable and we write a self-loop basic block
    that never exits to indicate that.
    """

    _graph: ControlFlowGraph = attr.ib()
    _pc_var: Var = attr.ib()
    entry_id: BasicBlockId = attr.ib()
    error_id: BasicBlockId = attr.ib()
    _jumps: List[Optional[BasicBlockId]] = attr.ib()

    def __init__(self, graph: ControlFlowGraph, pc_var: Var):
        self._graph = graph
        self._pc_var = pc_var
        jump_table_entry = graph.fresh_block()
        jump_table_entry.typ = BlockType.TRAMPOLINE
        self.entry_id = jump_table_entry.ident
        error = graph.fresh_block()
        error.typ = BlockType.TRAMPOLINE
        error.terminator = Terminator.goto(error.ident)
        self.error_id = error.ident
        self._jumps = list()

    def trampoline(self, target: Optional[BasicBlockId]) -> BasicBlockId:
        """
        Return the block id for a new trampoline that sets the program counter
        appropiately and then jumps to the jump table entry basic block.
        """
        block = self._graph.fresh_block()
        block.typ = BlockType.TRAMPOLINE
        target_pc_id = len(self._jumps)
        self._jumps.append(target)
        block.assignments = [(self._pc_var, NatLitExpr(target_pc_id))]
        block.terminator = Terminator.goto(self.entry_id)
        return block.ident

    def finalize(self):
        r"""
        Insert all jumps into the graph.

        The generated jump table intentially only uses strict equality checks
        against `pc`. While it is easily possible to generate a binary tree of
        comparisons using :math:`\leq` (therefore logarithmic size in the number
        of jumps), we decide against it. Inequalities make it harder to see
        which conditions exclude each other and this makes debugging e.g.
        weakest preexpectation generation (see :mod:`prodigy.pgcl.backward`) a bit
        harder.
        """
        pc_expr = VarExpr(self._pc_var)
        cond_block = self._graph[self.entry_id]
        for pc, jump_target in enumerate(self._jumps):
            eq_pc = BinopExpr(Binop.EQ, pc_expr, NatLitExpr(pc))
            if pc < len(self._jumps) - 1:
                next_cond_block = self._graph.fresh_block()
                next_cond_block.typ = BlockType.TRAMPOLINE
                next_cond_block_id = next_cond_block.ident
            else:
                next_cond_block_id = self.error_id
            cond_block.terminator = Terminator.branch(eq_pc, jump_target,
                                                      next_cond_block_id)
            cond_block = next_cond_block


def _write_docs_one_big_loop():
    program = parse_pgcl("""
        bool x;
        while (x) {
            while (x) {
                { } [0.5] { x:= false }
            }
        }
    """)
    graph = ControlFlowGraph.from_instructions(program.instructions)
    _write_docs_graphviz(graph.render_dot(), "cfg-one-big-loop1")
    one_big_loop(graph, "pc")
    _write_docs_graphviz(graph.render_dot(), "cfg-one-big-loop2")


def one_big_loop(graph: ControlFlowGraph, pc_var: Var):
    """
    Given a control-flow graph from a pGCL program with arbitrarily nested
    loops, modify the graph in such a way that there is only one big loop that
    branches to states of the loop. Use ``pc_var`` as the variable name for the
    program counter variable `pc` in the generated jump table (it must not be
    used elsewhere, but should be declared in the program already). For a CFG
    from a well-structured pGCL program, the result is basically one big loop.

    It is assumed all loop heads (see :class:`BlockType`) are properly
    annotated.

    How does it work? This function basically just replaces each jump to a loop
    header, and the program entry and exit with a trampoline
    (:func:`JumpTable.trampoline`).

    .. exec::

        from prodigy.pgcl.cfg import _write_docs_one_big_loop
        _write_docs_one_big_loop()

    Let's work through a (contrived) example. The program below uses two nested
    (but redundant) loops.

    .. code::

        bool x;
        while (x) {
            while (x) {
                { } [0.5] { x:= false }
            }
        }


    The resulting CFG is (from :meth:`ControlFlowGraph.from_instructions`):

    .. image:: _generated/cfg-one-big-loop1.svg
        :align: center
        :width: 500px

    After :meth:`one_big_loop`, we get a much more complicated diagram (see
    below). Now all loops go through the jump table basic blocks that branch on
    `pc`.

    .. image:: _generated/cfg-one-big-loop2.svg
        :align: center
        :width: 100%

    A subtlety can be found in the image above:  The first branch of the
    generated jump table points to the program end and its terminator has been
    flipped (:meth:`Terminator.flip()`) so that the `true` branch goes into the
    loop and  and the `false` branch exits it. We do this to exactly reproduce
    the shape of a generated `while` loop. Also note that the ``goto 5`` is
    unreachable. See :class:`JumpTable` documentation for more information.

    Unfortunately we cannot directly translate this control-flow graph back into
    a well-structured pGCL program. By "direct translation" I mean translation
    to a pGCL program while using each statement exactly once. Consider for
    example the ``pc := 1`` basic block. It is obviously needed before the big
    loop to enter the proper state, but it is also needed after the inner while
    loop exits to return back to the outer loop (``br (x) 2 7``).
    :func:`reduce_cfg_to_instrs` has extra functionality to handle this case and
    will appropriately duplicate some trampoline blocks.
    """
    blocks_via_jump_table = dedup_list(
        [None, graph.entry_id] +
        [block.ident for block in graph if block.typ == BlockType.LOOP_HEAD])
    all_blocks = set((block.ident for block in graph))

    # Initialize the jump table after collecting the blocks, because the
    # initialization already creates two blocks for the jump table that we do not
    # want to modify.
    jump_table = JumpTable(graph, pc_var)

    graph.substitute(
        {
            block_id: jump_table.trampoline(block_id)
            for block_id in blocks_via_jump_table
        },
        blocks=all_blocks)
    jump_table.finalize()

    # Change block types accordingly
    for block in graph:
        if block.typ == BlockType.LOOP_HEAD:
            block.typ = BlockType.FORWARD

    # Flip the entry basic block of the jump table so that we have a proper loop
    # where the `false` branch exits the jump table immediately.
    jump_table_entry = graph[jump_table.entry_id]
    jump_table_entry.typ = BlockType.LOOP_HEAD
    jump_table_entry.terminator.flip()


_DominanceFrontiers = Dict[BasicBlockId, List[BasicBlockId]]


def reduce_cfg_to_instrs(graph: ControlFlowGraph) -> List[Instr]:
    r"""
    Reduce a control-flow graph to a program (i.e. list of instructions). The
    given graph must have the same shape as one generated by
    :meth:`ControlFlowGraph.from_instructions`. This is the case for graphs
    transformed by :func:`one_big_loop`.

    .. rubric:: How Does It Work?

    Recreating a structured program from a control-flow graph is a frighteningly
    complex and not yet well studied problem. There is an ancient theorem called
    `Structured Program Theorem`_ that essentially states "flowcharts"
    (basically the class of control-flow graphs we care about) can be reduced to
    structured programs in a very simple way with pretty unreadable output.
    It is listed under "folk version of the theorem", generating a single while
    loop [1]_:

    .. code::

        pc = 1
        while pc > 0:
            if pc == 1:
                block1()
                pc := successor of block 1
            elif pc == 2:
                block2()
                pc := successor of block 2
            ...

    This was *not* done here. Instead, we use a slightly smarter algorithm that
    reconstructs hierarchical control structure using `Dominator trees`_. Let's
    define some terms.

    node :math:`d` dominates a node :math:`n`
        A node :math:`d` dominates a node :math:`n` if every path from the entry
        node of the control-flow graph to :math:`n` must go through `d`. A node
        :math:`d` *strictly* dominates a node :math:`n` if it dominates
        :math:`n` and :math:`n \neq d`.

    dominance frontier
        The dominance frontier of a node :math:`d` is the set of all nodes
        :math:`n` such that :math:`d` dominates an immediate predecessor of
        :math:`n`, but does not strictly dominate :math:`n`.

    region
        A region is defined by an *entry edge* and *exit edge*, where the entry
        edge dominates the exit edge, the exit edge post-dominates the entry
        edge, and any cycle that includes one also includes the other.

    For example, the basic block for the condition of an if statement dominates
    all blocks in its *true* and *false* branches. Blocks for successor
    statements to that conditional statement are also dominated. The same thing
    holds for a while loop. So for CFGs generated from structured programs (and
    specifically for those generated by
    :meth:`ControlFlowGraph.from_instructions`), it suffices to concatenate
    statements into a sequence as long as each statement dominates the next one.
    The set of basic blocks in such a sequence is called a *region*
    (:class:`_Region`, see also [fcd16]_).

    Fortunately, this implementation can be restricted to these
    program-generated CFGs and those modified by :func:`one_big_loop`. That
    means we can create regions without problems for *almost all* basic blocks.
    The only difficulty are generated trampoline blocks. We handle those
    separately in an ugly method (:func:`_join_regions`).

    .. rubric:: Restrictions and Generalizations

    It is not completely clear what the above restrictions on input CFGs mean
    formally. Ignoring trampoline blocks, the restrictions are more strict than
    just having CFGs *reducible* [2]_. On the other hand, a generalized
    algorithm is a hard task, particularly for CFGs with arbitrary `goto`. The
    *relooper* of *Emscripten* was an attempt in a production JS-to-LLVM
    compiler [zak11]_. Iozzelli has revisited the problem for the *Cheerp*
    compiler using a dominator tree-based approach [ioz19]_. In their article,
    some ideas implemented here are discovered independently (e.g. our
    :class:`JumpTable` under the name "dispatcher"). Their general algorithms
    may be of interest for an improvement of this method [3]_ at the cost of a
    much more complex implementation, although the current one hopefully
    suffices.

    .. seealso::

        `Documentation of internal classes and functions <cfg_internals.html>`_ can be found on another page.

    .. _Structured Program Theorem: https://en.wikipedia.org/wiki/Structured_program_theorem
    .. _Dominator trees: https://en.wikipedia.org/wiki/Dominator_(graph_theory)
    .. [1] The attentive reader might notice that such an output would subsume :func:`one_big_loop` completely. Just applying the "folk theorem" would have been an easier way of implementation, but at the cost of almost completely unreadable output.
    .. [2] A `reducible`_ control-flow graph can be split into a directed acyclic graph of forward edges and a set of back edges where the target dominates the source.
    .. [3] In contrast to our pGCL programs, their reconstructed code is allowed to use *break* and *continue* statements.
    .. _reducible: https://en.wikipedia.org/wiki/Control-flow_graph#Reducibility
    .. [fcd16] "`The Region Problem <https://zneak.github.io/fcd/2016/02/17/structuring.html>`_"  on the `fcd` decompiler blog, 2016.
    .. [zak11] "`Emscripten: an LLVM-to-JavaScript compiler <https://raw.githubusercontent.com/kripken/emscripten/8a6e2d67c156d9eaedf88b752be4d1cf4242e088/docs/paper.pdf>`_", Alon Zakai, OOPSLA '11.
    .. [ioz19] "`Solving the structured control flow problem once and for all <https://medium.com/leaningtech/solving-the-structured-control-flow-problem-once-and-for-all-5123117b1ee2>`_", Yuri Iozzelli, 2019.

    """
    # Remove traps. Those are generated from
    for block in graph:
        if block.terminator.if_false is not None:
            false_block = graph[block.terminator.if_false]
            if false_block.is_trap():
                block.terminator = Terminator.goto(block.terminator.if_true)

    nx_graph = graph.to_networkx()
    frontiers = dominance_frontiers(nx_graph, graph.entry_id)
    region = _Region.initial(graph.entry_id)
    _expand_region(graph, frontiers, region)
    assert region.successor_id is None
    return region.instrs


@attr.s
class _Region:
    """
    A region is like a basic block, but it contains general program statements
    instead of only assignments and has
    """

    dominator_id: Optional[BasicBlockId] = attr.ib()
    instrs: List[Instr] = attr.ib()
    successor_id: Optional[BasicBlockId] = attr.ib()

    @staticmethod
    def initial(initial_block_id: Optional[BasicBlockId]) -> '_Region':
        return _Region(initial_block_id, [], initial_block_id)

    @staticmethod
    def child(parent_id: BasicBlockId,
              block_id: Optional[BasicBlockId]) -> '_Region':
        return _Region(parent_id, [], block_id)

    def dominates_successor(self, frontiers: _DominanceFrontiers) -> bool:
        assert self.dominator_id is not None
        return self.successor_id not in frontiers[self.dominator_id]

    def add_block(self, block: BasicBlock, instrs: List[Instr],
                  successor_id: Optional[BasicBlockId]):
        self.dominator_id = block.ident
        self.instrs.extend(
            (AsgnInstr(var, rhs) for var, rhs in block.assignments))
        self.instrs.extend(instrs)
        self.successor_id = successor_id


def _expand_region(graph: ControlFlowGraph, frontiers: _DominanceFrontiers,
                   region: _Region):
    """
    See :func:`reduce_cfg_to_instrs`. Given a control-flow graph for a program,
    the dominance frontiers and a block id, create a block (i.e. list of
    instructions) where each statement in the block dominates the next
    statements. All statements in that block eventually end up in the second
    returned item, the successor's block id.
    """
    # Expand the region while it dominates the successor
    while region.successor_id is not None and region.dominates_successor(
            frontiers):
        block_id = region.successor_id
        block = graph[block_id]
        terminator = block.terminator

        # An unconditional goto is added to the end of the region
        if terminator.is_goto():
            region.add_block(block, [], terminator.if_true)
        else:
            # A conditional branch may be either an if statement or a while loop
            if_region = _Region.child(block_id, terminator.if_true)
            _expand_region(graph, frontiers, if_region)
            else_region = _Region.child(block_id, terminator.if_false)
            _expand_region(graph, frontiers, else_region)

            if block.typ == BlockType.LOOP_HEAD:
                assert if_region.successor_id == block_id
                while_instr: Instr = WhileInstr(terminator.condition,
                                                if_region.instrs)
                instrs = [while_instr] + else_region.instrs
                region.add_block(block, instrs, else_region.successor_id)
            else:
                joined_regions = _join_regions(graph, if_region, else_region)
                assert if_region.successor_id == else_region.successor_id
                if_ctor: Any = IfInstr if terminator.kind == TerminatorKind.BOOLEAN else ChoiceInstr
                if_instr: Instr = if_ctor(terminator.condition,
                                          if_region.instrs, else_region.instrs)
                region.add_block(block, [if_instr], if_region.successor_id)
                # see _join_regions documentation
                if joined_regions:
                    break


def _join_regions(graph: ControlFlowGraph, region1: _Region,
                  region2: _Region) -> bool:
    """
    For if instructions, both branches must continue with the same block. Let's
    call this block the *join point*. For programs generated using
    :meth:`ControlFlowGraph.from_instructions`, this will be the case. However,
    programs modified by :func:`one_big_loop` do not necessarily have proper
    join points. An example can be seen in the documentation of
    :func:`one_big_loop`. We handle those cases here explicitly by duplicating
    trampoline blocks.
    """
    if region1.successor_id == region2.successor_id:
        return False
    succ1 = graph[
        region1.successor_id] if region1.successor_id is not None else None
    succ2 = graph[
        region2.successor_id] if region2.successor_id is not None else None

    if succ1 is not None and succ1.typ == BlockType.TRAMPOLINE and succ1.terminator.if_true == region2.successor_id:
        assert succ1.terminator.is_goto()
        region1.add_block(succ1, [], succ1.terminator.if_true)
        return True

    if succ2 is not None and succ2.typ == BlockType.TRAMPOLINE and succ2.terminator.if_true == region1.successor_id:
        assert succ2.terminator.is_goto()
        region2.add_block(succ2, [], succ2.terminator.if_true)
        return True

    if succ1 is not None and succ2 is not None and succ1.typ == BlockType.TRAMPOLINE and succ2.typ == BlockType.TRAMPOLINE and succ1.terminator.if_true == succ2.terminator.if_true:
        assert succ1.terminator.is_goto()
        assert succ2.terminator.is_goto()
        region1.add_block(succ1, [], succ1.terminator.if_true)
        region2.add_block(succ2, [], succ2.terminator.if_true)
        return True

    raise Exception(
        f"Block {region1.dominator_id} has two irregular branches that end at {region1.successor_id} resp. {region2.successor_id}"
    )


def program_one_big_loop(program: Program, pc_var: Var) -> Program:
    """
    Apply :func:`one_big_loop` to the CFG generated from the given instructions
    (using :meth:`ControlFlowGraph.from_instructions`), and convert back to a
    list of instructions using :func:`reduce_cfg_to_instrs`.
    A new program will be returned.

    Use `pc_var` as the name of the generated variable (see :class:`JumpTable`).
    It must not be used already and will be added to the program's declarations
    by this function.

    :raises AssertionError: if `pc_var` is already declared
    """
    res = program.to_skeleton()
    res.add_variable(pc_var, NatType(None))
    graph = ControlFlowGraph.from_instructions(program.instructions)
    one_big_loop(graph, pc_var)
    res.instructions = reduce_cfg_to_instrs(graph)
    return res
