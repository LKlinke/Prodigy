"""
Random programs for property-based testing (Hypothesis).

:func:`programs` is a Hypothesis strategy for small, valid programs: assignments, sampling from
the built-in distributions, ``if``, probabilistic choice, ``observe``, ``loop(n)``, bounded
``while`` loops, modulo and floor division, and queries. A :class:`GeneratedProgram` is rendered
in both syntaxes (:meth:`GeneratedProgram.source`), so one strategy tests ``.ppy`` and ``.pgcl``.

:class:`Features` switches constructs on and off. The defaults leave out the classes of programs
with known open bugs, so that the test suite stays green while they are open; each switch names
its issue. A fuzzing run with a switch turned on finds programs of that class.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple, Union

from hypothesis import strategies as st

VARIABLES = ("x", "y", "z")
"""The program variables (``nat``); every program declares all of them."""
COUNTER = "c"
"""The counter of the bounded ``while`` loops; only the loop itself assigns it."""
PROBABILITIES = ("1/2", "1/3", "2/3", "1/4", "3/4", "1/5", "0.5", "0.25", "0.1")


@dataclass(frozen=True)
class Features:  # pylint: disable=too-many-instance-attributes
    """Which constructs the generator may use."""
    loops: bool = True
    """``loop(n)`` and bounded ``while`` loops (``while (c < k) { …; c = c + 1; }``)."""
    observe: bool = True
    infinite: bool = True
    """Distributions with infinite support: ``geometric``, ``poisson``, ``negative_binomial``, ``logdist``."""
    modulo: bool = True
    """``e % q`` in guards and assignments."""
    floor_division: bool = True
    """``e // q`` (``.ppy`` only: :meth:`GeneratedProgram.source` refuses to render it in ``.pgcl``)."""
    two_variable_guards: bool = False
    """Guards comparing two variables (``x < y``); with infinite support they end with an error (#225)."""
    subtraction: bool = True
    """``-`` on ``nat`` (monus since #220)."""
    state_probabilities: bool = True
    """Probabilities that depend on a variable (``[1/(x+1)]``, ``bernoulli(1/(x+1))``; case split since #213)."""
    variable_copies: bool = False
    """Assignments ``a = b`` of a variable: wrong on GiNaC if ``b`` does not occur in the function (#232)."""
    categorical: bool = False
    """``categorical({v: p, …})``: documented, but the analysis rejects it (#233)."""
    max_depth: int = 2
    max_statements: int = 4


DEFAULT_FEATURES = Features()


# ------------------------------------------------------------------------------------------------
# A small program tree, rendered in both syntaxes
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Num:
    """A natural literal."""
    value: int


@dataclass(frozen=True)
class Var:
    """A variable."""
    name: str


@dataclass(frozen=True)
class Prob:
    """A probability literal (a fraction or a decimal)."""
    text: str


@dataclass(frozen=True)
class Op:
    """A binary operation; ``op`` is the .ppy spelling."""
    op: str
    lhs: "Expr"
    rhs: "Expr"


@dataclass(frozen=True)
class Not:
    """A negated condition."""
    expr: "Expr"


@dataclass(frozen=True)
class Call:
    """A built-in distribution."""
    name: str
    args: Tuple["Expr", ...]


@dataclass(frozen=True)
class Categorical:
    """``categorical({v: p, …})``."""
    items: Tuple[Tuple[int, str], ...]


Expr = Union[Num, Var, Prob, Op, Not, Call, Categorical]

_PGCL_OPS = {"==": "=", "and": "&", "or": "||", "**": "^"}
_PRECEDENCE = {"or": 1, "and": 2, "<": 4, "<=": 4, ">": 4, ">=": 4, "==": 4, "+": 5, "-": 5,
               "*": 6, "/": 6, "//": 6, "%": 6}


def render_expr(expr: Expr, syntax: str) -> str:
    """``expr`` in the syntax ``ppy`` or ``pgcl``; every operation is parenthesised."""
    if isinstance(expr, Num):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Prob):
        return expr.text
    if isinstance(expr, Not):
        return f"not ({render_expr(expr.expr, syntax)})"
    if isinstance(expr, Op):
        if syntax == "pgcl" and expr.op == "//":
            raise ValueError("floor division // has no .pgcl syntax")
        op =_PGCL_OPS.get(expr.op, expr.op) if syntax == "pgcl" else expr.op
        return f"({render_expr(expr.lhs, syntax)} {op} {render_expr(expr.rhs, syntax)})"
    if isinstance(expr, Call):
        return f"{expr.name}({', '.join(render_expr(arg, syntax) for arg in expr.args)})"
    if isinstance(expr, Categorical):
        if syntax == "pgcl":
            return " + ".join(f"{value} : {prob}" for value, prob in expr.items)
        return "categorical({" + ", ".join(f"{value}: {prob}" for value, prob in expr.items) + "})"
    raise TypeError(expr)


@dataclass(frozen=True)
class Assign:
    """``var = expr``."""
    var: str
    expr: Expr


@dataclass(frozen=True)
class Observe:
    """``observe(cond)``."""
    cond: Expr


@dataclass(frozen=True)
class If:
    """``if (cond) {…} else {…}``."""
    cond: Expr
    then: Tuple["Stmt", ...]
    orelse: Tuple["Stmt", ...]


@dataclass(frozen=True)
class Choice:
    """``{…} [prob] {…}``."""
    prob: Expr
    lhs: Tuple["Stmt", ...]
    rhs: Tuple["Stmt", ...]


@dataclass(frozen=True)
class Repeat:
    """``loop(n) {…}``."""
    times: int
    body: Tuple["Stmt", ...]


@dataclass(frozen=True)
class Bounded:
    """``c = 0; while (c < bound) { body; c = c + 1; }``."""
    bound: int
    body: Tuple["Stmt", ...]


Stmt = Union[Assign, Observe, If, Choice, Repeat, Bounded]


def _block(stmts: Sequence[Stmt], syntax: str, indent: str) -> str:
    inner = indent + "    "
    lines = [render_stmt(stmt, syntax, inner) for stmt in stmts] or [inner + "skip;"]
    return "{\n" + "\n".join(lines) + "\n" + indent + "}"


def render_stmt(stmt: Stmt, syntax: str, indent: str = "") -> str:
    """``stmt`` in the syntax ``ppy`` or ``pgcl``."""
    assign = ":=" if syntax == "pgcl" else "="
    if isinstance(stmt, Assign):
        return f"{indent}{stmt.var} {assign} {render_expr(stmt.expr, syntax)};"
    if isinstance(stmt, Observe):
        return f"{indent}observe({render_expr(stmt.cond, syntax)});"
    if isinstance(stmt, If):
        return (f"{indent}if ({render_expr(stmt.cond, syntax)}) {_block(stmt.then, syntax, indent)}"
                f" else {_block(stmt.orelse, syntax, indent)}")
    if isinstance(stmt, Choice):
        return (f"{indent}{_block(stmt.lhs, syntax, indent)} [{render_expr(stmt.prob, syntax)}] "
                f"{_block(stmt.rhs, syntax, indent)}")
    if isinstance(stmt, Repeat):
        return f"{indent}loop({stmt.times}) {_block(stmt.body, syntax, indent)}"
    if isinstance(stmt, Bounded):
        step = Assign(COUNTER, Op("+", Var(COUNTER), Num(1)))
        cond = render_expr(Op("<", Var(COUNTER), Num(stmt.bound)), syntax)
        return (f"{indent}{COUNTER} {assign} 0;\n"
                f"{indent}while ({cond}) {_block(stmt.body + (step,), syntax, indent)}")
    raise TypeError(stmt)


@dataclass(frozen=True)
class Query:
    """``Pr[cond]`` (``kind == "Pr"``) or ``Ex[expr]`` (``kind == "Ex"``)."""
    kind: str
    expr: Expr


@dataclass(frozen=True)
class GeneratedProgram:
    """A generated program; :meth:`source` renders it."""
    body: Tuple[Stmt, ...]
    queries: Tuple[Query, ...] = field(default=())

    @property
    def while_loops(self) -> int:
        """The number of ``while`` loops (all bounded)."""
        def count(stmts: Sequence[Stmt]) -> int:
            total = 0
            for stmt in stmts:
                if isinstance(stmt, Bounded):
                    total += 1 + count(stmt.body)
                elif isinstance(stmt, If):
                    total += count(stmt.then) + count(stmt.orelse)
                elif isinstance(stmt, Choice):
                    total += count(stmt.lhs) + count(stmt.rhs)
                elif isinstance(stmt, Repeat):
                    total += count(stmt.body)
            return total
        return count(self.body)

    @property
    def max_iterations(self) -> int:
        """An unrolling depth after which every bounded loop has terminated."""
        def bound(stmts: Sequence[Stmt]) -> int:
            result = 0
            for stmt in stmts:
                if isinstance(stmt, Bounded):
                    result = max(result, stmt.bound + 1, bound(stmt.body))
                elif isinstance(stmt, If):
                    result = max(result, bound(stmt.then), bound(stmt.orelse))
                elif isinstance(stmt, Choice):
                    result = max(result, bound(stmt.lhs), bound(stmt.rhs))
                elif isinstance(stmt, Repeat):
                    result = max(result, bound(stmt.body))
            return result
        return bound(self.body)

    def decisions(self) -> dict:
        """Loop decisions under which the analysis is exact: unroll past every bound."""
        if not self.while_loops:
            return {}
        return {"*": f"unroll:iterations={self.max_iterations + 1}"}

    def source(self, syntax: str = "ppy") -> str:
        """The program text in ``ppy`` or ``pgcl``."""
        lines = []
        names = list(VARIABLES) + ([COUNTER] if self.while_loops else [])
        for name in names:
            lines.append(f"nat {name};" if syntax == "pgcl" else f"{name}: nat;")
        lines.extend(render_stmt(stmt, syntax) for stmt in self.body)
        for query in self.queries:
            text = render_expr(query.expr, syntax)
            if syntax == "pgcl":
                lines.append(f"?{query.kind}[{text}];")
            else:
                lines.append(f"query {query.kind}[{text}];")
        lines.append("!Print;" if syntax == "pgcl" else "print;")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------------------------------------

def _variables() -> st.SearchStrategy[Var]:
    return st.sampled_from(VARIABLES).map(Var)


def probabilities(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Expr]:
    """A probability: a literal, or (with ``state_probabilities``) ``1/(v + 1)``."""
    literal = st.sampled_from(PROBABILITIES).map(Prob)
    if not features.state_probabilities:
        return literal
    return st.one_of(literal, _variables().map(lambda v: Op("/", Num(1), Op("+", v, Num(1)))))


def arithmetic(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Expr]:
    """A natural-valued expression over the variables."""
    atoms = st.one_of(st.integers(0, 4).map(Num), _variables())
    options = [atoms if features.variable_copies else st.integers(0, 4).map(Num),
               st.builds(lambda a, b: Op("+", a, b), atoms, atoms),
               st.builds(lambda k, v: Op("*", Num(k), v), st.integers(2, 3), _variables())]
    if features.modulo:
        options.append(st.builds(lambda v, q: Op("%", v, Num(q)), _variables(), st.integers(2, 3)))
    if features.floor_division:
        options.append(st.builds(lambda v, q: Op("//", v, Num(q)), _variables(), st.integers(2, 3)))
    if features.subtraction:
        options.append(st.builds(lambda a, b: Op("-", a, b), atoms, atoms))
    return st.one_of(*options)


def conditions(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Expr]:
    """A guard over the variables."""
    compare = st.sampled_from(("<", "<=", ">", ">=", "=="))
    simple = [st.builds(lambda v, op, k: Op(op, v, Num(k)), _variables(), compare, st.integers(0, 3))]
    if features.modulo:
        simple.append(st.builds(lambda v, q, r: Op("==", Op("%", v, Num(q)), Num(r % q)),
                                _variables(), st.integers(2, 3), st.integers(0, 2)))
    if features.two_variable_guards:
        simple.append(st.builds(lambda a, op, b: Op(op, a, b), _variables(), compare, _variables()))
    atom = st.one_of(*simple)
    return st.one_of(atom, atom.map(Not),
                     st.builds(lambda op, a, b: Op(op, a, b), st.sampled_from(("and", "or")), atom, atom))


def samples(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Expr]:
    """A built-in distribution."""
    prob = probabilities(features)
    small = st.integers(0, 3)
    options = [
        st.builds(lambda p: Call("bernoulli", (p,)), prob),
        st.builds(lambda a, d: Call("unif", (Num(a), Num(a + d))), small, small),
        st.builds(lambda n, p: Call("binomial", (Num(n), p)), st.integers(1, 3), st.sampled_from(PROBABILITIES).map(Prob)),
    ]
    if features.categorical:
        options.append(st.lists(st.tuples(st.integers(0, 4), st.sampled_from(((1, 2), (1, 3), (1, 4), (1, 6)))),
                                min_size=1, max_size=3).map(_categorical))
    if features.infinite:
        literal = st.sampled_from(("1/2", "1/3", "2/3", "1/4", "0.5")).map(Prob)
        options += [
            st.builds(lambda p: Call("geometric", (p,)), literal),
            st.builds(lambda l: Call("poisson", (l,)), st.sampled_from(("1", "2", "1/2")).map(Prob)),
            st.builds(lambda r, p: Call("negative_binomial", (Num(r), p)), st.integers(1, 2), literal),
            st.builds(lambda p: Call("logdist", (p,)), st.sampled_from(("1/2", "1/3")).map(Prob)),
        ]
    return st.one_of(*options)


def _categorical(items: List[Tuple[int, Tuple[int, int]]]) -> Categorical:
    """Values with weights 1/n each; the last value takes the rest, so the probabilities sum to 1."""
    from fractions import Fraction  # pylint: disable=import-outside-toplevel
    values: List[Tuple[int, str]] = []
    rest = Fraction(1)
    used = set()
    for value, (num, den) in items:
        if value in used:
            continue
        used.add(value)
        weight = Fraction(num, den)
        if weight >= rest:
            break
        values.append((value, str(weight)))
        rest -= weight
    last = max(used, default=0) + 1
    values.append((last, str(rest)))
    return Categorical(tuple(values))


def statements(features: Features = DEFAULT_FEATURES, depth: Optional[int] = None) -> st.SearchStrategy[Stmt]:
    """A statement; blocks nest up to ``depth`` (default ``features.max_depth``)."""
    depth = features.max_depth if depth is None else depth
    var = st.sampled_from(VARIABLES)
    sample = st.builds(Assign, var, samples(features))
    leaves = [st.builds(Assign, var, arithmetic(features)), sample, sample]
    if features.observe:
        leaves.append(st.builds(Observe, conditions(features)))
    leaf = st.one_of(*leaves)
    if depth <= 0:
        return leaf
    block = st.lists(statements(features, depth - 1), min_size=0, max_size=2).map(tuple)
    compound = [st.builds(If, conditions(features), block, block),
                st.builds(Choice, probabilities(features), block, block)]
    if features.loops:
        compound += [st.builds(Repeat, st.integers(1, 3), block),
                     st.builds(Bounded, st.integers(1, 3), block)]
    return st.one_of(leaf, leaf, *compound)


def _no_nested_loops(stmt: Stmt, inside: bool = False) -> bool:
    if isinstance(stmt, Bounded):
        return not inside and all(_no_nested_loops(s, True) for s in stmt.body)
    if isinstance(stmt, If):
        return all(_no_nested_loops(s, inside) for s in stmt.then + stmt.orelse)
    if isinstance(stmt, Choice):
        return all(_no_nested_loops(s, inside) for s in stmt.lhs + stmt.rhs)
    if isinstance(stmt, Repeat):
        return all(_no_nested_loops(s, inside) for s in stmt.body)
    return True


def queries(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Query]:
    """``Pr[g]`` or ``Ex[e]``."""
    expectation = st.one_of(_variables(), st.builds(lambda a, b: Op("+", a, b), _variables(), _variables()))
    return st.one_of(st.builds(lambda c: Query("Pr", c), conditions(replace(features, two_variable_guards=False))),
                     st.builds(lambda e: Query("Ex", e), expectation))


def programs(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[GeneratedProgram]:
    """Small valid programs (bounded ``while`` loops do not nest: they share one counter)."""
    body = st.lists(statements(features), min_size=1, max_size=features.max_statements) \
        .filter(lambda stmts: all(_no_nested_loops(s) for s in stmts)).map(tuple)
    return st.builds(GeneratedProgram, body, st.lists(queries(features), min_size=1, max_size=3).map(tuple))
