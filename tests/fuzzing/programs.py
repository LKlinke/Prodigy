"""
Random programs for property-based testing (Hypothesis).

:func:`programs` is a Hypothesis strategy for small, valid programs: assignments, sampling from
the built-in distributions, ``if``, probabilistic choice, ``observe``, ``loop(n)``, bounded
``while`` loops, modulo and floor division, and queries. A :class:`GeneratedProgram` is rendered
in ``.ppy`` (:meth:`GeneratedProgram.source`; the ``.pgcl`` syntax was removed in v0.6.0).

With :attr:`Features.parameters`, probabilities may depend on the parameters ``p`` and ``q``
(``param p: real;``), from a class of expressions that are admissible probabilities for every value
of the parameters in (0, 1) (:data:`PARAMETRIC_PROBABILITIES`). :meth:`GeneratedProgram.instantiate`
replaces the parameters by values, which gives a program with constant probabilities (#453).

:class:`Features` switches constructs on and off. The defaults leave out the classes of programs
with known open bugs, so that the test suite stays green while they are open; each switch names
its issue. A fuzzing run with a switch turned on finds programs of that class.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from hypothesis import strategies as st

VARIABLES = ("x", "y", "z")
"""The program variables (``nat``); every program declares all of them."""
COUNTER = "c"
"""The counter of the bounded ``while`` loops; only the loop itself assigns it."""
PROBABILITIES = ("1/2", "1/3", "2/3", "1/4", "3/4", "1/5", "0.5", "0.25", "0.1")
PARAMETERS = ("p", "q")
"""The parameters (``param p: real;``) of parametric programs; a program declares those it uses."""
PARAMETER_VALUES = tuple(Fraction(*v) for v in ((1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (4, 5)))
"""The admissible values a parameter is instantiated with: rationals in (0, 1)."""


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
    """``e // q``."""
    two_variable_guards: bool = False
    """Guards comparing two variables (``x < y``); with infinite support they end with an error (#225)."""
    subtraction: bool = True
    """``-`` on ``nat`` (monus since #220)."""
    state_probabilities: bool = True
    """Probabilities that depend on a variable (``[1/(x+1)]``, ``bernoulli(1/(x+1))``; case split since #213)."""
    categorical: bool = True
    """``categorical({v: p, …})`` (computed since #233)."""
    parameters: bool = False
    """Probabilities with the parameters ``p`` and ``q`` (:data:`PARAMETRIC_PROBABILITIES`, #453)."""
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
class Param:
    """A parameter (``param p: real;``)."""
    name: str


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


Expr = Union[Num, Var, Prob, Param, Op, Not, Call, Categorical]

_PRECEDENCE = {"or": 1, "and": 2, "<": 4, "<=": 4, ">": 4, ">=": 4, "==": 4, "+": 5, "-": 5,
               "*": 6, "/": 6, "//": 6, "%": 6}


def render_expr(expr: Expr, syntax: str) -> str:
    """``expr`` in the syntax ``ppy`` (the only one); every operation is parenthesised."""
    if isinstance(expr, Num):
        return str(expr.value)
    if isinstance(expr, (Var, Param)):
        return expr.name
    if isinstance(expr, Prob):
        return expr.text
    if isinstance(expr, Not):
        return f"not ({render_expr(expr.expr, syntax)})"
    if isinstance(expr, Op):
        op = expr.op
        return f"({render_expr(expr.lhs, syntax)} {op} {render_expr(expr.rhs, syntax)})"
    if isinstance(expr, Call):
        return f"{expr.name}({', '.join(render_expr(arg, syntax) for arg in expr.args)})"
    if isinstance(expr, Categorical):
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
    """``stmt`` in the syntax ``ppy``."""
    assign = "="
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


def _expr_parameters(expr: Expr) -> List[str]:
    if isinstance(expr, Param):
        return [expr.name]
    if isinstance(expr, Op):
        return _expr_parameters(expr.lhs) + _expr_parameters(expr.rhs)
    if isinstance(expr, Not):
        return _expr_parameters(expr.expr)
    if isinstance(expr, Call):
        return [name for arg in expr.args for name in _expr_parameters(arg)]
    return []


_ARITHMETIC = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def _value(expr: Expr, values: Mapping[str, Fraction]) -> Fraction:
    """The value of a parametric probability (built from parameters, literals and arithmetic)."""
    if isinstance(expr, Param):
        return values[expr.name]
    if isinstance(expr, (Num, Prob)):
        return Fraction(str(expr.value) if isinstance(expr, Num) else expr.text)
    if isinstance(expr, Op) and expr.op in ("+", "-", "*", "/"):
        lhs, rhs = _value(expr.lhs, values), _value(expr.rhs, values)
        return _ARITHMETIC[expr.op](lhs, rhs)
    raise TypeError(expr)


def _instantiate_expr(expr: Expr, values: Mapping[str, Fraction]) -> Expr:
    """``expr`` with every parametric subexpression replaced by its value (a literal)."""
    if not _expr_parameters(expr):
        return expr
    if isinstance(expr, Call):
        return Call(expr.name, tuple(_instantiate_expr(arg, values) for arg in expr.args))
    if isinstance(expr, Not):
        return Not(_instantiate_expr(expr.expr, values))
    if isinstance(expr, Op) and expr.op not in ("+", "-", "*", "/"):
        return Op(expr.op, _instantiate_expr(expr.lhs, values), _instantiate_expr(expr.rhs, values))
    return Prob(str(_value(expr, values)))


def _children(stmt: "Stmt") -> List[Tuple["Stmt", ...]]:
    if isinstance(stmt, If):
        return [stmt.then, stmt.orelse]
    if isinstance(stmt, Choice):
        return [stmt.lhs, stmt.rhs]
    if isinstance(stmt, (Repeat, Bounded)):
        return [stmt.body]
    return []


def _stmt_parameters(stmt: "Stmt") -> List[str]:
    own: List[str] = []
    if isinstance(stmt, Assign):
        own = _expr_parameters(stmt.expr)
    elif isinstance(stmt, (Observe, If)):
        own = _expr_parameters(stmt.cond)
    elif isinstance(stmt, Choice):
        own = _expr_parameters(stmt.prob)
    return own + [name for block in _children(stmt) for child in block for name in _stmt_parameters(child)]


def _instantiate_stmt(stmt: "Stmt", values: Mapping[str, Fraction]) -> "Stmt":
    def block(stmts: Tuple["Stmt", ...]) -> Tuple["Stmt", ...]:
        return tuple(_instantiate_stmt(child, values) for child in stmts)
    if isinstance(stmt, Assign):
        return Assign(stmt.var, _instantiate_expr(stmt.expr, values))
    if isinstance(stmt, Observe):
        return Observe(_instantiate_expr(stmt.cond, values))
    if isinstance(stmt, If):
        return If(_instantiate_expr(stmt.cond, values), block(stmt.then), block(stmt.orelse))
    if isinstance(stmt, Choice):
        return Choice(_instantiate_expr(stmt.prob, values), block(stmt.lhs), block(stmt.rhs))
    if isinstance(stmt, Repeat):
        return Repeat(stmt.times, block(stmt.body))
    if isinstance(stmt, Bounded):
        return Bounded(stmt.bound, block(stmt.body))
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
    def parameters(self) -> Tuple[str, ...]:
        """The parameters the program uses, sorted."""
        return tuple(sorted({name for stmt in self.body for name in _stmt_parameters(stmt)}
                            | {name for query in self.queries for name in _expr_parameters(query.expr)}))

    def instantiate(self, values: Mapping[str, Fraction]) -> "GeneratedProgram":
        """The program with each parameter replaced by its value: every parametric probability becomes the
        literal of its value."""
        return GeneratedProgram(tuple(_instantiate_stmt(stmt, values) for stmt in self.body),
                                tuple(Query(q.kind, _instantiate_expr(q.expr, values)) for q in self.queries))

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
        """The program text in ``ppy``."""
        lines = [f"param {name}: real;" for name in self.parameters]
        names = list(VARIABLES) + ([COUNTER] if self.while_loops else [])
        for name in names:
            lines.append(f"{name}: nat;")
        lines.extend(render_stmt(stmt, syntax) for stmt in self.body)
        for query in self.queries:
            text = render_expr(query.expr, syntax)
            lines.append(f"query {query.kind}[{text}];")
        lines.append("print;")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------------------------------------

def _variables() -> st.SearchStrategy[Var]:
    return st.sampled_from(VARIABLES).map(Var)


def _parametric(builder) -> st.SearchStrategy[Expr]:
    return st.sampled_from(PARAMETERS).map(lambda name: builder(Param(name)))


def parametric_probabilities() -> st.SearchStrategy[Expr]:
    """
    A probability with parameters, from the class ``p``, ``1 - p``, ``p/2``, ``(1 - p)/2``, ``(1 + p)/2``
    and ``p*q``: affine maps ``a + b·p`` with ``a`` and ``a + b`` in [0, 1], and the product of two such maps.
    Each maps the parameter values in (0, 1) into (0, 1), so it is an admissible probability (and an argument
    of ``geometric``, which must be positive) for every value the program can be instantiated with
    (:data:`PARAMETER_VALUES`).
    """
    half = Num(2)
    return st.one_of(
        _parametric(lambda p: p),
        _parametric(lambda p: Op("-", Num(1), p)),
        _parametric(lambda p: Op("/", p, half)),
        _parametric(lambda p: Op("/", Op("-", Num(1), p), half)),
        _parametric(lambda p: Op("/", Op("+", Num(1), p), half)),
        st.just(Op("*", Param(PARAMETERS[0]), Param(PARAMETERS[1]))),
    )


def _constant_probabilities(features: Features) -> st.SearchStrategy[Expr]:
    """A literal probability, or (with ``parameters``) a parametric one."""
    literal = st.sampled_from(PROBABILITIES).map(Prob)
    return st.one_of(literal, parametric_probabilities()) if features.parameters else literal


def probabilities(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Expr]:
    """A probability: a literal, (with ``parameters``) a parametric one, or (with ``state_probabilities``)
    ``1/(v + 1)``."""
    literal = _constant_probabilities(features)
    if not features.state_probabilities:
        return literal
    return st.one_of(literal, _variables().map(lambda v: Op("/", Num(1), Op("+", v, Num(1)))))


def arithmetic(features: Features = DEFAULT_FEATURES) -> st.SearchStrategy[Expr]:
    """A natural-valued expression over the variables."""
    atoms = st.one_of(st.integers(0, 4).map(Num), _variables())
    options = [atoms,
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
        st.builds(lambda n, p: Call("binomial", (Num(n), p)), st.integers(1, 3), _constant_probabilities(features)),
    ]
    if features.categorical:
        options.append(st.lists(st.tuples(st.integers(0, 4), st.sampled_from(((1, 2), (1, 3), (1, 4), (1, 6)))),
                                min_size=1, max_size=3).map(_categorical))
    if features.infinite:
        literal: st.SearchStrategy[Expr] = st.sampled_from(("1/2", "1/3", "2/3", "1/4", "0.5")).map(Prob)
        rate: st.SearchStrategy[Expr] = st.sampled_from(("1", "2", "1/2")).map(Prob)
        if features.parameters:
            literal = st.one_of(literal, parametric_probabilities())
            rate = st.one_of(rate, st.sampled_from(PARAMETERS).map(Param))
        options += [
            st.builds(lambda p: Call("geometric", (p,)), literal),
            st.builds(lambda l: Call("poisson", (l,)), rate),
            st.builds(lambda r, p: Call("negative_binomial", (Num(r), p)), st.integers(1, 2), literal),
            st.builds(lambda p: Call("logdist", (p,)), st.sampled_from(("1/2", "1/3")).map(Prob)),
        ]
    return st.one_of(*options)


def _categorical(items: List[Tuple[int, Tuple[int, int]]]) -> Categorical:
    """Values with weights 1/n each; the last value takes the rest, so the probabilities sum to 1."""
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


def parameter_values() -> st.SearchStrategy[Dict[str, Fraction]]:
    """A value for every parameter, from :data:`PARAMETER_VALUES`."""
    return st.fixed_dictionaries({name: st.sampled_from(PARAMETER_VALUES) for name in PARAMETERS})
