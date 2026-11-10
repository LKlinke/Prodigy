"""
A Monte-Carlo simulator for programs: an independent oracle for the analysis.

The simulator runs a parsed program (:class:`~prodigy.pgcl.ast.Program`) many times on concrete
states with a seeded random number generator. It shares only the parser with the analysis, none of
the engines' code, so a wrong generating function, a wrong filter or a wrong normalisation in an
engine shows up as a difference between an exact result and the simulation. The tests use it in two
ways: the examples in ``pgfexamples/`` are checked against it (``tests/simulation``), and random
programs from the fuzzer (``tests/fuzzing``) are.

Semantics (the same as the analysis, see ``docs/source/language.md``):

* every variable starts at 0 (``false``), unless an initial state is given;
* ``observe(g)`` rejects the run if ``g`` does not hold (a *violated* run); the results are
  conditioned on the runs that are not violated, like the normalisation of the analysis;
* ``abort`` ends the run without a result (an *aborted* run, mass that is lost);
* ``-`` is truncated subtraction (monus, ``max(a - b, 0)``), the semantics decided in #220; the
  analysis rejects subtractions that may become negative, so the two agree where it computes;
* ``/`` is rational division; assigning a value that is not a natural number to a variable is an
  error of the program (:class:`SimulationError`), the analysis rejects such programs too;
* comparisons are evaluated over the rationals, booleans count as 1 and 0;
* ``while`` loops run until a budget of loop iterations per run is used up; such a run is
  *unfinished* and widens the confidence intervals (it might still terminate or not);
* ``query { x = e; observe(g); }`` conditions locally: ``x`` is drawn again until ``g`` holds;
* ``infer(f, args)`` / ``sample(d)`` draw by rejection sampling: ``f(args)`` is run until a run
  terminates and passes the observations inside ``f``.

Not simulated (:class:`UnsupportedConstruct`): parameters, ``sample_pgf``, ``unif_c``, and
``Pr[d …]``/``Ex[d]`` of ``dist`` values inside a program.

The comparison (:func:`compare`) is statistical, not a proof: an exact value of the analysis must
lie in a confidence interval of the simulation (exact Clopper-Pearson intervals for probabilities,
normal intervals for expected values). With a fixed seed the outcome is reproducible, and the default ``z = 5`` makes
a false alarm for a correct result very unlikely (about 6e-7 per compared number). Small errors
(below the width of the interval) and errors in the tail of a distribution beyond what the runs
reach are not found.
"""
from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from prodigy.pgcl.ast import (AbortInstr, AsgnInstr, Binop, BinopExpr, BoolLitExpr, CategoricalExpr,
                              ChoiceInstr, ExpectationExpr, ExpectationInstr, Expr, FunctionCallExpr,
                              IfInstr, InferExpr, Instr, LoopInstr, NatLitExpr, ObserveInstr,
                              PlotInstr, PrintInstr, ProbabilityExpr, ProbabilityQueryInstr, Program,
                              RealLitExpr, SampleExpr, SkipInstr, Unop, UnopExpr, VarExpr, WhileInstr)
from prodigy.pgcl.ast.instructions import OptimizationQuery, QueryInstr
from prodigy.pgcl.ast.types import BoolType, DistributionType

Number = Union[int, Fraction]
Value = Union[int, Fraction, bool, "DistValue"]
State = Dict[str, Any]


class SimulationError(Exception):
    """The program goes wrong on a concrete run (e.g. a value that is not a natural number)."""


class UnsupportedConstruct(SimulationError):
    """The simulator does not implement this construct."""


class _Rejected(Exception):
    """A run violated an observation."""


class _Aborted(Exception):
    """A run reached ``abort``."""


class _Unfinished(Exception):
    """A run used up its loop budget."""


@dataclass(frozen=True)
class DistValue:
    """The value of a ``dist`` variable: the call ``function(*arguments)`` to infer from."""
    function: str
    arguments: Tuple[Tuple[str, Any], ...]


@dataclass
class Simulation:
    """The outcome of :func:`simulate`: the final states of the accepted runs and the counts of the rest."""
    runs: int
    samples: List[Dict[str, Any]] = field(default_factory=list)
    """Final states (program variables without block and ``dist`` variables) of the runs that
    terminated and passed every observation."""
    violated: int = 0
    aborted: int = 0
    unfinished: int = 0

    @property
    def conditioned_runs(self) -> int:
        """The runs that did not violate an observation; results are conditioned on them."""
        return self.runs - self.violated

    def probability(self, event: Callable[[Dict[str, Any]], bool], z: float = 5.0) -> "Estimate":
        """The probability of ``event`` among the accepted runs, as the analysis reports it after
        normalisation: P(terminates, passes, event) / P(passes)."""
        hits = sum(1 for state in self.samples if event(state))
        return Estimate.proportion(hits, self.conditioned_runs, self.unfinished, z)

    def mass(self, z: float = 5.0) -> "Estimate":
        """The mass of the normalised output distribution (P(terminates | passes))."""
        return Estimate.proportion(len(self.samples), self.conditioned_runs, self.unfinished, z)

    def violation(self, z: float = 5.0) -> "Estimate":
        """The probability of violating an observation (the unnormalised error mass)."""
        return Estimate.proportion(self.violated, self.runs, self.unfinished, z)

    def expectation(self, value: Callable[[Dict[str, Any]], Number], z: float = 5.0) -> "Estimate":
        """The expected value of ``value`` on the normalised output (sub-)distribution."""
        n = self.conditioned_runs
        if n == 0:
            return Estimate(math.nan, -math.inf, math.inf, 0)
        values = [float(value(state)) for state in self.samples]
        mean = sum(values) / n
        square = sum(v * v for v in values) / n
        deviation = math.sqrt(max(square - mean * mean, 0.0) * n / max(n - 1, 1))
        # with few distinct values the sample deviation may be tiny although the true one is not;
        # one more unit of the largest observed value keeps the interval honest for small n
        spread = z * max(deviation, max((abs(v) for v in values), default=0.0) / n) / math.sqrt(n)
        upper = math.inf if self.unfinished else mean + spread
        return Estimate(mean, mean - spread, upper, n)


@dataclass(frozen=True)
class Estimate:
    """An estimate with a confidence interval ``[lower, upper]`` from ``n`` runs."""
    value: float
    lower: float
    upper: float
    n: int

    @staticmethod
    def proportion(hits: int, n: int, unfinished: int, z: float) -> "Estimate":
        """``hits`` of ``n`` runs, of which ``unfinished`` runs might still count (the interval of
        hits/n from below and of (hits + unfinished)/n from above, :func:`proportion_interval`)."""
        if n == 0:
            return Estimate(math.nan, 0.0, 1.0, 0)
        return Estimate(hits / n, proportion_interval(hits, n, z)[0],
                        proportion_interval(min(hits + unfinished, n), n, z)[1], n)

    def contains(self, lower: float, upper: Optional[float] = None) -> bool:
        """Whether the interval meets ``[lower, upper]`` (a single exact value if ``upper`` is None)."""
        upper = lower if upper is None else upper
        if math.isnan(lower) or math.isnan(upper):
            return False
        slack = 1e-9  # floating-point rounding of the reported values
        return lower <= self.upper + slack and self.lower - slack <= upper


@functools.lru_cache(maxsize=4096)
def proportion_interval(hits: int, n: int, z: float) -> Tuple[float, float]:
    """
    The exact (Clopper-Pearson) confidence interval of the proportion ``hits / n``, at the level of
    a normal interval with ``z`` standard deviations (z = 5: a correct value lies outside with a
    probability below 6e-7). Unlike the Wilson or normal intervals it is conservative also for
    rare events with only a few hits.
    """
    # pylint: disable=import-outside-toplevel
    from scipy.stats import beta  # type: ignore[import-untyped]
    alpha = math.erfc(z / math.sqrt(2))
    lower = 0.0 if hits == 0 else float(beta.ppf(alpha / 2, hits, n - hits + 1))
    upper = 1.0 if hits >= n else float(beta.ppf(1 - alpha / 2, hits + 1, n - hits))
    return lower, upper


# ------------------------------------------------------------------------------------------------
# Running programs
# ------------------------------------------------------------------------------------------------

_QUERIES = (ProbabilityQueryInstr, ExpectationInstr, PrintInstr, PlotInstr, OptimizationQuery)


def statements(program: Program) -> List[Instr]:
    """The instructions of ``program`` before its queries."""
    result: List[Instr] = []
    for instr in program.instructions:
        if isinstance(instr, _QUERIES):
            break
        result.append(instr)
    return result


def queries(program: Program) -> List[Instr]:
    """The queries at the end of ``program``."""
    return program.instructions[len(statements(program)):]


def simulate(program: Program, runs: int = 10000, *, seed: Any = 0, loop_budget: int = 10000,
             initial: Optional[Mapping[str, Any]] = None) -> Simulation:
    """
    Runs ``program`` ``runs`` times (see the module documentation for the semantics).

    :param seed: seed of the random number generator; the same seed gives the same simulation.
    :param loop_budget: loop iterations per run (``while`` bodies, rejection-sampling attempts and
        ``loop(n)`` bodies together); a run that needs more is unfinished.
    :param initial: initial values of variables (default 0).
    :raises UnsupportedConstruct: if the program uses a construct the simulator does not implement.
    :raises SimulationError: if a run goes wrong (e.g. assigns a value that is not natural).
    """
    if program.parameters:
        raise UnsupportedConstruct(f"parameters ({', '.join(sorted(program.parameters))})")
    machine = _Machine(program, random.Random(seed), loop_budget)
    body = statements(program)
    hidden = set(program.block_variables) | {var for var, typ in program.variables.items()
                                             if isinstance(typ, DistributionType)}
    shown = [var for var in program.variables if var not in hidden]
    result = Simulation(runs)
    for _ in range(runs):
        state: State = {var: 0 for var in program.variables}
        for var in program.block_variables:
            state[var] = 0
        if initial:
            state.update({var: value for var, value in initial.items() if var not in program.block_variables})
        machine.budget = loop_budget
        try:
            machine.block(body, state)
        except _Rejected:
            result.violated += 1
            continue
        except _Aborted:
            result.aborted += 1
            continue
        except _Unfinished:
            result.unfinished += 1
            continue
        result.samples.append({var: state[var] for var in shown})
    return result


def _natural(value: Any, what: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise SimulationError(f"{what} is not a natural number: {value}")
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise SimulationError(f"{what} is not a natural number: {value}")
    return value


def _probability(value: Any, what: str) -> Fraction:
    if isinstance(value, bool):
        value = int(value)
    prob = Fraction(value)
    if not 0 <= prob <= 1:
        raise SimulationError(f"{what} is not a probability: {value}")
    return prob


class _Machine:
    """Executes instructions on one concrete state."""

    def __init__(self, program: Program, rng: random.Random, loop_budget: int):
        self.program = program
        self.rng = rng
        self.budget = loop_budget
        self.loop_budget = loop_budget

    # -- instructions ------------------------------------------------------------------------------

    def block(self, instrs: Sequence[Instr], state: State) -> None:
        """Runs a list of instructions."""
        for instr in instrs:
            self.instruction(instr, state)

    def _tick(self) -> None:
        self.budget -= 1
        if self.budget < 0:
            raise _Unfinished()

    def instruction(self, instr: Instr, state: State) -> None:  # pylint: disable=too-many-branches
        """Runs one instruction."""
        if isinstance(instr, SkipInstr):
            return
        if isinstance(instr, AbortInstr):
            raise _Aborted()
        if isinstance(instr, AsgnInstr):
            state[instr.lhs] = self.assigned(instr.lhs, instr.rhs, state)
        elif isinstance(instr, ObserveInstr):
            if not self.truth(instr.cond, state):
                raise _Rejected()
        elif isinstance(instr, IfInstr):
            self.block(instr.true if self.truth(instr.cond, state) else instr.false, state)
        elif isinstance(instr, ChoiceInstr):
            prob = _probability(self.value(instr.prob, state), f"the probability {instr.prob}")
            self.block(instr.lhs if self.bernoulli(prob) else instr.rhs, state)
        elif isinstance(instr, WhileInstr):
            while self.truth(instr.cond, state):
                self._tick()
                self.block(instr.body, state)
        elif isinstance(instr, LoopInstr):
            for _ in range(_natural(self.value(instr.iterations, state), "the number of iterations")):
                self.block(instr.body, state)
        elif isinstance(instr, QueryInstr):
            self.query_block(instr, state)
        else:
            raise UnsupportedConstruct(f"the instruction {instr}")

    def query_block(self, instr: QueryInstr, state: State) -> None:
        """``query { x = e; observe(g); }`` conditions locally: ``x`` is drawn again until ``g``
        holds, the rest of the state stays (the analysis normalises per state of the others)."""
        if len(instr.instrs) != 2 or not isinstance(instr.instrs[0], AsgnInstr) \
                or not isinstance(instr.instrs[1], ObserveInstr):
            raise UnsupportedConstruct(f"the query block {instr}")
        assignment, observation = instr.instrs
        assert isinstance(assignment, AsgnInstr) and isinstance(observation, ObserveInstr)
        while True:
            self._tick()
            trial = dict(state)
            trial[assignment.lhs] = self.assigned(assignment.lhs, assignment.rhs, trial)
            if self.truth(observation.cond, trial):
                state.update(trial)
                return

    def assigned(self, var: str, rhs: Expr, state: State) -> Any:
        """The value of ``rhs`` for the assignment to ``var``, checked against the type of ``var``."""
        value = self.value(rhs, state)
        typ = self.program.variables.get(var)
        if isinstance(typ, DistributionType) or isinstance(value, DistValue):
            return value
        if isinstance(typ, BoolType):
            number = _natural(value, f"the value assigned to {var}")
            if number > 1:
                raise SimulationError(f"the value assigned to the bool {var} is {number}")
            return number
        return _natural(value, f"the value assigned to {var}")

    # -- expressions -------------------------------------------------------------------------------

    def truth(self, expr: Expr, state: State) -> bool:
        """The value of a condition."""
        value = self.value(expr, state)
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise SimulationError(f"the condition {expr} has the value {value}")

    def value(self, expr: Expr, state: State) -> Any:  # pylint: disable=too-many-return-statements
        """The value of an expression in ``state`` (sampling expressions draw a value)."""
        if isinstance(expr, NatLitExpr):
            return expr.value
        if isinstance(expr, BoolLitExpr):
            return expr.value
        if isinstance(expr, RealLitExpr):
            if expr.is_infinite():
                raise UnsupportedConstruct("infinity in a program")
            return expr.to_fraction()
        if isinstance(expr, VarExpr):
            if expr.var in state:
                return state[expr.var]
            if expr.var in self.program.constants:
                return self.value(self.program.constants[expr.var], state)
            raise UnsupportedConstruct(f"the name {expr.var} (a parameter?)")
        if isinstance(expr, UnopExpr):
            inner = self.value(expr.expr, state)
            if expr.operator == Unop.NEG:
                return not self._bool(inner, expr)
            return -inner
        if isinstance(expr, BinopExpr):
            return self.binop(expr, state)
        if isinstance(expr, CategoricalExpr):
            return self.categorical(expr, state)
        if isinstance(expr, FunctionCallExpr):
            return self.call(expr, state)
        if isinstance(expr, InferExpr):
            function = self.program.functions.get(expr.function.function)
            if function is None:
                raise SimulationError(f"unknown function {expr.function.function}")
            arguments = function.params_to_dict(expr.function.params)
            return DistValue(expr.function.function,
                             tuple(sorted((var, self.value(arg, state)) for var, arg in arguments.items())))
        if isinstance(expr, SampleExpr):
            return self.sample(state[expr.dist])
        if isinstance(expr, (ProbabilityExpr, ExpectationExpr)):
            raise UnsupportedConstruct(f"{expr} inside a program")
        raise UnsupportedConstruct(f"the expression {expr}")

    @staticmethod
    def _bool(value: Any, expr: Expr) -> bool:
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise SimulationError(f"{expr} is not a boolean: {value}")

    def binop(self, expr: BinopExpr, state: State) -> Any:  # pylint: disable=too-many-return-statements
        """The value of a binary operation."""
        op = expr.operator
        if op == Binop.AND:
            return self._bool(self.value(expr.lhs, state), expr) and self._bool(self.value(expr.rhs, state), expr)
        if op == Binop.OR:
            return self._bool(self.value(expr.lhs, state), expr) or self._bool(self.value(expr.rhs, state), expr)
        lhs, rhs = self.value(expr.lhs, state), self.value(expr.rhs, state)
        if isinstance(lhs, DistValue) or isinstance(rhs, DistValue):
            raise UnsupportedConstruct(f"arithmetic on dist values in {expr}")
        lhs, rhs = Fraction(lhs), Fraction(rhs)
        comparisons = {Binop.LEQ: lhs <= rhs, Binop.LT: lhs < rhs, Binop.GT: lhs > rhs,
                       Binop.GEQ: lhs >= rhs, Binop.EQ: lhs == rhs} if op.returns_boolean() else {}
        if op in comparisons:
            return comparisons[op]
        if op == Binop.PLUS:
            return _normal(lhs + rhs)
        if op == Binop.MINUS:
            return _normal(max(lhs - rhs, Fraction(0)))
        if op == Binop.TIMES:
            return _normal(lhs * rhs)
        if op == Binop.POWER:
            if rhs.denominator != 1:
                raise UnsupportedConstruct(f"the non-integer power {expr}")
            if lhs == 0 and rhs < 0:
                raise SimulationError(f"division by zero in {expr}")
            return _normal(lhs ** int(rhs))
        if rhs == 0:
            raise SimulationError(f"division by zero in {expr}")
        if op == Binop.DIVIDE:
            return _normal(lhs / rhs)
        if op == Binop.FLOOR_DIVIDE:
            return _normal(Fraction(math.floor(lhs / rhs)))
        if op == Binop.MODULO:
            return _normal(lhs - rhs * math.floor(lhs / rhs))
        raise UnsupportedConstruct(f"the operator {op}")

    # -- sampling ----------------------------------------------------------------------------------

    def bernoulli(self, prob: Fraction) -> bool:
        """True with the exact probability ``prob``."""
        return self.rng.randrange(prob.denominator) < prob.numerator

    def categorical(self, expr: CategoricalExpr, state: State) -> Any:
        """A draw from a categorical expression."""
        draw = Fraction(self.rng.random())
        total = Fraction(0)
        for value, prob in expr.exprs:
            total += prob.to_fraction()
            if draw < total:
                return self.value(value, state)
        return self.value(expr.exprs[-1][0], state)

    def call(self, expr: FunctionCallExpr, state: State) -> Any:
        """A draw from a built-in distribution, or the value returned by a function."""
        if expr.function in self.program.functions:
            return self.run_function(expr.function, self.program.functions[expr.function].params_to_dict(expr.params),
                                     state)
        args = expr.params[0]
        if expr.function == "iid":
            count = _natural(self.value(args[1], state), f"the number of samples in {expr}")
            inner = args[0]
            if not isinstance(inner, FunctionCallExpr):
                raise UnsupportedConstruct(f"the distribution {inner} in iid")
            return sum(_natural(self.call(inner, state), "a sample") for _ in range(count))
        values = [self.value(arg, state) for arg in args]
        return self.draw(expr.function, values, expr)

    def draw(self, name: str, values: List[Any], expr: Expr) -> int:  # pylint: disable=too-many-return-statements
        """A draw from the built-in distribution ``name`` with the argument values ``values``."""
        if name in ("unif", "unif_d"):
            low, high = (_natural(v, f"a bound of {expr}") for v in values)
            if low > high:
                raise SimulationError(f"empty range in {expr}")
            return self.rng.randint(low, high)
        if name == "bernoulli":
            return int(self.bernoulli(_probability(values[0], f"the argument of {expr}")))
        if name == "binomial":
            trials = _natural(values[0], f"the number of trials in {expr}")
            prob = _probability(values[1], f"the probability in {expr}")
            return sum(self.bernoulli(prob) for _ in range(trials))
        if name == "geometric":
            return self.geometric(_probability(values[0], f"the argument of {expr}"), expr)
        if name == "negative_binomial":
            successes = _natural(values[0], f"the number of successes in {expr}")
            prob = _probability(values[1], f"the probability in {expr}")
            return sum(self.geometric(prob, expr) for _ in range(successes))
        if name == "poisson":
            return self.poisson(Fraction(values[0]), expr)
        if name == "logdist":
            return self.logarithmic(_probability(values[0], f"the argument of {expr}"), expr)
        raise UnsupportedConstruct(f"the distribution {name}")

    def geometric(self, prob: Fraction, expr: Expr) -> int:
        """The number of failures before the first success."""
        if prob == 0:
            raise SimulationError(f"{expr} with success probability 0")
        if prob == 1:
            return 0
        uniform = 1.0 - self.rng.random()  # in (0, 1]
        return int(math.floor(math.log(uniform) / math.log(1 - float(prob))))

    def poisson(self, lam: Fraction, expr: Expr) -> int:
        """A Poisson draw (Knuth's method)."""
        if lam < 0 or lam > 100:
            raise UnsupportedConstruct(f"{expr}: Poisson with mean {lam}")
        threshold = math.exp(-float(lam))
        count, product = 0, self.rng.random()
        while product > threshold:
            count += 1
            product *= self.rng.random()
        return count

    def logarithmic(self, prob: Fraction, expr: Expr) -> int:
        """A draw from the logarithmic distribution P(k) = -p^k / (k log(1 - p)), k >= 1."""
        if not 0 < prob < 1:
            raise SimulationError(f"{expr}: the parameter must lie in (0, 1)")
        p = float(prob)
        uniform = self.rng.random()
        k, mass = 1, -p / math.log(1 - p)
        while uniform > mass:
            uniform -= mass
            mass *= p * k / (k + 1)
            k += 1
            if mass == 0.0:
                break
        return k

    # -- functions and nested inference -----------------------------------------------------------------

    def run_function(self, name: str, arguments: Mapping[str, Expr], state: State) -> Any:
        """Runs the function ``name`` on the values of ``arguments`` and returns its value."""
        values = {var: self.value(arg, state) for var, arg in arguments.items()}
        return self._function_body(name, values)

    def _function_body(self, name: str, values: Mapping[str, Any]) -> Any:
        function = self.program.functions[name]
        local: State = dict(values)
        for var in function.variables:
            local.setdefault(var, 0)
        saved = self.program
        self.program = Program.from_function(function, saved, copy_functions=True)
        try:
            self.block(function.instructions, local)
            return self.value(function.returns, local)
        finally:
            self.program = saved

    def sample(self, dist: Any) -> Any:
        """A draw from the value of a ``dist`` variable, by rejection sampling."""
        if not isinstance(dist, DistValue):
            raise SimulationError(f"sample of a value that is not a distribution: {dist}")
        while True:
            self._tick()
            try:
                return self._function_body(dist.function, dict(dist.arguments))
            except _Rejected:
                continue
            except _Aborted:
                continue


def _normal(value: Fraction) -> Number:
    return int(value) if value.denominator == 1 else value


# ------------------------------------------------------------------------------------------------
# Queries and comparison with an analysis result
# ------------------------------------------------------------------------------------------------

def _evaluator(program: Program, expr: Expr) -> Callable[[Dict[str, Any]], Any]:
    machine = _Machine(program, random.Random(0), 0)

    def evaluate(state: Dict[str, Any]) -> Any:
        return machine.value(expr, dict(state))
    return evaluate


@dataclass
class Mismatch:
    """A number of the analysis that lies outside the confidence interval of the simulation."""
    what: str
    reported: Tuple[float, float]
    estimate: Estimate

    def __str__(self) -> str:
        low, high = self.reported
        value = f"{low}" if low == high else f"[{low}, {high}]"
        est = self.estimate
        return (f"{self.what}: analysis {value}, simulation {est.value:.6g} "
                f"(interval [{est.lower:.6g}, {est.upper:.6g}], {est.n} runs)")


def _reported(entry: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    low, high = entry.get("lower_float"), entry.get("upper_float")
    if low is None and high is None:
        value = entry.get("value_float")
        if value is None:
            return None
        return float(value), float(value)
    return (float(low) if low is not None else -math.inf, float(high) if high is not None else math.inf)


def compare(program: Program, result: Mapping[str, Any], simulation: Simulation, *, z: float = 5.0,
            marginal_rows: int = 12) -> List[Mismatch]:
    """
    Compares the JSON result of an analysis (``AnalysisResult.to_json()``, status ``ok``) with a
    simulation of the same program: the queries ``Pr[g]``, ``Ex[e]`` and ``Pr[x]``, the first
    ``marginal_rows`` rows of every marginal, and the mass of the output distribution. Every
    reported value or interval ``[lower, upper]`` must meet the simulation's confidence interval.
    Values the analysis reports as infinite or without a float (symbolic) are skipped.
    """
    mismatches: List[Mismatch] = []

    def check(what: str, reported: Optional[Tuple[float, float]], estimate: Estimate) -> None:
        if reported is None or any(math.isnan(r) for r in reported) or reported[0] == math.inf:
            return
        if not estimate.contains(*reported):
            mismatches.append(Mismatch(what, reported, estimate))

    kind = (result.get("bounds") or {}).get("kind", "exact")
    residual = float((result.get("bounds") or {}).get("residual_mass_float") or 0.0)
    for entry in result.get("queries", []):
        expr_text = entry.get("expr")
        query_kind = entry.get("kind")
        instr = _query_instruction(program, entry.get("index"))
        if instr is None:
            continue
        if query_kind == "probability" and isinstance(instr, ProbabilityQueryInstr):
            event = _evaluator(program, instr.expr)
            check(f"Pr[{expr_text}]", _reported(entry),
                  simulation.probability(_holds(event), z))
        elif query_kind == "expectation" and isinstance(instr, ExpectationInstr):
            value = _evaluator(program, instr.expr)
            check(f"Ex[{expr_text}]", _reported(entry), simulation.expectation(value, z))
    marginals = ((result.get("distribution") or {}).get("marginals") or {})
    for var, marginal in marginals.items():
        if var not in program.variables:
            continue
        for row in marginal.get("rows", [])[:marginal_rows]:
            prob = row.get("p")
            if prob is None:
                continue
            value = row["value"]
            low, high = float(prob), float(prob)
            if kind == "lower":
                high = low + residual
            elif kind == "upper":
                low = 0.0
            check(f"Pr[{var} == {value}]", (low, high),
                  simulation.probability(_holds(_evaluator(program, BinopExpr(Binop.EQ, VarExpr(var),
                                                                               NatLitExpr(value)))), z))
    mass = (result.get("distribution") or {}).get("mass_float")
    if mass is not None:
        low, high = float(mass), float(mass)
        if kind == "lower":
            high = low + residual
        elif kind == "upper":
            low = 0.0
        check("mass", (low, high), simulation.mass(z))
    return mismatches


def _holds(condition: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], bool]:
    return lambda state: bool(condition(state))


def _query_instruction(program: Program, index: Optional[int]) -> Optional[Instr]:
    if index is None:
        return None
    countable = [q for q in queries(program) if isinstance(q, (ProbabilityQueryInstr, ExpectationInstr,
                                                                 PrintInstr, PlotInstr, OptimizationQuery))]
    return countable[index] if 0 <= index < len(countable) else None
