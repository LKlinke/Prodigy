"""
Nested inference in the forward analysis: values of type ``dist``, ``infer`` and ``sample``.

The semantics is the one of ``docs/source/language-design.md`` ("Nested inference"):
``infer(f, a)`` is the distribution of the results of ``f(a)``, given that ``f(a)`` terminates and
passes its observations (the sub-distribution divided by its mass Z; Z = 0 is an error), and every
``sample(d)`` is an independent draw from ``d``.

Encoding
--------
A value of type ``dist`` is a *site*, :class:`DistSite`: a function and its arguments. The analysis
numbers the sites it meets (:class:`DistRegistry`, ids from 1), and a ``dist`` variable ``d`` is a
natural number in the joint generating function: the id of its site, 0 while it has no value.
An argument of a site is a natural number, another (closed) site, or a :class:`Slot`: the value
of a state-dependent argument, kept in the slot variables of ``d`` (``d_arg0__``, ...). A site
without slots is *closed*; its normalised generating function is computed once, exactly, when
``infer`` meets it (:meth:`DistRegistry.closed_form`).

The exact cases of the design are computed as follows:

- ``x = sample(d)``: split the distribution by the value of ``d`` (finitely many sites); for a
  closed site, marginalise ``x`` and multiply by G_d(x), as ``iid`` does. A site with slots exists
  only for a function without ``observe`` that terminates almost surely for every argument
  (Z ≡ 1, proven when ``infer`` creates it); then the draw is the call ``f(a)`` itself.
- ``d = infer(f, a)``: arguments without program variables are constants; arguments that depend
  on the state are split into their (finite) values, one closed site each; with infinite
  support they become slots, if Z ≡ 1 is proven, and are not implemented otherwise.
- ``d2 = d`` copies the id and the slots; a call ``g(d)`` passes them to the parameter.
- ``Pr[d == k]``, ``Ex[d]``: split by the value of ``d``, compute the number from G_d and
  substitute it into the instruction.
- A loop that reads ``dist`` variables but assigns none: split by their values, take them out of
  the distribution and bind them (:attr:`ProgramInfo.dist_bindings`) while the loop runs, so the
  loop strategies never see them. Assigning a ``dist`` variable inside a loop (a ``dist`` value
  in the loop's state) is not implemented.

Every other case ends with an :class:`~prodigy.analysis.exceptions.UnsupportedFeatureError` that
names the construct. At the queries and at the end of the program the ``dist`` variables are
taken out of the distribution; :func:`dist_values` describes their values.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Set, Tuple, Union)

import sympy
from frozendict import frozendict

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.exceptions import (UndefinedDistributionError,
                                         UnsupportedFeatureError)
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution, MarginalType
from prodigy.distribution.distribution import constant_value
from prodigy.pgcl import (AsgnInstr, Binop, BinopExpr, BoolLitExpr,
                          CategoricalExpr, Expr, ExprClass, Function,
                          FunctionCallExpr, Instr, LoopInstr, NatLitExpr, NatType,
                          ObserveInstr, Program, RealLitExpr, Unop, UnopExpr,
                          VarExpr, WhileInstr)
from prodigy.pgcl.ast.expressions import (ExpectationExpr, InferExpr,
                                          ProbabilityExpr, SampleExpr)
from prodigy.pgcl.ast.types import DistributionType
from prodigy.pgcl.ast.walk import (Walk, mut_expr_children, walk_expr,
                                   walk_instr_exprs, walk_instrs)
from prodigy.pgcl.util.ref import Mut

Runner = Callable[[Union[Instr, Sequence[Instr]], ProgramInfo, Distribution, Distribution,
                   ForwardAnalysisConfig], Tuple[Distribution, Distribution]]
"""Runs instructions (``compute_semantics`` or the untraced ``_semantics``)."""

VALUE_VARIABLE = "dist_value__"
"""The variable of the generating function of a closed site."""


@dataclass(frozen=True)
class Slot:
    """An argument of a site whose value is kept in the slot variable ``index`` of the ``dist``
    variable (see :func:`slot_variable`)."""
    index: int


@dataclass(frozen=True)
class DistSite:
    """The value ``infer(function, args)``: ``args`` gives every variable of the function its
    value (a natural number, a closed site for a ``dist`` parameter, or a :class:`Slot`)."""
    function: str
    args: Tuple[Tuple[str, Union[int, "DistSite", Slot]], ...]
    given: Tuple[str, ...] = field(default=(), compare=False)
    """The parameters the call names (for :meth:`describe`; the other variables are 0)."""

    @property
    def closed(self) -> bool:
        return not any(isinstance(value, Slot) for _, value in self.args)

    @property
    def slots(self) -> List[int]:
        return sorted(value.index for _, value in self.args if isinstance(value, Slot))

    def with_slots(self, values: Sequence[int]) -> "DistSite":
        """The closed site with the slots ``0, 1, …`` replaced by ``values``."""
        args = tuple((var, values[value.index] if isinstance(value, Slot) else value)
                     for var, value in self.args)
        return DistSite(self.function, args, self.given)

    def describe(self, syntax: Optional[str] = None) -> str:
        """``infer(f, a = 2, d = infer(g))`` in ``.ppy``, ``infer { f(a := 2) }`` in pGCL (the
        syntax of the running analysis by default); a slot is shown as ``…``."""
        if syntax is None:
            from prodigy.analysis.session import current_session  # pylint: disable=import-outside-toplevel
            syntax = current_session().syntax
        values = dict(self.args)
        assign = " = " if syntax == "ppy" else " := "
        arguments = []
        for var in self.given:
            value = values.get(var, 0)
            if isinstance(value, DistSite):
                text = value.describe(syntax)
            elif isinstance(value, Slot):
                text = "…"
            else:
                text = str(value)
            arguments.append(f"{var}{assign}{text}")
        if syntax == "ppy":
            return f"infer({', '.join([self.function] + arguments)})"
        return f"infer {{ {self.function}({', '.join(arguments)}) }}"


@dataclass
class ClosedDist:
    """The normalised distribution of a closed site."""
    site: DistSite
    distribution: Distribution
    """The generating function, in :data:`VALUE_VARIABLE`."""
    mass: Any
    """The normalising constant Z (a sympy number)."""
    values: Dict[str, Any] = field(default_factory=dict)
    """Cache of ``Pr``/``Ex`` values (sympy), by the text of the expression."""


class DistRegistry:
    """The sites one analysis has met, with their ids and closed forms."""

    def __init__(self) -> None:
        self._sites: List[DistSite] = []
        self._ids: Dict[DistSite, int] = {}
        self._closed: Dict[DistSite, ClosedDist] = {}
        self.functions: Dict[str, Function] = {}

    def site_id(self, site: DistSite) -> int:
        found = self._ids.get(site)
        if found is None:
            self._sites.append(site)
            found = self._ids[site] = len(self._sites)
        return found

    def site(self, site_id: int) -> Optional[DistSite]:
        """The site with ``site_id``, ``None`` for 0 (no value)."""
        if site_id == 0:
            return None
        if not 0 < site_id <= len(self._sites):
            raise UndefinedDistributionError(f"Unknown value of a dist variable (site {site_id}).")
        return self._sites[site_id - 1]

    def closed(self, site: DistSite) -> ClosedDist:
        return self._closed[site]

    def has_closed(self, site: DistSite) -> bool:
        return site in self._closed

    def store(self, closed: ClosedDist) -> None:
        self._closed[closed.site] = closed


_REGISTRY: contextvars.ContextVar[Optional[DistRegistry]] = \
    contextvars.ContextVar("prodigy_dist_registry", default=None)


def registry() -> DistRegistry:
    """The registry of the running analysis (a new one if none runs)."""
    found = _REGISTRY.get()
    if found is None:
        found = DistRegistry()
        _REGISTRY.set(found)
    return found


@contextlib.contextmanager
def analysis_registry() -> Iterator[DistRegistry]:
    """Gives an analysis its own registry, unless it runs inside one (e.g. a function)."""
    found = _REGISTRY.get()
    if found is not None:
        yield found
        return
    token = _REGISTRY.set(DistRegistry())
    try:
        yield _REGISTRY.get()  # type: ignore[misc]
    finally:
        _REGISTRY.reset(token)


# ---------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------

def slot_variable(var: str, index: int) -> str:
    """The name of slot ``index`` of the ``dist`` variable ``var``."""
    return f"{var}_arg{index}__"


def dist_variables(prog_info: ProgramInfo) -> Set[str]:
    """The ``dist`` variables of the program (or function) that is analysed."""
    return {var for var, typ in prog_info.program.variables.items()
            if isinstance(typ, DistributionType)}


def _slots_present(distribution: Distribution, var: str) -> List[str]:
    present = distribution.get_variables()
    result = []
    index = 0
    while slot_variable(var, index) in present:
        result.append(slot_variable(var, index))
        index += 1
    return result


def encoding_variables(distribution: Distribution, var: str) -> List[str]:
    """The variables of ``distribution`` that encode the ``dist`` variable ``var``."""
    found = [var] if var in distribution.get_variables() else []
    return found + _slots_present(distribution, var)


def _add_variables(distribution: Distribution, *variables: str) -> Distribution:
    missing = [var for var in variables if var not in distribution.get_variables()]
    if not missing:
        return distribution
    return distribution.set_variables(*distribution.get_variables(), *missing)


def _assign(distribution: Distribution, var: str, value: Union[int, str]) -> Distribution:
    distribution = _add_variables(distribution, var)
    rhs: Expr = NatLitExpr(value) if isinstance(value, int) else VarExpr(value)
    return distribution.update(BinopExpr(Binop.EQ, VarExpr(var), rhs))


def without(distribution: Distribution, variables: Sequence[str]) -> Distribution:
    """``distribution`` without ``variables`` (those it has)."""
    present = [var for var in variables if var in distribution.get_variables()]
    if not present:
        return distribution
    if not distribution.get_variables() - set(present):
        # nothing would be left; keep the mass in a distribution without variables
        return distribution.factory().from_expr(distribution.get_probability_mass())
    return distribution.marginal(*present, method=MarginalType.EXCLUDE)


def strip_dist_variables(distribution: Distribution, prog_info: ProgramInfo) -> Distribution:
    """The distribution of the program state without the encoding of the ``dist`` variables."""
    variables: List[str] = []
    for var in sorted(dist_variables(prog_info)):
        variables.extend(encoding_variables(distribution, var))
    return without(distribution, variables)


# ---------------------------------------------------------------------------------------
# Splitting the distribution by the values of dist variables
# ---------------------------------------------------------------------------------------

@dataclass
class DistPart:
    """The part of a distribution where a ``dist`` variable has one value."""
    distribution: Distribution
    site: Optional[DistSite]
    """``None``: the variable has no value; a site with slots: their values have infinite
    support."""


def _values(distribution: Distribution, variables: Sequence[str]) -> Optional[List[Tuple[int, ...]]]:
    """
    The values of ``variables`` in the support of ``distribution``, ``None`` for infinite support
    (``NonFiniteMarginalError`` if, in addition, their marginal is not a finite measure). The values
    come from the support (:meth:`Distribution.finite_support`), not the marginal, whose mass of a
    value can cancel to 0 on a signed measure (#256).
    """
    support = distribution.finite_support(*variables)
    if support is None:
        distribution.checked_marginal(*variables)
        return None
    values: List[Tuple[int, ...]] = []
    for coefficient, state in support:
        if _sym(coefficient) == 0:
            continue
        valuation = dict(state.items())
        value = tuple(int(valuation.get(var, 0)) for var in variables)
        if value not in values:
            values.append(value)
    return values


def _split(distribution: Distribution, variables: Sequence[str],
           values: List[Tuple[int, ...]]) -> Iterator[Tuple[Tuple[int, ...], Distribution]]:
    if len(values) == 1:
        yield values[0], distribution
        return
    for value in values:
        conditions = [BinopExpr(Binop.EQ, VarExpr(var), NatLitExpr(val))
                      for var, val in zip(variables, value)]
        yield value, distribution.filter(BinopExpr.reduce(Binop.AND, conditions))


def split_by_value(distribution: Distribution, var: str, prog_info: ProgramInfo) -> List[DistPart]:
    """Splits ``distribution`` by the value of the ``dist`` variable ``var``; slots with finite
    support are split into their values, too."""
    bound = prog_info.dist_bindings.get(var)
    if bound is not None:
        return [DistPart(distribution, bound)]
    if var not in distribution.get_variables():
        return [DistPart(distribution, None)]
    found = _values(distribution, [var])
    assert found is not None, "a dist variable has finitely many values"
    reg = registry()
    parts: List[DistPart] = []
    for (site_id,), part in _split(distribution, [var], found):
        site = reg.site(site_id)
        if site is None or site.closed:
            parts.append(DistPart(part, site))
            continue
        slots = [slot_variable(var, index) for index in site.slots]
        slot_values = _values(part, slots)
        if slot_values is None:
            parts.append(DistPart(part, site))
            continue
        for values, sub in _split(part, slots, slot_values):
            parts.append(DistPart(sub, site.with_slots(values)))
    if not parts:
        parts.append(DistPart(distribution, None))
    return parts


def split_by_values(distribution: Distribution, variables: Sequence[str],
                    prog_info: ProgramInfo) -> List[Tuple[Distribution, Dict[str, Optional[DistSite]]]]:
    """Splits by the values of several ``dist`` variables."""
    parts: List[Tuple[Distribution, Dict[str, Optional[DistSite]]]] = [(distribution, {})]
    for var in variables:
        refined = []
        for part, sites in parts:
            for sub in split_by_value(part, var, prog_info):
                refined.append((sub.distribution, {**sites, var: sub.site}))
        parts = refined
    return parts


def _sum(parts: List[Tuple[Distribution, Distribution]], distribution: Distribution,
         error_prob: Distribution) -> Tuple[Distribution, Distribution]:
    """The sum of the results of the parts of ``distribution``; their error probabilities are
    added to ``error_prob``."""
    if not parts:
        return distribution * "0", error_prob
    result, error = parts[0]
    error = error_prob + error
    for other, other_error in parts[1:]:
        variables = result.get_variables() | other.get_variables()
        result = result.set_variables(*variables) + other.set_variables(*variables)
        error = error + other_error
    return result, error


def _spell(node: Any) -> str:
    """``node`` in the syntax of the program."""
    from prodigy.analysis.session import current_session  # pylint: disable=import-outside-toplevel
    return current_session().spell(node)


def _unassigned(var: str, construct: str) -> UndefinedDistributionError:
    return UndefinedDistributionError(
        f"{construct}: the dist variable {var} has no value here; assign it first, "
        f"with {var} = infer(f, …).")


# ---------------------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------------------

def _sym(value: Any) -> Any:
    from prodigy.analysis.result import sym  # pylint: disable=import-outside-toplevel
    return sym(value)


def function_program_info(prog_info: ProgramInfo, function: Function,
                          bindings: Optional[Dict[str, DistSite]] = None) -> ProgramInfo:
    """The program info for running the body of ``function``."""
    return replace(prog_info, program=Program.from_function(function, prog_info.program),
                   dist_bindings=frozendict(bindings or {}))


def _is_plain(expr: Expr) -> bool:
    """Whether ``expr`` is an expression that ``Distribution.update`` evaluates (no draws)."""
    return not any(isinstance(ref.val, (SampleExpr, InferExpr, FunctionCallExpr, CategoricalExpr))
                   for ref in walk_expr(Walk.DOWN, Mut.alloc(expr)))


def assign_return_value(distribution: Distribution, function: Function, var: str,
                        prog_info: ProgramInfo, config: ForwardAnalysisConfig, run: Runner,
                        error_prob: Distribution) -> Tuple[Distribution, Distribution]:
    """Assigns the return value of ``function`` to the new variable ``var``."""
    return _assign_value(_add_variables(distribution, var), var, function.returns, prog_info, config,
                         run, error_prob)


def _assign_value(distribution: Distribution, var: str, expr: Expr, prog_info: ProgramInfo,
                  config: ForwardAnalysisConfig, run: Runner,
                  error_prob: Distribution) -> Tuple[Distribution, Distribution]:
    """``var = expr`` for a variable ``var`` of type ``nat`` that the program does not declare.
    A plain expression is an update; a draw (``sample(d)``, ``categorical(…)``, a call) runs
    like an assignment in the program."""
    if _is_plain(expr):
        return distribution.update(BinopExpr(Binop.EQ, VarExpr(var), expr)), error_prob
    program = copy.copy(prog_info.program)
    program.variables = {**program.variables, var: NatType(None)}
    return run(AsgnInstr(var, expr), replace(prog_info, program=program), distribution, error_prob,
               config)


def _has_observe(function: Function) -> bool:
    return any(isinstance(ref.val, ObserveInstr) for ref in walk_instrs(Walk.DOWN, function.instructions))


def closed_form(site: DistSite, prog_info: ProgramInfo, config: ForwardAnalysisConfig,
                run: Runner) -> ClosedDist:
    """The normalised distribution of the closed ``site``, computed exactly (once)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.loop_strategies import Bound
    from prodigy.analysis.session import current_session
    reg = registry()
    if reg.has_closed(site):
        return reg.closed(site)
    function = reg.functions[site.function]
    factory = config.factory
    start = factory.one(*function.variables)
    for var, value in site.args:
        if isinstance(value, DistSite):
            closed_form(value, prog_info, config, run)
            value = reg.site_id(value)
        assert isinstance(value, int)
        start = _assign(start, var, value)
    sub_info = function_program_info(prog_info, function)
    session = current_session()
    zero = factory.from_expr("0")
    with session.nested_inference(site.describe()) as scope:
        result, error = run(function.instructions, sub_info, start, zero, config)
        result, error = assign_return_value(result, function, VALUE_VARIABLE, sub_info, config, run,
                                            error)
    if scope.kind != Bound.EXACT:
        raise UnsupportedFeatureError(
            f"{site.describe()}: the analysis of {site.function} gives only a bound ({scope.kind}), "
            "not its exact distribution, so it cannot be normalised. Normalising an approximate "
            "distribution is not implemented.")
    result = result.marginal(VALUE_VARIABLE)
    mass = sympy.nsimplify(_sym(result.get_probability_mass()))
    if mass.free_symbols:
        raise UnsupportedFeatureError(
            f"{site.describe()}: the normalising constant {mass} depends on parameters; proving "
            "that it is positive is not implemented.")
    if mass == 0:
        raise UndefinedDistributionError(
            f"{site.describe()} is undefined: the probability that {site.function} terminates and "
            "passes its observations is 0 (normalising constant Z = 0).")
    if mass.is_positive is not True:
        raise UndefinedDistributionError(f"{site.describe()}: the normalising constant {mass} is not positive.")
    distribution = result if mass == 1 else result * str(sympy.Integer(1) / mass)
    closed = ClosedDist(site, distribution, mass)
    reg.store(closed)
    return closed


def _constant_argument(expr: Expr) -> Optional[int]:
    """The value of an argument without variables (``None`` if it has variables)."""
    if isinstance(expr, NatLitExpr):
        return expr.value
    if isinstance(expr, BoolLitExpr):
        return int(expr.value)
    value = constant_value(expr) if isinstance(expr, BinopExpr) else None
    if value is None:
        return None
    if value.denominator != 1 or value < 0:
        raise UnsupportedFeatureError(f"The argument {_spell(expr)} is not a natural number.")
    return value.numerator


def _depends_on_state(expr: Expr, prog_info: ProgramInfo) -> bool:
    names = {ref.val.var for ref in walk_expr(Walk.DOWN, Mut.alloc(expr)) if isinstance(ref.val, VarExpr)}
    if names & set(prog_info.program.parameters):
        raise UnsupportedFeatureError(
            f"An argument of infer that depends on a parameter ({_spell(expr)}) is not implemented.")
    return bool(names)


def infer(instr: AsgnInstr, prog_info: ProgramInfo, distribution: Distribution,
          error_prob: Distribution, config: ForwardAnalysisConfig, run: Runner
          ) -> Tuple[Distribution, Distribution]:
    """``d = infer(f, args…)``."""
    assert isinstance(instr.rhs, InferExpr)
    call = instr.rhs.function
    if call.function not in prog_info.program.functions:
        raise UnsupportedFeatureError(
            f"{_spell(instr.rhs)}: infer inside a function is not implemented (functions cannot call "
            "functions yet).")
    function = prog_info.program.functions[call.function]
    reg = registry()
    reg.functions[call.function] = function
    given = tuple(decl.var for decl in function.declarations[:len(call.params[0])]) + \
        tuple(call.params[1].keys())
    assignments = function.params_to_dict(call.params)
    types = {decl.var: decl.typ for decl in function.declarations}

    # dist arguments: split by their values (they must be closed)
    dist_args = {var: expr for var, expr in assignments.items()
                 if isinstance(types.get(var), DistributionType)}
    dist_arg_vars = []
    for var, expr in dist_args.items():
        if isinstance(expr, NatLitExpr) and expr.value == 0:
            continue  # a dist variable of f without argument: no value
        if not isinstance(expr, VarExpr):
            raise UnsupportedFeatureError(
                f"The dist argument {_spell(expr)} of {_spell(instr.rhs)} must be a variable.")
        dist_arg_vars.append(expr.var)

    results: List[Tuple[Distribution, Distribution]] = []
    for part, sites in split_by_values(distribution, sorted(set(dist_arg_vars)), prog_info):
        fixed: Dict[str, Union[int, DistSite, Slot]] = {}
        for var, expr in dist_args.items():
            if isinstance(expr, VarExpr):
                site = sites[expr.var]
                if site is None:
                    raise _unassigned(expr.var, _spell(instr.rhs))
                if not site.closed:
                    raise UnsupportedFeatureError(
                        f"{_spell(instr.rhs)}: the dist argument {expr.var} depends on the state with "
                        "infinite support; this is not implemented.")
                fixed[var] = site
            else:
                fixed[var] = 0
        results.append(_infer_nat_arguments(instr, function, given, assignments, fixed, prog_info,
                                            part, config, run))
    return _sum(results, distribution, error_prob)


def _infer_nat_arguments(instr: AsgnInstr, function: Function, given: Tuple[str, ...],
                         assignments: Dict[str, Expr], fixed: Dict[str, Union[int, DistSite, Slot]],
                         prog_info: ProgramInfo, distribution: Distribution,
                         config: ForwardAnalysisConfig, run: Runner) -> Tuple[Distribution, Distribution]:
    target = instr.lhs
    zero = distribution * "0"
    variable_args: Dict[str, Expr] = {}
    for var, expr in assignments.items():
        if var in fixed:
            continue
        value = None if _depends_on_state(expr, prog_info) else _constant_argument(expr)
        if value is None:
            variable_args[var] = expr
        else:
            fixed[var] = value
    # the encoding of the old value of the target goes; the new one is set below
    old = encoding_variables(distribution, target)

    name = function_name(instr)

    def make_site(values: Dict[str, Union[int, DistSite, Slot]]) -> DistSite:
        return DistSite(name, tuple((decl.var, values[decl.var]) for decl in function.declarations),
                        given)

    if not variable_args:
        site = make_site(fixed)
        closed_form(site, prog_info, config, run)
        return _set_site(without(distribution, old[1:]), target, registry().site_id(site)), zero

    # evaluate the state-dependent arguments into fresh variables
    temps: Dict[str, str] = {}
    extended = distribution
    for var, expr in variable_args.items():
        temp = extended.get_fresh_variable()
        extended = _add_variables(extended, temp).update(BinopExpr(Binop.EQ, VarExpr(temp), expr))
        temps[var] = temp
    temp_vars = list(temps.values())
    values = _values(extended, temp_vars)
    if values is not None:
        parts = []
        for arg_values, part in _split(extended, temp_vars, values):
            site = make_site({**fixed, **dict(zip(temps.keys(), arg_values))})
            if _sym(part.get_probability_mass()) == 0:
                continue
            closed_form(site, prog_info, config, run)
            part = _set_site(without(part, old[1:] + temp_vars), target, registry().site_id(site))
            parts.append((part, zero))
        return _sum(parts, distribution, zero)

    # infinite support: only if Z ≡ 1, then sample(d) is the call f(a)
    describe = _spell(instr.rhs)
    if _has_observe(function):
        raise UnsupportedFeatureError(
            f"{describe}: the argument {', '.join(_spell(e) for e in variable_args.values())} depends "
            "on the state with infinite support, and the normalising constant of "
            f"{function_name(instr)} may depend on it (it has observe). Not implemented.")
    slot_site = make_site({**fixed, **{var: Slot(index) for index, var in enumerate(temps)}})
    _prove_almost_sure_termination(slot_site, extended.marginal(*temp_vars), temps, prog_info,
                                   config, run, describe)
    result = without(extended, old[1:])
    for index, temp in enumerate(temp_vars):
        result = _assign(result, slot_variable(target, index), temp)
    result = without(result, temp_vars)
    return _set_site(result, target, registry().site_id(slot_site)), zero


def function_name(instr: AsgnInstr) -> str:
    assert isinstance(instr.rhs, InferExpr)
    return instr.rhs.function.function


def _set_site(distribution: Distribution, var: str, site_id: int) -> Distribution:
    return _assign(distribution, var, site_id)


def _prove_almost_sure_termination(site: DistSite, arguments: Distribution, temps: Dict[str, str],
                                   prog_info: ProgramInfo, config: ForwardAnalysisConfig,
                                   run: Runner, describe: str) -> None:
    """Proves Z(a) = 1 for every value a of the arguments: running the function on the
    distribution of the arguments keeps the mass of every argument value."""
    from prodigy.analysis.loop_strategies import Bound  # pylint: disable=import-outside-toplevel
    from prodigy.analysis.session import current_session  # pylint: disable=import-outside-toplevel
    reg = registry()
    function = reg.functions[site.function]
    assignments: Dict[str, Expr] = {}
    for var, value in site.args:
        if isinstance(value, Slot):
            assignments[var] = VarExpr(list(temps.values())[value.index])
        elif isinstance(value, DistSite):
            assignments[var] = NatLitExpr(reg.site_id(value))
        else:
            assignments[var] = NatLitExpr(value)
    result_var = arguments.get_fresh_variable()
    with current_session().nested_inference(site.describe()) as scope:
        result, _ = call_in_joint(function, assignments, {}, result_var, prog_info,
                                  _add_variables(arguments, result_var), arguments * "0", config, run)
    proven = scope.kind == Bound.EXACT and result.marginal(*temps.values()) == arguments
    if not proven:
        raise UnsupportedFeatureError(
            f"{describe}: the argument depends on the state with infinite support; this is only "
            f"implemented when {site.function} terminates almost surely for every argument (then "
            "Z = 1), which could not be proven.")


# ---------------------------------------------------------------------------------------
# Function calls on the joint distribution
# ---------------------------------------------------------------------------------------

def call_in_joint(function: Function, assignments: Dict[str, Expr],
                  bindings: Dict[str, DistSite], lhs: str, prog_info: ProgramInfo,
                  distribution: Distribution, error_prob: Distribution,
                  config: ForwardAnalysisConfig, run: Runner) -> Tuple[Distribution, Distribution]:
    """``lhs = function(args)`` with arguments that depend on the state: the body runs on the
    joint distribution, so the result keeps its dependence on the caller's variables.
    ``assignments`` gives every variable of the function its initial value (an expression in
    the caller's variables; a ``dist`` parameter gets a variable of the caller, whose slots are
    copied as well)."""
    types = {decl.var: decl.typ for decl in function.declarations}
    dist = distribution
    # 1. the arguments, evaluated in the caller's state
    temps: Dict[str, str] = {}
    for var, expr in assignments.items():
        if var in bindings:
            continue
        sources = [str(expr.var)] if isinstance(types.get(var), DistributionType) \
            and isinstance(expr, VarExpr) else []
        if sources:
            sources += _slots_present(dist, sources[0])
            for index, source in enumerate(sources):
                temp = dist.get_fresh_variable()
                dist = _assign(dist, temp, source) if source in dist.get_variables() \
                    else _assign(dist, temp, 0)
                temps[var if index == 0 else slot_variable(var, index - 1)] = temp
            continue
        temp = dist.get_fresh_variable()
        dist, error_prob = _assign_value(_add_variables(dist, temp), temp, expr, prog_info, config,
                                         run, error_prob)
        temps[var] = temp
    # 2. caller variables with the names of the function's variables are kept aside
    own = set(function.variables) | {name for name in temps if name not in function.variables}
    saved: Dict[str, str] = {}
    for var in sorted(own & dist.get_variables()):
        keep = dist.get_fresh_variable()
        dist = _assign(dist, keep, var)
        saved[var] = keep
    # 3. the function's variables start at their arguments (0 for the others)
    dist = _add_variables(dist, *sorted(own))
    for var in sorted(own):
        dist = _assign(dist, var, temps[var]) if var in temps else _assign(dist, var, 0)
    dist = without(dist, list(temps.values()))
    # 4. the body, then the return value
    sub_info = function_program_info(prog_info, function, bindings)
    zero = error_prob * "0"
    result, error = run(function.instructions, sub_info, dist, zero, config)
    value = result.get_fresh_variable()
    result, error = assign_return_value(result, function, value, sub_info, config, run, error)
    # 5. back to the caller's variables
    leftovers = [var for var in result.get_variables() if var in own]
    for name in function.variables:
        leftovers.extend(_slots_present(result, name))
    result = without(result, leftovers)
    for var, keep in saved.items():
        result = _assign(result, var, keep)
    result = without(result, list(saved.values()))
    result = _assign(result, lhs, value)
    result = without(result, [value])
    return result, error_prob + error


# ---------------------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------------------

def _rename_value(closed: ClosedDist, var: str) -> Distribution:
    """The generating function of ``closed`` in the variable ``var``."""
    source = closed.distribution
    return source.set_variables(VALUE_VARIABLE, var).update(
        BinopExpr(Binop.EQ, VarExpr(var), VarExpr(VALUE_VARIABLE))).marginal(var)


def sample(instr: AsgnInstr, prog_info: ProgramInfo, distribution: Distribution,
           error_prob: Distribution, config: ForwardAnalysisConfig, run: Runner
           ) -> Tuple[Distribution, Distribution]:
    """``x = sample(d)``: an independent draw from the value of ``d``."""
    assert isinstance(instr.rhs, SampleExpr)
    var = instr.rhs.dist
    target = instr.lhs
    reg = registry()
    results: List[Tuple[Distribution, Distribution]] = []
    for part in split_by_value(distribution, var, prog_info):
        site = part.site
        if site is None:
            if _sym(part.distribution.get_probability_mass()) == 0:
                continue
            raise _unassigned(var, _spell(instr.rhs))
        if site.closed:
            closed = closed_form(site, prog_info, config, run)
            marginal = without(part.distribution, [target])
            results.append((marginal * _rename_value(closed, target), error_prob * "0"))
            continue
        # Z ≡ 1 was proven: the draw is the call f(a), with the arguments from the slots
        function = reg.functions[site.function]
        assignments: Dict[str, Expr] = {}
        for name, value in site.args:
            if isinstance(value, Slot):
                assignments[name] = VarExpr(slot_variable(var, value.index))
            elif isinstance(value, DistSite):
                assignments[name] = NatLitExpr(reg.site_id(value))
            else:
                assignments[name] = NatLitExpr(value)
        results.append(call_in_joint(function, assignments, {}, target, prog_info, part.distribution,
                                     error_prob * "0", config, run))
    return _sum(results, distribution, error_prob)


def copy_dist(instr: AsgnInstr, prog_info: ProgramInfo, distribution: Distribution,
              error_prob: Distribution) -> Tuple[Distribution, Distribution]:
    """``d2 = d``: the copy has the same value (independent draws from both)."""
    assert isinstance(instr.rhs, VarExpr)
    source, target = instr.rhs.var, instr.lhs
    if source == target:
        return distribution, error_prob
    bound = prog_info.dist_bindings.get(source)
    result = without(distribution, encoding_variables(distribution, target)[1:])
    if bound is not None:
        return _set_site(result, target, registry().site_id(bound)), error_prob
    if source not in result.get_variables():
        return _set_site(result, target, 0), error_prob
    result = _assign(result, target, source)
    for index, slot in enumerate(_slots_present(result, source)):
        result = _assign(result, slot_variable(target, index), slot)
    return result, error_prob


# ---------------------------------------------------------------------------------------
# Pr[d …] and Ex[d]
# ---------------------------------------------------------------------------------------

def _dist_queries(instr: Instr) -> List[Union[ProbabilityExpr, ExpectationExpr]]:
    """The ``Pr[d …]`` and ``Ex[d]`` in the instruction's own expressions (not in its blocks)."""
    found: List[Union[ProbabilityExpr, ExpectationExpr]] = []
    for value in instr.__dict__.values():
        if isinstance(value, ExprClass):
            for ref in walk_expr(Walk.DOWN, Mut.alloc(value.cast())):
                if isinstance(ref.val, (ProbabilityExpr, ExpectationExpr)):
                    found.append(ref.val)
    return found


def _queried_variable(node: Union[ProbabilityExpr, ExpectationExpr], dist_vars: Set[str]) -> str:
    """The ``dist`` variable of ``Pr[d …]`` or ``Ex[d]`` (the type checker ensures the form)."""
    if isinstance(node, ExpectationExpr):
        assert isinstance(node.expr, VarExpr)
        return node.expr.var
    cond = node.cond
    assert isinstance(cond, BinopExpr)
    if isinstance(cond.lhs, VarExpr) and cond.lhs.var in dist_vars:
        return cond.lhs.var
    assert isinstance(cond.rhs, VarExpr)
    return cond.rhs.var


def _constant_argument_or_none(expr: Expr) -> Optional[int]:
    try:
        return _constant_argument(expr)
    except UnsupportedFeatureError:
        return None


def query_value(node: Union[ProbabilityExpr, ExpectationExpr], var: str,
                site: DistSite, prog_info: ProgramInfo, config: ForwardAnalysisConfig,
                run: Runner) -> Any:
    """The value of ``Pr[d …]`` or ``Ex[d]`` for the closed value ``site`` of ``d`` (sympy)."""
    closed = closed_form(site, prog_info, config, run)
    key = str(node)
    if key in closed.values:
        return closed.values[key]
    value_var = VarExpr(VALUE_VARIABLE)
    try:
        try:
            value = _engine_value(node, var, closed, value_var)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            if not isinstance(node, ExpectationExpr):
                raise
            # e.g. GiNaC cannot take the limit of G'(z) at 1 if Ex[d] = ∞; SymPy can
            function = _sym(str(closed.distribution))
            points = [symbol for symbol in function.free_symbols if str(symbol) == VALUE_VARIABLE]
            if len(points) != 1 or len(function.free_symbols) != 1:
                raise
            value = sympy.limit(sympy.diff(function, points[0]), points[0], 1, "-")
    except UnsupportedFeatureError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise UnsupportedFeatureError(
            f"{_spell(node)} for {site.describe()} could not be computed exactly: {exc}") from exc
    value = sympy.nsimplify(value) if not value.has(sympy.oo) else value
    closed.values[key] = value
    return value


def _engine_value(node: Union[ProbabilityExpr, ExpectationExpr], var: str, closed: ClosedDist,
                  value_var: VarExpr) -> Any:
    if isinstance(node, ExpectationExpr):
        text = str(closed.distribution.get_expected_value_of(value_var)).strip()
        if text.lower() in ("infinity", "oo", "inf", "∞"):
            return sympy.oo
        value = _sym(text)
    else:
        cond = node.cond
        assert isinstance(cond, BinopExpr)
        dist_left = isinstance(cond.lhs, VarExpr) and cond.lhs.var == var
        other = cond.rhs if dist_left else cond.lhs
        number = _constant_argument_or_none(other)
        if number is None:
            raise UnsupportedFeatureError(
                f"{_spell(node)}: comparing a dist value with a number that depends on the program state "
                "is not implemented.")
        rewritten = BinopExpr(cond.operator, value_var, NatLitExpr(number)) if dist_left \
            else BinopExpr(cond.operator, NatLitExpr(number), value_var)
        value = _sym(closed.distribution.get_probability_of(rewritten))
    if value.free_symbols or value.has(sympy.zoo, sympy.nan):
        raise UnsupportedFeatureError(
            f"{_spell(node)} = {value} is not a number; Pr and Ex that depend on parameters are "
            "not implemented inside a program.")
    return value


def _to_sympy(expr: Expr, values: Dict[int, Any]) -> Optional[Any]:
    """``expr`` as a sympy value if it has no variables (with the values of ``Pr``/``Ex``)."""
    # pylint: disable=too-many-return-statements
    if id(expr) in values:
        return values[id(expr)]
    if isinstance(expr, NatLitExpr):
        return sympy.Integer(expr.value)
    if isinstance(expr, RealLitExpr):
        return sympy.oo if expr.is_infinite() else sympy.Rational(expr.to_fraction())
    if isinstance(expr, BoolLitExpr):
        return sympy.true if expr.value else sympy.false
    if isinstance(expr, UnopExpr) and expr.operator == Unop.NEG:
        inner = _to_sympy(expr.expr, values)
        return None if inner is None else sympy.Not(inner)
    if not isinstance(expr, BinopExpr):
        return None
    lhs, rhs = _to_sympy(expr.lhs, values), _to_sympy(expr.rhs, values)
    if lhs is None or rhs is None:
        return None
    operations = {
        Binop.PLUS: lambda a, b: a + b, Binop.MINUS: lambda a, b: a - b,
        Binop.TIMES: lambda a, b: a * b, Binop.DIVIDE: lambda a, b: a / b,
        Binop.POWER: lambda a, b: a ** b,
        Binop.EQ: sympy.Eq, Binop.LT: sympy.Lt, Binop.LEQ: sympy.Le,
        Binop.GT: sympy.Gt, Binop.GEQ: sympy.Ge,
        Binop.AND: sympy.And, Binop.OR: sympy.Or,
    }
    operation = operations.get(expr.operator)
    return None if operation is None else operation(lhs, rhs)


def _from_sympy(value: Any, where: Expr) -> Expr:
    if value is sympy.true or value is sympy.false:
        return BoolLitExpr(bool(value))
    if isinstance(value, (sympy.logic.boolalg.Boolean, sympy.core.relational.Relational)):
        raise UnsupportedFeatureError(f"{_spell(where)}: the comparison {value} cannot be decided exactly.")
    if value == sympy.oo:
        return RealLitExpr.infinity()
    if value.is_Rational:
        return RealLitExpr(Fraction(int(value.p), int(value.q)))
    if value is sympy.nan or value.has(sympy.nan, sympy.zoo):
        raise UnsupportedFeatureError(f"{_spell(where)} is undefined (nan).")
    raise UnsupportedFeatureError(
        f"{_spell(where)} = {value} is not a rational number; using it as a probability or in arithmetic "
        "of the program is not implemented (comparisons work).")


def _fold(expr: Expr, values: Dict[int, Any]) -> Expr:
    """Replaces the subexpressions with a resolved ``Pr``/``Ex`` by their value."""
    def contains_value(node: Expr) -> bool:
        return any(id(ref.val) in values for ref in walk_expr(Walk.DOWN, Mut.alloc(node)))

    root = Mut.alloc(expr)

    def visit(ref: Mut) -> None:
        node = ref.val
        if not contains_value(node):
            return
        value = _to_sympy(node, values)
        if value is not None:
            ref.val = _from_sympy(value, node)
            return
        for child in mut_expr_children(ref):
            visit(child)

    visit(root)
    return root.val


def resolve_queries(instr: Instr, prog_info: ProgramInfo, distribution: Distribution,
                    error_prob: Distribution, config: ForwardAnalysisConfig, run: Runner,
                    run_shell: Runner) -> Optional[Tuple[Distribution, Distribution]]:
    """Runs ``instr`` with its ``Pr[d …]`` and ``Ex[d]`` replaced by their values, split by the
    values of the ``dist`` variables they read; ``None`` if it has none."""
    nodes = _dist_queries(instr)
    if not nodes:
        return None
    dist_vars = dist_variables(prog_info)
    variables = sorted({_queried_variable(node, dist_vars) for node in nodes})
    results: List[Tuple[Distribution, Distribution]] = []
    for part, sites in split_by_values(distribution, variables, prog_info):
        values: Dict[int, Any] = {}
        for node in nodes:
            var = _queried_variable(node, dist_vars)
            site = sites[var]
            if site is None:
                if _sym(part.get_probability_mass()) == 0:
                    break
                raise _unassigned(var, _spell(node))
            if not site.closed:
                raise UnsupportedFeatureError(
                    f"{_spell(node)}: the value of {var} depends on the program state with infinite support "
                    f"({site.describe()}); Pr and Ex of such a dist are not implemented.")
            values[id(node)] = query_value(node, var, site, prog_info, config, run)
        else:
            shell = copy.copy(instr)
            for name, value in list(instr.__dict__.items()):
                if isinstance(value, ExprClass):
                    setattr(shell, name, _fold(_copy_keeping_ids(value.cast(), values), values))
            results.append(_run_shell(shell, instr, prog_info, part, error_prob * "0", config, run_shell))
    return _sum(results, distribution, error_prob)


def _copy_keeping_ids(expr: Expr, values: Dict[int, Any]) -> Expr:
    """A copy of ``expr`` whose resolved nodes keep their identity (they are replaced anyway)."""
    memo = {id(node): node for node in _nodes(expr) if id(node) in values}
    return copy.deepcopy(expr, memo)


def _nodes(expr: Expr) -> List[Expr]:
    return [ref.val for ref in walk_expr(Walk.DOWN, Mut.alloc(expr))]


def _run_shell(shell: Instr, original: Instr, prog_info: ProgramInfo, distribution: Distribution,
               error_prob: Distribution, config: ForwardAnalysisConfig, run: Runner
               ) -> Tuple[Distribution, Distribution]:
    from prodigy.analysis.session import current_session  # pylint: disable=import-outside-toplevel
    if isinstance(shell, WhileInstr) and isinstance(original, WhileInstr):
        with current_session().loop_alias(shell, original):
            return run(shell, prog_info, distribution, error_prob, config)
    return run(shell, prog_info, distribution, error_prob, config)


# ---------------------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------------------

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


def run_loop(instr: Union[WhileInstr, LoopInstr], prog_info: ProgramInfo,
             distribution: Distribution, error_prob: Distribution, config: ForwardAnalysisConfig,
             run: Runner) -> Optional[Tuple[Distribution, Distribution]]:
    """Runs a loop with the ``dist`` variables bound to their values (see the module
    documentation); ``None`` if there is nothing to bind."""
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
    for part, sites in split_by_values(distribution, present, prog_info):
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
                restore[var] = registry().site_id(site)
        inner = replace(prog_info, dist_bindings=frozendict(bindings))
        out, error = run(instr, inner, without(part, removed), error_prob * "0", config)
        for var, site_id in restore.items():
            out = _set_site(out, var, site_id)
        results.append((out, error))
    return _sum(results, distribution, error_prob)


# ---------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------

@dataclass
class DistValue:
    """One value of a ``dist`` variable at the end of the program."""
    variable: str
    probability: Any
    """The probability of this value (sympy); 1 if the value does not depend on the state."""
    site: Optional[DistSite]
    closed: Optional[ClosedDist]


def dist_values(distribution: Distribution, prog_info: ProgramInfo) -> List[DistValue]:
    """The values of the ``dist`` variables in ``distribution``."""
    found: List[DistValue] = []
    reg = registry()
    for var in sorted(dist_variables(prog_info)):
        for part in split_by_value(distribution, var, prog_info):
            probability = sympy.nsimplify(_sym(part.distribution.get_probability_mass()))
            if probability == 0:
                continue
            closed = reg.closed(part.site) if part.site is not None and reg.has_closed(part.site) else None
            found.append(DistValue(var, probability, part.site, closed))
    return found


def _value_json(value: DistValue, syntax: str, limit: int) -> Dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import exact_and_float, to_latex, univariate_table
    probability, probability_float = exact_and_float(value.probability)
    site = value.site
    entry: Dict[str, Any] = {
        "probability": probability,
        "probability_float": probability_float,
        "infer": site.describe(syntax) if site is not None else None,
        "function": site.function if site is not None else None,
        "arguments": {} if site is None else {
            var: (arg.describe(syntax) if isinstance(arg, DistSite) else
                  None if isinstance(arg, Slot) else str(arg))
            for var, arg in site.args if var in site.given},
        "closed": value.closed is not None,
        "closed_form": None, "latex": None, "normalizing_constant": None,
        "normalizing_constant_float": None, "mean": None, "mean_float": None,
        "finite_support": None, "table": None,
    }
    if site is None:
        entry["note"] = "no value: the variable was not assigned"
        return entry
    if value.closed is None:
        entry["note"] = (f"the arguments depend on the state with infinite support; every draw is "
                         f"the call {site.function}(…) (Z = 1 is proven)")
        return entry
    renamed = _rename_value(value.closed, value.variable)
    entry["closed_form"] = str(renamed)
    entry["latex"] = to_latex(renamed)
    entry["normalizing_constant"], entry["normalizing_constant_float"] = exact_and_float(value.closed.mass)
    try:
        mean = renamed.get_expected_value_of(VarExpr(value.variable))
        entry["mean"], entry["mean_float"] = exact_and_float(
            "oo" if str(mean).strip().lower() in ("infinity", "oo", "inf") else mean)
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        entry["finite_support"] = bool(renamed.is_finite())
    except Exception:  # pylint: disable=broad-except
        pass
    if limit > 0:
        try:
            entry["table"] = univariate_table(renamed, value.variable, limit)
        except Exception as exc:  # pylint: disable=broad-except
            entry["table"] = {"variable": value.variable, "error": str(exc), "rows": []}
    return entry


def dist_values_json(values: List[DistValue], syntax: str, limit: int) -> List[Dict[str, Any]]:
    """The ``distributions`` section of the result JSON (see :mod:`prodigy.analysis.result`)."""
    result: List[Dict[str, Any]] = []
    by_variable: Dict[str, List[DistValue]] = {}
    for value in values:
        by_variable.setdefault(value.variable, []).append(value)
    for variable, found in by_variable.items():
        entries = [_value_json(value, syntax, limit) for value in found]
        result.append({
            "variable": variable,
            "depends_on_state": len(found) > 1 or any(value.site is not None and value.closed is None
                                                      for value in found),
            "values": entries,
        })
    return result


def for_trace(distribution: Distribution, program: Optional[Program], syntax: str
              ) -> Tuple[Distribution, List[Dict[str, Any]]]:
    """A step of the trace shows the program state without the encoding of the ``dist``
    variables, and their values next to it (variable, probability, ``infer`` text, closed form and
    its LaTeX)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.result import to_latex
    if program is None:
        return distribution, []
    prog_info = ProgramInfo(program)
    if not dist_variables(prog_info) & distribution.get_variables():
        return distribution, []
    shown: List[Dict[str, Any]] = []
    try:
        for value in dist_values(distribution, prog_info):
            renamed = _rename_value(value.closed, value.variable) if value.closed is not None else None
            closed = str(renamed) if renamed is not None else None
            shown.append({
                "variable": value.variable,
                "probability": str(value.probability),
                "infer": value.site.describe(syntax) if value.site is not None else None,
                "closed_form": closed,
                # rendered with KaTeX like the result's closed forms (#217); short ones only
                "latex": to_latex(renamed) if renamed is not None and len(closed or "") <= 600 else None,
            })
    except Exception:  # pylint: disable=broad-except
        shown = []
    return strip_dist_variables(distribution, prog_info), shown
