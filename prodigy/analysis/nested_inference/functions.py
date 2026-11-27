"""
``infer``, ``sample``, copies of ``dist`` values and function calls on the joint distribution.

``d = infer(f, a)`` creates a site (closed, or with slots if Z ≡ 1 is proven for every argument), ``x = sample(d)``
draws from it, ``d2 = d`` copies it, and a call whose arguments depend on the state runs the body on the joint
distribution (:func:`call_in_joint`). See the package documentation for the cases.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import sympy
from frozendict import frozendict

from prodigy.analysis.exceptions import UndefinedDistributionError, UnsupportedFeatureError
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.loops.api import Bound
from prodigy.distribution import Distribution
from prodigy.distribution.distribution import constant_value
from prodigy.language import (AsgnInstr, Binop, BinopExpr, BoolLitExpr, CategoricalExpr, Expr, Function,
                              FunctionCallExpr, NatLitExpr, NatType, ObserveInstr, Program, VarExpr)
from prodigy.language.ast.expressions import InferExpr, SampleExpr
from prodigy.language.ast.types import DistributionType
from prodigy.language.checks import recursion
from prodigy.language.ast.walk import Walk, walk_expr, walk_instrs
from prodigy.language.util.ref import Mut
from prodigy.analysis.nested_inference.sites import (ClosedDist, DistSite, Slot, VALUE_VARIABLE, _add_variables,
                                                     _assign, _rename_value, _set_site, _slots_present, _spell, _split,
                                                     _sum, _sym, _unassigned, _values, encoding_variables,
                                                     slot_variable, split_by_value, split_by_values, without)

def function_program_info(prog_info: ProgramInfo, function: Function,
                          bindings: Optional[Dict[str, DistSite]] = None) -> ProgramInfo:
    """The program info for running the body of ``function``. Functions may call other
    functions (#209), but recursion is not implemented (#188)."""
    for name, candidate in prog_info.program.functions.items():
        if candidate is function:
            calls = recursion(prog_info.program, name)
            if calls is not None:
                raise UnsupportedFeatureError(
                    f"Recursive functions are not implemented: {calls}.")
    return replace(prog_info, program=Program.from_function(function, prog_info.program,
                                                                 copy_functions=True),
                   dist_bindings=frozendict(bindings or {}))


def _is_plain(expr: Expr) -> bool:
    """Whether ``expr`` is an expression that ``Distribution.update`` evaluates (no draws)."""
    return not any(isinstance(ref.val, (SampleExpr, InferExpr, FunctionCallExpr, CategoricalExpr))
                   for ref in walk_expr(Walk.DOWN, Mut.alloc(expr)))


def assign_return_value(frame: Frame, function: Function, var: str, state: ProgramState) -> ProgramState:
    """Assigns the return value of ``function`` to the new variable ``var`` (``frame`` runs the body)."""
    return _assign_value(frame, var, function.returns, ProgramState(_add_variables(state.dist, var), state.error))


def _assign_value(frame: Frame, var: str, expr: Expr, state: ProgramState) -> ProgramState:
    """``var = expr`` for a variable ``var`` of type ``nat`` that the program does not declare.
    A plain expression is an update; a draw (``sample(d)``, ``categorical(…)``, a call) runs
    like an assignment in the program."""
    if _is_plain(expr):
        return ProgramState(state.dist.update(BinopExpr(Binop.EQ, VarExpr(var), expr)), state.error)
    program = copy.copy(frame.program.program)
    program.variables = {**program.variables, var: NatType(None)}
    return frame.for_program(replace(frame.program, program=program)).run(AsgnInstr(var, expr), state)


def _has_observe(function: Function) -> bool:
    return any(isinstance(ref.val, ObserveInstr) for ref in walk_instrs(Walk.DOWN, function.instructions))


def closed_form(frame: Frame, site: DistSite) -> ClosedDist:
    """The normalised distribution of the closed ``site``, computed exactly (once)."""
    reg = frame.context.dist_registry
    if reg.has_closed(site):
        return reg.closed(site)
    function = reg.functions[site.function]
    factory = frame.config.factory
    start = factory.one(*function.variables)
    for var, value in site.args:
        if isinstance(value, DistSite):
            closed_form(frame, value)
            value = reg.site_id(value)
        assert isinstance(value, int)
        start = _assign(start, var, value)
    body = frame.for_program(function_program_info(frame.program, function))
    with frame.context.nested_inference(site.describe()) as scope:
        state = body.run(function.instructions, ProgramState(start, factory.from_expr("0")))
        result, _ = assign_return_value(body, function, VALUE_VARIABLE, state)
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


def infer(frame: Frame, instr: AsgnInstr, state: ProgramState) -> ProgramState:
    """``d = infer(f, args…)``."""
    assert isinstance(instr.rhs, InferExpr)
    prog_info, (distribution, error_prob) = frame.program, state
    call = instr.rhs.function
    if call.function not in prog_info.program.functions:
        raise UnsupportedFeatureError(
            f"{_spell(instr.rhs)}: unknown function {call.function}.")
    function = prog_info.program.functions[call.function]
    reg = frame.context.dist_registry
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
    for part, sites in split_by_values(distribution, sorted(set(dist_arg_vars)), prog_info, reg):
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
        results.append(_infer_nat_arguments(frame, instr, _Call(function, given, assignments), fixed, part))
    return _sum(results, distribution, error_prob)


@dataclass(frozen=True)
class _Call:
    """The function of an ``infer``, the names of its given arguments and the expressions of all arguments."""

    function: Function
    given: Tuple[str, ...]
    assignments: Dict[str, Expr]


def _infer_nat_arguments(frame: Frame, instr: AsgnInstr, call: _Call,
                         fixed: Dict[str, Union[int, DistSite, Slot]], distribution: Distribution) -> ProgramState:
    """``d = infer(f, args…)`` on a part in which the dist arguments are ``fixed``: the natural arguments are
    constants, finitely many values (one closed site each) or a slot (only if Z ≡ 1 is proven)."""
    function, given, assignments = call.function, call.given, call.assignments
    prog_info, reg = frame.program, frame.context.dist_registry
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
        closed_form(frame, site)
        return ProgramState(_set_site(without(distribution, old[1:]), target, reg.site_id(site)), zero)

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
            site = make_site({**fixed, **dict(zip(temps.keys(), arg_values, strict=True))})
            if _sym(part.get_probability_mass()) == 0:
                continue
            closed_form(frame, site)
            part = _set_site(without(part, old[1:] + temp_vars), target, reg.site_id(site))
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
    _prove_almost_sure_termination(frame, slot_site, extended.marginal(*temp_vars), temps, describe)
    result = without(extended, old[1:])
    for index, temp in enumerate(temp_vars):
        result = _assign(result, slot_variable(target, index), temp)
    result = without(result, temp_vars)
    return ProgramState(_set_site(result, target, reg.site_id(slot_site)), zero)


def function_name(instr: AsgnInstr) -> str:
    assert isinstance(instr.rhs, InferExpr)
    return instr.rhs.function.function




def _prove_almost_sure_termination(frame: Frame, site: DistSite, arguments: Distribution, temps: Dict[str, str],
                                   describe: str) -> None:
    """Proves Z(a) = 1 for every value a of the arguments: running the function on the
    distribution of the arguments keeps the mass of every argument value."""
    reg = frame.context.dist_registry
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
    with frame.context.nested_inference(site.describe()) as scope:
        result, _ = call_in_joint(frame, function, assignments, {}, result_var,
                                  ProgramState(_add_variables(arguments, result_var), arguments * "0"))
    proven = scope.kind == Bound.EXACT and result.marginal(*temps.values()) == arguments
    if not proven:
        raise UnsupportedFeatureError(
            f"{describe}: the argument depends on the state with infinite support; this is only "
            f"implemented when {site.function} terminates almost surely for every argument (then "
            "Z = 1), which could not be proven.")


# ---------------------------------------------------------------------------------------
# Function calls on the joint distribution
# ---------------------------------------------------------------------------------------

# The call (function, arguments, dist bindings, target) is the instruction already taken apart by its handler,
# which needs the parts itself.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def call_in_joint(frame: Frame, function: Function, assignments: Dict[str, Expr],
                  bindings: Dict[str, DistSite], lhs: str, state: ProgramState) -> ProgramState:
    """``lhs = function(args)`` with arguments that depend on the state: the body runs on the
    joint distribution, so the result keeps its dependence on the caller's variables.
    ``assignments`` gives every variable of the function its initial value (an expression in
    the caller's variables; a ``dist`` parameter gets a variable of the caller, whose slots are
    copied as well)."""
    types = {decl.var: decl.typ for decl in function.declarations}
    dist, error_prob = state
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
        dist, error_prob = _assign_value(frame, temp, expr, ProgramState(_add_variables(dist, temp), error_prob))
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
    body = frame.for_program(function_program_info(frame.program, function, bindings))
    result, error = body.run(function.instructions, ProgramState(dist, error_prob * "0"))
    value = result.get_fresh_variable()
    result, error = assign_return_value(body, function, value, ProgramState(result, error))
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
    return ProgramState(result, error_prob + error)


# ---------------------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------------------



def sample(frame: Frame, instr: AsgnInstr, state: ProgramState) -> ProgramState:
    """``x = sample(d)``: an independent draw from the value of ``d``."""
    assert isinstance(instr.rhs, SampleExpr)
    distribution, error_prob = state
    var = instr.rhs.dist
    target = instr.lhs
    reg = frame.context.dist_registry
    results: List[Tuple[Distribution, Distribution]] = []
    for part in split_by_value(distribution, var, frame.program, reg):
        site = part.site
        if site is None:
            if _sym(part.distribution.get_probability_mass()) == 0:
                continue
            raise _unassigned(var, _spell(instr.rhs))
        if site.closed:
            closed = closed_form(frame, site)
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
        results.append(call_in_joint(frame, function, assignments, {}, target,
                                     ProgramState(part.distribution, error_prob * "0")))
    return _sum(results, distribution, error_prob)


def copy_dist(frame: Frame, instr: AsgnInstr, state: ProgramState) -> ProgramState:
    """``d2 = d``: the copy has the same value (independent draws from both)."""
    assert isinstance(instr.rhs, VarExpr)
    distribution = state.dist
    source, target = instr.rhs.var, instr.lhs
    if source == target:
        return state
    bound = frame.program.dist_bindings.get(source)
    result = without(distribution, encoding_variables(distribution, target)[1:])
    if bound is not None:
        return ProgramState(_set_site(result, target, frame.context.dist_registry.site_id(bound)), state.error)
    if source not in result.get_variables():
        return ProgramState(_set_site(result, target, 0), state.error)
    result = _assign(result, target, source)
    for index, slot in enumerate(_slots_present(result, source)):
        result = _assign(result, slot_variable(target, index), slot)
    return ProgramState(result, state.error)
