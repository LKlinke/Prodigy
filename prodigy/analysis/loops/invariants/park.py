"""
The strategy ``invariant``: a loop-free program I with Φ(I) = I is an upper bound on the loop by Park's lemma
(exact if the loop terminates almost surely); a refutation needs a witness (:mod:`prodigy.analysis.loops.witness`).
"""
from __future__ import annotations

from typing import Any, Dict, List

from lark.exceptions import LarkError

from prodigy.language import parse_program, Program

from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.exceptions import ParameterSynthesisRequired
from prodigy.analysis.loops import witness as witnesses
from prodigy.analysis.loops.witness import show_valuation
from prodigy.analysis.loops.api import Bound, Delivers, LoopOutcome, LoopStrategy, LoopTask, ParamSpec
from prodigy.analysis.loops.registry import register_loop_strategy
from prodigy.distribution import Distribution, State
from prodigy.language.operations import cav_phi
from prodigy.util.color import Style
from prodigy.analysis.loops.common import InvariantRefutedError


def _strategy_invariant(task: LoopTask) -> LoopOutcome:
    instruction, prog_info, config = task.instruction, task.prog_info, task.config
    inv_src = task.param("invariant")
    try:
        inv_prog = parse_program(inv_src)
    except (SyntaxError, LarkError) as exc:
        raise ValueError(f"The invariant is not a valid .ppy program: {exc}") from exc

    prog = Program(declarations=[],
                   variables=prog_info.variables,
                   constants=prog_info.constants,
                   parameters=prog_info.parameters,
                   instructions=[instruction],
                   functions=prog_info.functions)
    # template parameters: parameters of the invariant that the program does not have
    template = sorted(str(p) for p in set(inv_prog.parameters) - set(prog_info.parameters))
    if template and task.session.parameter_synthesis == "refuse":
        raise _synthesis_required(task, inv_src, template, [])
    print(f"{Style.YELLOW}Verifying invariant...{Style.RESET}")

    # First we create the modified input program in order to fit the premise of Park's Lemma
    phi_inv = cav_phi(prog, inv_prog)

    with task.session.sub_analysis("invariant check") as scope:
        answer, result = check_equivalence(inv_prog, phi_inv, config, task.analyzer)

    certificate: Dict[str, Any] = {
        "type": "invariant",
        "rule": "Park",
        "claim": "Φ(I) = I for all inputs, hence ⟦while⟧ ⪯ ⟦I⟧",
        "invariant": inv_src,
    }
    if answer:
        assert isinstance(result, list)
        constraints = [{str(k): str(v) for k, v in solution.items()} for solution in result]
        certificate["verdict"] = "verified"
        certificate["constraints"] = constraints
        messages = []
        if len(result) == 0:
            print(Style.OKGREEN + "Invariant successfully validated!" + Style.RESET)
        else:
            print(
                f"{Style.OKGREEN}Invariant validated under the following constraints:{Style.RESET} {result}"
            )
            messages.append(f"The invariant holds under the constraints {constraints}; "
                            "the result still contains the parameters.")
        if template or any(constraints):
            # parameter synthesis: a comparison of the loop with the invariant program (#154)
            if task.session.parameter_synthesis == "refuse":
                raise _synthesis_required(task, inv_src, template, constraints)
            if task.session.parameter_synthesis == "hint":
                hint = ("This is parameter synthesis, a comparison of the loop with the invariant program: "
                        "`prodigy check_equality PROGRAM INVARIANT` reports the parameter values with a verdict.")
                print(f"{Style.YELLOW}Hint:{Style.RESET} {hint}")
                messages.append(hint)
        if config.show_intermediate_steps:
            print(Style.YELLOW +
                  "Compute the result using the invariant" +
                  Style.RESET)
        dist, error_prob = task.analyzer(inv_prog.instructions, prog_info, task.distribution, task.error_prob,
                                         config)
        if scope.kind == Bound.EXACT:
            kind = Bound.UPPER
            messages.append("Upper bound by Park's lemma; exact if the loop terminates almost surely.")
        elif scope.kind == Bound.UPPER:
            # an over-approximated body Φ' ⪰ Φ with Φ'(I) = I gives Φ(I) ⪯ I, so Park still applies
            kind = Bound.UPPER
            messages.append("Upper bound by Park's lemma; loops inside the body were over-approximated "
                            "during the check, which keeps the argument sound.")
        else:
            kind = Bound.UNKNOWN
            messages.append("Loops inside the body were under-approximated during the check; no guarantee.")
        return LoopOutcome(dist, error_prob, kind=kind, certificate=certificate, messages=messages)

    if answer is False:
        assert isinstance(result, State)
        raise _refutation(task, inv_prog, phi_inv, result, scope.kind, certificate)

    assert answer is None
    assert isinstance(result, Distribution)
    print(
        f'{Style.OKRED}Could not determine whether invariant is valid.{Style.RESET} Phi(I) - I: {result}'
    )
    certificate["verdict"] = "unknown"
    certificate["difference"] = str(result)
    raise InvariantRefutedError("Could not determine whether invariant is valid.", details=certificate)


def _refutation(task: LoopTask, inv_prog: Program, phi_inv: Program, counterexample: State, kind: str,
                certificate: Dict[str, Any]) -> InvariantRefutedError:
    """The error for a raw "different" of the check Φ(I) = I. The raw answer is no proof: the solver may
    fail to find parameter values, and loops inside the body may have been approximated during the check
    (#403). So the invariant counts as ``refuted`` only with a witness, an output state whose probabilities
    under I and Φ(I) provably differ on the input of the counterexample (:mod:`prodigy.analysis.loops.witness`,
    as in :mod:`prodigy.analysis.equivalence.verdict`); otherwise the verdict is ``unknown``."""
    state = {str(k): int(v) for k, v in counterexample.valuations.items()}
    shown = show_valuation(state)
    certificate["counterexample"] = state
    witness = witnesses.search_witness(inv_prog, phi_inv, task.session, state, task.analyzer)
    if witness is not None:
        output = show_valuation(witness["state"], "the empty state")
        print(f"{Style.OKRED}Invariant refuted.{Style.RESET} On the input {shown}, I and Φ(I) give provably "
              f"different probabilities to the output state {output}.")
        certificate["verdict"] = "refuted"
        certificate["witness"] = witness
        return InvariantRefutedError(
            f"The invariant is refuted: on the input {shown}, the invariant program I and one unrolling of the "
            f"loop followed by I, Φ(I), give provably different probabilities to the output state {output} "
            f"(I: {witness['first']['lower']} to {witness['first']['upper']}, Φ(I): {witness['second']['lower']} "
            f"to {witness['second']['upper']}).", details=certificate)
    reasons = []
    if kind != Bound.EXACT:
        reasons.append(f"loops inside the body were approximated during the check ({kind})")
    if task.prog_info.parameters or inv_prog.parameters:
        reasons.append("the programs have parameters, and the solver found no parameter values that make them "
                       "equal, which is no proof that there are none")
    why = "; ".join(reasons) or "no output state on that input has provably different probabilities"
    print(f"{Style.OKRED}Invariant not proven.{Style.RESET} The check found I and Φ(I) to differ (candidate "
          f"counterexample: the input {shown}), but no witness was confirmed: {why}.")
    certificate["verdict"] = "unknown"
    return InvariantRefutedError(
        f"The invariant is not proven (and not refuted): the check found I and Φ(I) to differ, with the input "
        f"{shown} as candidate counterexample, but no witness was confirmed: {why}.", details=certificate)


def _synthesis_required(task: LoopTask, inv_src: str, template: List[str],
                        constraints: List[Dict[str, str]]) -> ParameterSynthesisRequired:
    """The error of the strategy ``invariant`` for parameter synthesis in an analysis (#154), with what a
    front end needs to open the loop and the invariant as two programs to compare."""
    names = template or sorted({name for constraint in constraints for name in constraint})
    what = (f"The invariant has the parameter{'s' if len(names) > 1 else ''} {', '.join(names)}"
            if template else f"The check of the invariant fixes {', '.join(names)}")
    where = task.loop.describe() if task.loop is not None else "the loop"
    return ParameterSynthesisRequired(
        f"{what}: finding parameter values is parameter synthesis, a comparison of {where} with the invariant "
        "program, not an analysis step. Open the pair in Compare, or use an invariant without parameters.",
        details={"type": "parameter_synthesis", "loop": task.loop.id if task.loop is not None else None,
                 "parameters": names, "constraints": constraints, "invariant": inv_src,
                 **_loop_program(task)})


def _loop_program(task: LoopTask) -> Dict[str, Any]:
    """The program to compare with the invariant: the program itself if the loop is all it does, else its
    declarations and the loop (``program``, ``.ppy``), and the id of the loop in it."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.loops.api import isolate_loop
    session = task.session
    if session.program is None:
        return {"program": None, "program_loop": None}
    isolated = isolate_loop(session.program, task.instruction, session.source)
    if isolated is None:
        return {"program": None, "program_loop": None}
    return {"program": isolated.program, "program_loop": isolated.loop}


_INVARIANT_PARAMS = (
    ParamSpec("invariant", "program", "Invariant (loop-free .ppy program)",
              prompt="Invariant file:\t",
              help="A loop-free program I with the loop's variables and no parameters to find; checked "
                   "against one unrolling of the loop (Park's lemma). Finding parameter values is a "
                   "comparison: Compare mode, or prodigy check_equality."),
    ParamSpec("invariant_file", "file", "Invariant file", prompt="Invariant file:\t",
              alias_of="invariant", help="Path of a file containing the invariant (CLI only)."),
)

register_loop_strategy(LoopStrategy(
    name="invariant",
    title="Check invariant",
    group="exact",
    outcome="upper bound",
    summary="Exact if the loop terminates almost surely, otherwise an upper bound; you provide a loop-free "
            "program that behaves like the loop; one check, usually fast.",
    legacy_key="1",
    soundness=Bound.UPPER,
    delivers=(Delivers(Bound.UPPER, "Φ(I) = I is proven for the given loop-free I"),),
    description="Checks that a loop-free program I satisfies I ≡ if (guard) { body; I } on all "
                "inputs and continues with I. By Park's lemma the result over-approximates the "
                "loop; it is exact if the loop terminates almost surely. I is concrete: an invariant "
                "with parameters to be found (parameter synthesis) is a comparison of the loop with I, "
                "done in the GUI's Compare mode or with `prodigy check_equality`.",
    params=_INVARIANT_PARAMS,
    run=_strategy_invariant,
))
