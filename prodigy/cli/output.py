"""Text output of the commands: the summary of an analysis result (``prodigy analyze``), the verdict of a
comparison (``prodigy check_equality``) and the plots of a run (``prodigy main``)."""
from typing import Any, Dict, Mapping, Optional

from prodigy.analysis.session import AnalysisSession
from prodigy.analysis.tools import Comparison, show_valuation
from prodigy.util.color import Style

_QUERY_LABELS = {"probability": "Pr", "expectation": "E", "variance": "Var", "covariance": "Cov"}

AnalysisJson = Dict[str, Any]
"""The JSON result of an analysis (``prodigy/analysis-result@1``, :mod:`prodigy.analysis.result`); typed with P9."""


def format_value(exact: Optional[str], number: Optional[float]) -> str:
    """A number as six significant digits, with its exact form if that is short."""
    if number is not None:
        text = f"{number:.6g}"
        return text if exact is None or exact == text else f"{text}  ({exact})" if len(exact) <= 40 else text
    return exact if exact is not None else "?"


def _print_status(result: AnalysisJson) -> None:
    status = result["status"]
    if status == "ok":
        return
    error = result.get("error") or {}
    print(f"{Style.OKRED}{status}:{Style.RESET} {error.get('message', '')}")
    if error.get("verdict") == "refuted":
        print("Verdict: refuted (the invariant is provably wrong).")
    elif error.get("verdict") == "unknown":
        print("Verdict: not proven (neither proven nor refuted; the invariant may still be correct).")
    if status == "needs_decision":
        print("Choose strategies with --loop ID=STRATEGY[:NAME=VALUE,...]; "
              "`prodigy outline FILE` lists the loops, `prodigy loop_strategies` the strategies.")


def _print_distribution(result: AnalysisJson) -> None:
    dist = result.get("distribution")
    if dist:
        print(f"{Style.OKBLUE}Result:{Style.RESET}\t{dist['closed_form']}")
        if dist.get("error_probability") not in (None, "0"):
            print(f"{Style.OKBLUE}Observation violated with probability:{Style.RESET}\t{dist['error_probability']}")
    bounds = result.get("bounds") or {}
    if bounds and result["status"] == "ok":
        print(f"{Style.OKBLUE}Guarantee:{Style.RESET}\t{bounds.get('explanation', bounds.get('kind'))}")
        if bounds.get("upper"):
            print(f"{Style.OKBLUE}Upper bound per state:{Style.RESET}\tproven by the certificate of "
                  f"{bounds['upper']['origin']}")


def _interval(query: Mapping[str, Any]) -> str:
    """The interval of a query that is not exact, e.g. ``∈ [0.5, ∞]  (lower)``; empty for an exact one."""
    if query["bound"] == "exact":
        return ""
    low = format_value(query["lower"], query["lower_float"]) if query["lower"] is not None else "?"
    high = format_value(query["upper"], query["upper_float"]) if query["upper"] is not None else "∞" \
        if query["bound"] == "lower" else "?"
    source = "; upper end from the certificate" if query.get("upper_source") == "certificate" else ""
    return f"\t∈ [{low}, {high}]  ({query['bound']}{source})"


def _query_line(query: Mapping[str, Any]) -> Optional[str]:
    """The line of the summary for one query, None for a query without one (``print``, ``plot``)."""
    where = f"line {query['line']}: " if query.get("line") else ""
    kind = query["kind"]
    if query.get("error"):
        label = _QUERY_LABELS.get(kind, kind)
        label = f"{label}[{query['expr']}]" if kind in _QUERY_LABELS else \
            f"{label} of {query['expr']}" if query["expr"] else label
        return f"  {where}{label}: {Style.OKRED}not computed:{Style.RESET} {query['error']['message']}"
    if kind in _QUERY_LABELS:
        label = f"{_QUERY_LABELS[kind]}[{query['expr']}]"
        return f"  {where}{label} = {format_value(query['value'], query['value_float'])}{_interval(query)}"
    if kind == "marginal":
        return f"  {where}marginal of {query['expr']}: {query['value']}"
    if kind == "optimization":
        return f"  {where}optimum for {query['expr']}: {query['value']}"
    return None


def _loop_line(loop: Mapping[str, Any]) -> str:
    where = f" (line {loop['line']})" if loop.get("line") else ""
    line = f"  loop {loop['id']}{where} [{loop['strategy']}] {loop['kind']}"
    if loop["messages"]:
        line += ": " + " ".join(loop["messages"])
    elif loop.get("failed"):
        # the error itself is printed above (#333)
        line += " (the strategy failed, see the error above)"
    return line


def print_summary(result: AnalysisJson) -> None:
    """The text output of ``prodigy analyze``: status, distribution and guarantee, queries, loops, warnings."""
    _print_status(result)
    _print_distribution(result)
    for query in result.get("queries", []):
        line = _query_line(query)
        if line is not None:
            print(line)
    for loop in result.get("loops", []):
        print(_loop_line(loop))
    for warning in result.get("warnings", []):
        print(f"{Style.YELLOW}Warning:{Style.RESET} {warning}")
    print(f"CPU-time elapsed: {result['timing'].get('analysis', 0):04f} seconds")


def _print_verdict(verdict: Mapping[str, Any]) -> None:
    if verdict["verdict"] == "equivalent":
        print(f"Verdict: {Style.OKGREEN}A is equivalent to B{Style.RESET}")
    elif verdict["verdict"] == "not_equivalent":
        print(f"Verdict: {Style.OKRED}A is not equivalent to B{Style.RESET}")
    else:
        relation = verdict.get("relation")
        proven = f" (one direction proven: {relation})" if relation else ""
        print(f"Verdict: {Style.OKYELLOW}undecided{Style.RESET}{proven}")
    print(f"Reason: {verdict['reason']}")
    if verdict.get("constraints"):
        print("Parameter values: " + "; ".join(
            ", ".join(f"{name} = {value}" for name, value in constraint.items())
            for constraint in verdict["constraints"]))


def print_comparison(comparison: Comparison, first: str, second: str) -> None:
    """The text output of ``prodigy check_equality`` for programs A (``first``) and B (``second``)."""
    check, verdict, witness = comparison.check, comparison.verdict, comparison.witness
    print(f"Program A: {first}\nProgram B: {second}")
    answers = {"equal": "the computed distributions are equal", "different": "the computed distributions differ",
               "undecided": "undecided", "failed": "did not finish"}
    line = f"Check for every input: {answers.get(check.answer, check.answer)}"
    if check.error is not None and check.error.get("type") == "variables":
        line = "Check for every input: not run"
    elif check.answer != "failed":
        line += f" (analysis of A: {check.first}, of B: {check.second})"
    print(line)
    _print_verdict(verdict)
    if witness is not None:
        shown_input = ", ".join(f"{k} = {v}" for k, v in sorted(comparison.input.items()) if v) or "every variable 0"
        state = show_valuation(witness["state"], "the empty state")
        print(f"Witness: on the input {shown_input}, the output state {state} has probability")
        for name, side in (("A", witness["first"]), ("B", witness["second"])):
            value = side["lower"] if side["lower"] == side["upper"] else f"in [{side['lower']}, {side['upper']}]"
            print(f"  {value} in program {name} ({side['kind']})")
    elif check.difference is not None and verdict["verdict"] == "unknown":
        print(f"Difference of the outputs (A - B) for every input: {check.difference}")


def draw_plots(session: AnalysisSession) -> None:
    """Draws the ``plot`` queries of a finished run with matplotlib (the analysis only records their data, T7)."""
    plots = [query for query in session.queries if query.kind == "plot"]
    if not plots:
        return
    # matplotlib is loaded only when a program plots
    from prodigy.plotting.plotter import PlotError, plot  # pylint: disable=import-outside-toplevel
    for query in plots:
        try:
            plot(query.data)
        except PlotError as exc:
            print(f"{Style.YELLOW}Plot of {query.expr}:{Style.RESET} {exc}")
