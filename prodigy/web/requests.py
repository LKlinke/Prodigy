"""
Analysis requests of the web GUI (JSON in, JSON out).

A request looks like::

    {
      "source": "nat x; ...",
      "input_distribution": "x^3",          # optional, default: all variables 0
      "engine": "sympy" | "ginac" | "symengine",
      "normalize": true,                     # condition on observations
      "simplification": true,
      "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 50}}, "*": ...},
      "coefficients": 30,                    # coefficients per marginal in the result
      "record_steps": false,
      "synthesis_strategy": "default",       # for EVT invariant synthesis
      "solver": "sympy" | "z3"
    }
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

ENGINES = {
    "sympy": "SymPy",
    "ginac": "GiNaC",
    "symengine": "SymEngine",
}

MAX_COEFFICIENTS = 200


def validate_request(request: Dict[str, Any]) -> List[str]:
    """Problems that can be found without running the analysis."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
    from prodigy.analysis.session import parse_decisions, validate_decisions
    from prodigy.analysis.solver.solver_type import SolverType

    problems = []
    if not isinstance(request.get("source"), str) or not request["source"].strip():
        problems.append("The program is empty.")
    if request.get("engine", "sympy") not in ENGINES:
        problems.append(f"Unknown engine {request.get('engine')!r}.")
    try:
        coefficients = int(request.get("coefficients", 30))
        if not 0 <= coefficients <= MAX_COEFFICIENTS:
            problems.append(f"Coefficients must be between 0 and {MAX_COEFFICIENTS}.")
    except (TypeError, ValueError):
        problems.append("Coefficients must be a number.")
    if str(request.get("synthesis_strategy", "default")).upper() not in SynthesisStrategies.__members__:
        problems.append(f"Unknown synthesis strategy {request.get('synthesis_strategy')!r}.")
    if str(request.get("solver", "sympy")).upper() not in SolverType.__members__:
        problems.append(f"Unknown solver {request.get('solver')!r}.")
    try:
        decisions = parse_decisions(request.get("decisions") or {})
        problems.extend(validate_decisions(decisions))
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def make_config(request: Dict[str, Any]):
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.config import ForwardAnalysisConfig
    from prodigy.analysis.evtinvariants.heuristics.strategies import SynthesisStrategies
    from prodigy.analysis.solver.solver_type import SolverType
    return ForwardAnalysisConfig(
        engine=ForwardAnalysisConfig.Engine[str(request.get("engine", "sympy")).upper()],
        use_simplification=bool(request.get("simplification", True)),
        normalize=bool(request.get("normalize", True)),
        strategy=SynthesisStrategies[str(request.get("synthesis_strategy", "default")).upper()],
        solver_type=SolverType[str(request.get("solver", "sympy")).upper()],
    )


def run_analysis(request: Dict[str, Any], *, output: Optional[Callable[[str], None]] = None,
                 progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Runs one analysis request and returns the JSON result (never raises for analysis errors)."""
    # pylint: disable=import-outside-toplevel
    from probably.pgcl import parse_pgcl

    from prodigy.analysis.session import AnalysisSession

    problems = validate_request(request)
    if problems:
        return {"status": "error", "error": {"type": "request", "message": " ".join(problems),
                                             "problems": problems}}
    source = request["source"]
    try:
        program = parse_pgcl(source)
    except Exception as exc:  # pylint: disable=broad-except
        return {"status": "error", "error": {"type": "syntax", "message": str(exc).split("\n", maxsplit=1)[0]}}
    input_distribution = request.get("input_distribution")
    if isinstance(input_distribution, str) and not input_distribution.strip():
        input_distribution = None
    session = AnalysisSession(
        program,
        source=source,
        config=make_config(request),
        decisions=request.get("decisions") or {},
        input_distribution=input_distribution,
        allow_files=False,
        capture_output=True,
        output_callback=output,
        progress_callback=progress,
        record_steps=bool(request.get("record_steps", False)),
        coefficient_limit=int(request.get("coefficients", 30)),
    )
    return session.run().to_json()
