"""
The checks of the fuzzing tests: one program, analysed on every engine (and with simplification on
and off), compared with each other and with the simulator (:mod:`prodigy.simulator`).

:func:`check_program` returns the list of disagreements; an empty list means the program passed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.equality import decide_equal_strings
from prodigy.pgcl.parser import parse_pgcl
from prodigy.simulator import SimulationError, UnsupportedConstruct, compare, simulate

Engine = ForwardAnalysisConfig.Engine


@dataclass(frozen=True)
class Setup:
    """One analysis configuration."""
    name: str
    engine: Any
    simplification: bool = False


SETUPS = (Setup("sympy", Engine.SYMPY), Setup("sympy+simplify", Engine.SYMPY, True),
          Setup("ginac", Engine.GINAC), Setup("symengine", Engine.SYMENGINE))
_NON_FINITE = ("nan", "zoo", "oo", "inf")


def analyse(source: str, syntax: str, setup: Setup, decisions: Mapping[str, Any]) -> Dict[str, Any]:
    """The JSON result of analysing ``source`` with ``setup``."""
    program = parse_pgcl(source, syntax)
    config = ForwardAnalysisConfig(engine=setup.engine, use_simplification=setup.simplification)
    session = AnalysisSession(program, source=source, syntax=syntax, config=config, decisions=dict(decisions),
                              capture_output=True, coefficient_limit=12)
    return session.run().to_json()


def _non_finite(text: Optional[str]) -> bool:
    if not text:
        return False
    tokens = set(text.replace("(", " ").replace(")", " ").replace("*", " ").replace("-", " ").split())
    return any(word in tokens for word in _NON_FINITE)


def _numbers(result: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """The exact numbers of a result, keyed by what they are."""
    numbers: Dict[str, Optional[str]] = {}
    distribution = result.get("distribution") or {}
    numbers["distribution"] = distribution.get("closed_form")
    numbers["mass"] = distribution.get("mass")
    for query in result.get("queries", []):
        if query.get("kind") in ("probability", "expectation"):
            numbers[f"query {query.get('index')} {query.get('expr')}"] = query.get("value")
    return numbers


def check_program(source: str, syntax: str = "ppy", decisions: Optional[Mapping[str, Any]] = None, *,
                  setups=SETUPS, runs: int = 2000, seed: Any = 0,
                  report: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Analyses ``source`` with every setup and returns the disagreements:

    * a result with status ``ok`` that contains ``nan``, ``zoo`` or ``oo`` (#165);
    * two ``ok`` results whose distribution, mass or query values are provably different
      (:func:`~prodigy.distribution.equality.decide_equal_strings`), or that differ in the kind of
      guarantee (exact, lower, upper);
    * an ``ok`` result outside the confidence intervals of the simulation.

    Results with another status (e.g. a "not implemented" error on one engine) are incomplete, not
    wrong, and are not compared. ``report`` (if given) receives the results by setup.
    """
    decisions = decisions or {}
    problems: List[str] = []
    results = {setup.name: analyse(source, syntax, setup, decisions) for setup in setups}
    if report is not None:
        report.update(results)
    ok = {name: result for name, result in results.items() if result["status"] == "ok"}
    for name, result in ok.items():
        for what, value in _numbers(result).items():
            if _non_finite(value) and not (what.startswith("query") and value in ("oo", "inf", "\\infty")):
                problems.append(f"{name}: status ok, but {what} is {value}")
    names = sorted({var for result in ok.values() for var in (result.get("distribution") or {}).get("variables", [])}
                   | {"x", "y", "z", "c"})
    reference_name = next(iter(ok), None)
    if reference_name is not None:
        reference = ok[reference_name]
        expected = _numbers(reference)
        for name, result in ok.items():
            if name == reference_name:
                continue
            if result["bounds"]["kind"] != reference["bounds"]["kind"]:
                problems.append(f"{name}: guarantee {result['bounds']['kind']}, "
                                f"{reference_name}: {reference['bounds']['kind']}")
            for what, value in _numbers(result).items():
                other = expected.get(what)
                if value is None or other is None:
                    continue
                if decide_equal_strings(value, other, names) is False:
                    problems.append(f"{what}: {name} gives {value}, {reference_name} gives {other}")
        try:
            simulation = simulate(parse_pgcl(source, syntax), runs, seed=seed)
        except UnsupportedConstruct:
            simulation = None
        except SimulationError as error:
            problems.append(f"the analysis accepts the program, but a run fails: {error}")
            simulation = None
        if simulation is not None:
            for name, result in ok.items():
                problems.extend(f"{name} vs. simulation: {m}"
                                for m in compare(parse_pgcl(source, syntax), result, simulation))
    return problems


def statuses(source: str, syntax: str = "ppy", decisions: Optional[Mapping[str, Any]] = None,
             setups=SETUPS) -> Dict[str, str]:
    """The status of every setup (for reports)."""
    return {setup.name: analyse(source, syntax, setup, decisions or {})["status"] for setup in setups}


def finite(value: Optional[float]) -> bool:
    """Whether ``value`` is a finite float."""
    return value is not None and math.isfinite(value)
