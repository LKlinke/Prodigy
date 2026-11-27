"""
The values of the ``dist`` variables at the end of the program and in the steps of the trace (the ``distributions``
section of the result).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sympy

from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution import Distribution
from prodigy.language import Program, VarExpr
from prodigy.util.logger import log_setup
from prodigy.analysis.nested_inference.sites import (ClosedDist, DistRegistry, DistSite, Slot, _rename_value, _sym,
                                                     dist_variables, split_by_value, strip_dist_variables)

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


@dataclass
class DistValue:
    """One value of a ``dist`` variable at the end of the program."""
    variable: str
    probability: Any
    """The probability of this value (sympy); 1 if the value does not depend on the state."""
    site: Optional[DistSite]
    closed: Optional[ClosedDist]


def dist_values(distribution: Distribution, prog_info: ProgramInfo, reg: DistRegistry) -> List[DistValue]:
    """The values of the ``dist`` variables in ``distribution`` (``reg`` knows their sites)."""
    found: List[DistValue] = []
    for var in sorted(dist_variables(prog_info)):
        for part in split_by_value(distribution, var, prog_info, reg):
            probability = sympy.nsimplify(_sym(part.distribution.get_probability_mass()))
            if probability == 0:
                continue
            closed = reg.closed(part.site) if part.site is not None and reg.has_closed(part.site) else None
            found.append(DistValue(var, probability, part.site, closed))
    return found


def _value_json(value: DistValue, limit: int) -> Dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import exact_and_float, to_latex, univariate_table
    probability, probability_float = exact_and_float(value.probability)
    site = value.site
    entry: Dict[str, Any] = {
        "probability": probability,
        "probability_float": probability_float,
        "infer": site.describe() if site is not None else None,
        "function": site.function if site is not None else None,
        "arguments": {} if site is None else {
            var: (arg.describe() if isinstance(arg, DistSite) else
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
        # the mean is optional in the display of a dist value
        logger.debug("No mean for the dist value %s", value.variable, exc_info=True)
    try:
        entry["finite_support"] = bool(renamed.is_finite())
    except Exception:  # pylint: disable=broad-except
        # optional in the display, as the mean
        logger.debug("Finite support of the dist value %s undecided", value.variable, exc_info=True)
    if limit > 0:
        try:
            entry["table"] = univariate_table(renamed, value.variable, limit)
        except Exception as exc:  # pylint: disable=broad-except
            entry["table"] = {"variable": value.variable, "error": str(exc), "rows": []}
    return entry


def dist_values_json(values: List[DistValue], limit: int) -> List[Dict[str, Any]]:
    """The ``distributions`` section of the result JSON (see :mod:`prodigy.analysis.result`)."""
    result: List[Dict[str, Any]] = []
    by_variable: Dict[str, List[DistValue]] = {}
    for value in values:
        by_variable.setdefault(value.variable, []).append(value)
    for variable, found in by_variable.items():
        entries = [_value_json(value, limit) for value in found]
        result.append({
            "variable": variable,
            "depends_on_state": len(found) > 1 or any(value.site is not None and value.closed is None
                                                      for value in found),
            "values": entries,
        })
    return result


def for_trace(distribution: Distribution, program: Optional[Program], reg: DistRegistry
              ) -> Tuple[Distribution, List[Dict[str, Any]]]:
    """A step of the trace shows the program state without the encoding of the ``dist``
    variables, and their values next to it (variable, probability, ``infer`` text, closed form and
    its LaTeX)."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.values import to_latex
    if program is None:
        return distribution, []
    prog_info = ProgramInfo(program)
    if not dist_variables(prog_info) & distribution.get_variables():
        return distribution, []
    shown: List[Dict[str, Any]] = []
    try:
        for value in dist_values(distribution, prog_info, reg):
            renamed = _rename_value(value.closed, value.variable) if value.closed is not None else None
            closed = str(renamed) if renamed is not None else None
            shown.append({
                "variable": value.variable,
                "probability": str(value.probability),
                "infer": value.site.describe() if value.site is not None else None,
                "closed_form": closed,
                # rendered with KaTeX like the result's closed forms (#217); short ones only
                "latex": to_latex(renamed) if renamed is not None and len(closed or "") <= 600 else None,
            })
    except Exception:  # pylint: disable=broad-except
        shown = []
    return strip_dist_variables(distribution, prog_info), shown
