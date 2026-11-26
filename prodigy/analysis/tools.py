"""
The analysis tools beside the session, as the front ends (``prodigy.cli``, ``prodigy.web``) use them (T8 of
``docs/source/architecture.md``).

The front-end API of the analysis is the session, the result, the configuration, the loop registry, the trace,
the values, the checks and the exceptions (their own modules), and this module: what the CLI's extra commands
and the GUI's extra endpoints need from the analysis, re-exported from the modules that implement it. Front ends
import nothing else from ``prodigy.analysis`` (``tests/test_layering.py``), so the modules behind this one can be
moved or split without touching a front end; only the imports here follow.

This module owns no logic. Grouped by what the front ends do with it:

- running instructions without a session (``prodigy`` subcommands for single loops): :func:`compute_semantics`,
  :class:`ProgramInfo`, initial distributions from text;
- the option enums of the loop strategies (choices of the CLI and the GUI) and their defaults;
- EVT invariants: synthesis and verification; sandwich bounds (``prodigy evt_sandwich``, EVT sub-invariants);
- independence of variables (``prodigy independent_vars``);
- the comparison of two programs (``prodigy check_equality``, the GUI's compare mode);
- the joint view of two variables (GUI), the program's assigned distribution variables (GUI diagnostics);
- candidates of a language model (GUI suggestions; the answer is untrusted and validated there).
"""
from prodigy.analysis.ai_candidates import (FORMS, SYSTEM_PROMPT, TARGET_PARAMS, AIConfigurationError, AIRequestError,
                                            CandidateError, candidate_from_answer, get_provider, prompt)
from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.equivalence.verdict import (UNKNOWN, CheckOutcome, Comparison, analyse_on_input, compare, decide,
                                                  different_variables, find_witness, run_check, show_valuation)
from prodigy.analysis.evtinvariants.heuristics.positivity.heuristics_factory import PositivityHeuristics
from prodigy.analysis.evtinvariants.heuristics.strategies import (DEFAULT_MAX_DEGREE, SELECTABLE_POSITIVITY_HEURISTICS,
                                                                  SynthesisStrategies)
from prodigy.analysis.evtinvariants.heuristics.templates.templates_factory import TemplateHeuristics
from prodigy.analysis.evtinvariants.invariant_synthesis import (DEFAULT_SYNTHESIS_TIME_LIMIT,
                                                                DEFAULT_TEMPLATE_TIME_LIMIT, evt_invariant_synthesis)
from prodigy.analysis.evtinvariants.invariant_verification import evt_invariant_verification
from prodigy.analysis.independence import independent_vars
from prodigy.analysis.input_distribution import InputDistributionError, input_distribution_from_text
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.joint_grid import GRID_HEADER, MAX_AXIS, GridError, check_window, grid_from_known, joint_grid
from prodigy.analysis.lowerbounds.parameters import TemplateShape
from prodigy.analysis.lowerbounds.problem import load_problem
from prodigy.analysis.lowerbounds.sandwich import evt_sandwich, print_result, write_json
from prodigy.analysis.lowerbounds.subinvariant import evt_subinvariant_verification
from prodigy.analysis.nested_inference import assigned_dist_variables
from prodigy.analysis.solver.solver_type import SolverType

__all__ = [
    # running instructions without a session
    "compute_semantics", "ProgramInfo", "InputDistributionError", "input_distribution_from_text",
    # option enums of the loop strategies
    "DEFAULT_MAX_DEGREE", "SELECTABLE_POSITIVITY_HEURISTICS", "PositivityHeuristics", "SolverType",
    "SynthesisStrategies", "TemplateHeuristics",
    # EVT invariants and sandwich bounds
    "DEFAULT_SYNTHESIS_TIME_LIMIT", "DEFAULT_TEMPLATE_TIME_LIMIT", "evt_invariant_synthesis",
    "evt_invariant_verification", "evt_subinvariant_verification", "TemplateShape", "load_problem", "evt_sandwich",
    "print_result", "write_json",
    # independence
    "independent_vars",
    # comparison of two programs
    "UNKNOWN", "CheckOutcome", "Comparison", "analyse_on_input", "compare", "decide", "different_variables",
    "find_witness", "run_check", "show_valuation",
    # GUI: joint view, diagnostics
    "GRID_HEADER", "MAX_AXIS", "GridError", "check_window", "grid_from_known", "joint_grid", "assigned_dist_variables",
    # GUI: candidates of a language model
    "FORMS", "SYSTEM_PROMPT", "TARGET_PARAMS", "AIConfigurationError", "AIRequestError", "CandidateError",
    "candidate_from_answer", "get_provider", "prompt",
]
