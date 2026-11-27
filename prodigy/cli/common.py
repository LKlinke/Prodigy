"""What the commands share: reading and checking programs, the options ``--syntax`` and ``--loop``, loop decisions,
expressions given as options and the initial distribution."""
import json
import logging
import sys
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, TypeVar

import click

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import DEFAULT_KEY, LoopRegistry, parse_decisions, validate_decisions
from prodigy.analysis.tools import input_distribution_from_text
from prodigy.distribution import Distribution
from prodigy.language import Instr, Program, WhileInstr
from prodigy.language.parser import check_syntax, parse_program
from prodigy.language.syntax_errors import ProgramSyntaxError
from prodigy.util.color import Style
from prodigy.util.logger import log_setup

logger = log_setup("CLI", logging.DEBUG)

_SYNTAX_HELP = ("Syntax of the program files: ppy, the only one (kept for compatibility; the .pgcl syntax "
                "was removed in v0.6.0, convert such programs with Prodigy v0.5.x).")

Command = TypeVar("Command", bound=Callable[..., Any])  # a click command function


def syntax_option(command: Command) -> Command:
    """The option ``--syntax`` of the commands that read programs."""
    # no Choice: a removed syntax is answered with the removal message (check_syntax), not with click's error
    return click.option('--syntax', metavar="ppy", default=None, help=_SYNTAX_HELP)(command)


def _with_source_line(message: str, source: str, line: Optional[int], column: Optional[int]) -> str:
    lines = source.split("\n")
    if line is None or column is None or not 1 <= line <= len(lines):
        return message
    return f"{message}\n    {lines[line - 1]}\n    {' ' * (column - 1)}^"


def read_program(program_file: IO, syntax: Optional[str]) -> Tuple[str, Program]:
    """Read and parse a ``.ppy`` program file; ``--syntax`` and the file extension are only checked (the
    boundary of the CLI, :func:`prodigy.language.parser.check_syntax`)."""
    source = program_file.read()
    name = getattr(program_file, "name", "<program>")
    try:
        check_syntax(syntax, name)
    except ValueError as exc:
        raise click.ClickException(f"{name}: {exc}") from exc
    try:
        program = parse_program(source)
    except ProgramSyntaxError as exc:
        raise click.ClickException(_with_source_line(f"{name}: {exc}", source, exc.line, exc.column)) from exc
    return source, program


def require_type_correct(program_file: IO, program: Program, source: str) -> None:
    """Ends with the GUI's message if the type checker rejects the program (#334); prints its warnings on
    stderr."""
    # pylint: disable=import-outside-toplevel
    from prodigy.analysis.checks import type_error
    from prodigy.language.diagnostics import language_warning_diagnostics
    diagnostic = type_error(program, source)
    name = getattr(program_file, "name", "<program>")
    if diagnostic is not None:
        raise click.ClickException(_with_source_line(f"{name}: type error: {diagnostic['message']}", source,
                                                     diagnostic["line"], diagnostic["column"]))
    # warnings go to stderr, so that --json stays clean (#377, #335)
    for warning in language_warning_diagnostics(program, source):
        print(f"{name}:{warning['line']}:{warning['column']}: {Style.YELLOW}warning:{Style.RESET} "
              f"{warning['message']}", file=sys.stderr)


def checked_program(program_file: IO, syntax: Optional[str]) -> Tuple[str, Program]:
    """:func:`read_program` and :func:`require_type_correct`: the source and the program of a type-correct file."""
    source, program = read_program(program_file, syntax)
    require_type_correct(program_file, program, source)
    return source, program


LOOP_OPTION_HELP = ("Strategy for a loop, as ID=STRATEGY[:NAME=VALUE,...], e.g. 0=unroll:iterations=100 or "
                    "'*=auto' for all loops. See `prodigy loop_strategies` for the strategies by group: "
                    "automatic, exact result, bounds, explore.")


RawDecisions = Dict[str, Any]
"""Loop decisions as read from ``--loop`` and ``--decisions``: keys are loop ids or ``*``, values are strategy
names or JSON objects (``Any`` is correct here; :func:`~prodigy.analysis.session.parse_decisions` checks them)."""


def collect_decisions(loop_specs: Tuple[str, ...], decisions_file: Optional[IO]) -> RawDecisions:
    """The decisions of ``--decisions`` (a JSON file), overridden by those of ``--loop``."""
    decisions: RawDecisions = {}
    if decisions_file is not None:
        loaded = json.load(decisions_file)
        decisions.update(loaded.get("decisions", loaded) if isinstance(loaded, dict) else {})
    for spec in loop_specs:
        key, sep, rest = spec.partition("=")
        key = key.strip()
        if sep and (key == DEFAULT_KEY or key.isdigit()):
            decisions[key] = rest
        else:
            decisions[DEFAULT_KEY] = spec
    return decisions


def checked_decisions(decisions: RawDecisions, program: Program) -> RawDecisions:
    """``decisions`` (from ``--loop``/``--decisions``) after checking them, in particular their generating
    functions, before any engine sees them; exits with the problems otherwise."""
    try:
        parsed = parse_decisions(decisions)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--loop' / '--decisions'") from exc
    problems = validate_decisions(parsed, LoopRegistry(program))
    if problems:
        for problem in problems:
            print(f"{Style.OKRED}Error:{Style.RESET} {problem}", file=sys.stderr)
        sys.exit(2)
    return decisions


def expression_option(text: str, option: str, program: Program) -> str:
    """The generating function ``text`` of the command-line option ``option``, checked before any engine
    sees it (SymPy's parser evaluates Python code): a ``.ppy`` expression over numbers, the
    program's variables and parameters and the functions of
    :data:`~prodigy.language.expression_allowlist.MATH_FUNCTIONS`. Returned in the engines' spelling."""
    # pylint: disable=import-outside-toplevel
    from prodigy.language.expression_allowlist import MATH_FUNCTIONS, InadmissibleExpression, admissible_expression
    names = [str(v) for v in program.variables] + [str(p) for p in program.parameters]
    try:
        return admissible_expression(text.strip(), names=names, functions=MATH_FUNCTIONS)
    except InadmissibleExpression as exc:
        raise click.BadParameter(str(exc), param_hint=f"'{option}'") from exc


def initial_distribution(config: ForwardAnalysisConfig, input_dist: Optional[str], program: Program) -> Distribution:
    """The distribution the program starts in: the argument ``INPUT_DIST`` if given, else every variable 0."""
    if input_dist is None:
        return config.factory.one(*program.variables.keys())
    return input_distribution_from_text(config, input_dist, program)


def loop_parts(prog: Program) -> Tuple[List[Instr], WhileInstr, List[Instr]]:
    """The instructions before the only top-level loop, the loop and the instructions after it."""
    loops = [1 if isinstance(instr, WhileInstr) else 0 for instr in prog.instructions]
    if sum(loops) != 1:
        raise ValueError(f"There are {sum(loops)} top-level loops in the program. Exactly 1 is supported.")
    k = loops.index(1)
    loop = prog.instructions[k]
    assert isinstance(loop, WhileInstr)
    return prog.instructions[:k], loop, prog.instructions[k + 1:]
