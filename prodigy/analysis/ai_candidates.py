"""
Loop invariants proposed by a language model and checked by Prodigy.

The model only proposes candidates. Every candidate goes through the same exact check as an
invariant typed in by hand (strategies ``evt_invariant`` and ``evt_subinvariant``), so a wrong
answer makes an attempt fail but never makes a result unsound. When a candidate is rejected, the
next request tells the model why (e.g. Φ(I) − I), so it can correct itself.

Answers are untrusted input. Before any computer algebra system sees a candidate (SymPy's parser
evaluates Python code), it must consist of numbers, the program's variables and parameters,
``+ - * / ^``, parentheses, ``sqrt`` and ``exp``, and parse as an expression of the program's
syntax (pGCL or ``.ppy``, see :func:`admissible_expression`).

Configuration, read from the environment of ``prodigy`` (it never reaches the browser):

* ``ANTHROPIC_API_KEY``: Claude API, model ``PRODIGY_AI_MODEL`` (default ``claude-sonnet-5-5``);
  ``ANTHROPIC_WORKSPACE_ID`` for a key that is not scoped to a workspace.
* ``OPENAI_API_KEY`` and/or ``PRODIGY_AI_BASE_URL`` with ``PRODIGY_AI_MODEL``: an OpenAI-compatible
  chat-completions API, e.g. ``http://localhost:11434/v1`` for a local model served by Ollama.
* ``PRODIGY_AI_PROVIDER`` (``anthropic`` or ``openai``) chooses if both are configured;
  ``PRODIGY_AI_TIMEOUT``: seconds per request (default 120).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from prodigy.pgcl.parser import parse_expectation

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2026-10-21"
FUNCTIONS = frozenset({"sqrt", "exp"})
"""Functions a candidate of a language model may use."""
MATH_FUNCTIONS = frozenset({"sqrt", "exp", "log", "sin", "cos", "tan"})
"""Functions a generating function typed in by a person may use (see :func:`admissible_expression`)."""
FORMS = ("evt_invariant", "evt_subinvariant")
TARGET_PARAMS = {"evt_invariant": "invariant", "evt_subinvariant": "subinvariant"}
"""The parameter of each strategy in :data:`FORMS` that takes the candidate."""


class AIConfigurationError(Exception):
    """No language model is configured, or its configuration is incomplete."""


class AIRequestError(Exception):
    """The request to the language model failed."""


class CandidateError(ValueError):
    """An answer contains no admissible candidate."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    ask: Callable[[str, str], str]
    """(system prompt, user prompt) -> text of the answer"""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    """Why the last answer ended (``stop_reason``) and the kinds of its content blocks."""


def _post(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise AIRequestError(f"{url} answered {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise AIRequestError(f"{url} could not be reached: {exc}") from exc


def get_provider(env: Optional[Mapping[str, str]] = None) -> Provider:
    """The configured language model (see the module documentation)."""
    env = os.environ if env is None else env
    choice = env.get("PRODIGY_AI_PROVIDER", "").strip().lower()
    if not choice:
        if env.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif env.get("OPENAI_API_KEY") or env.get("PRODIGY_AI_BASE_URL"):
            choice = "openai"
    timeout = float(env.get("PRODIGY_AI_TIMEOUT") or 120)
    if choice == "anthropic":
        return _anthropic(env, timeout)
    if choice == "openai":
        return _openai(env, timeout)
    if choice:
        raise AIConfigurationError(f"Unknown PRODIGY_AI_PROVIDER {choice!r} (anthropic or openai).")
    raise AIConfigurationError(
        "No language model is configured. Set ANTHROPIC_API_KEY (Claude API), or OPENAI_API_KEY or "
        "PRODIGY_AI_BASE_URL together with PRODIGY_AI_MODEL (OpenAI-compatible API, e.g. a local model) "
        "in the environment of prodigy, and start it again.")


def _anthropic(env: Mapping[str, str], timeout: float) -> Provider:
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        raise AIConfigurationError("PRODIGY_AI_PROVIDER=anthropic needs ANTHROPIC_API_KEY.")
    model = env.get("PRODIGY_AI_MODEL") or DEFAULT_ANTHROPIC_MODEL
    headers = {"authorization": f"Bearer {key}", "anthropic-version": ANTHROPIC_VERSION}
    if env.get("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = env["ANTHROPIC_WORKSPACE_ID"]

    diagnostics: Dict[str, Any] = {}

    def ask(system: str, user: str) -> str:
        # room for the model's reasoning; only the tokens used are billed
        reply = _post(ANTHROPIC_URL, headers, {"model": model, "max_tokens": 16000, "system": system,
                                               "messages": [{"role": "user", "content": user}]}, timeout)
        blocks = reply.get("content") or []
        diagnostics.clear()
        diagnostics.update(stop_reason=reply.get("stop_reason"), content=[b.get("type") for b in blocks])
        return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")

    return Provider("anthropic", model, ask, diagnostics)


def _openai(env: Mapping[str, str], timeout: float) -> Provider:
    model = env.get("PRODIGY_AI_MODEL")
    if not model:
        raise AIConfigurationError("An OpenAI-compatible API needs PRODIGY_AI_MODEL (the name of the model).")
    url = (env.get("PRODIGY_AI_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    headers = {"authorization": f"Bearer {env['OPENAI_API_KEY']}"} if env.get("OPENAI_API_KEY") else {}

    diagnostics: Dict[str, Any] = {}

    def ask(system: str, user: str) -> str:
        reply = _post(url, headers, {"model": model, "messages": [{"role": "system", "content": system},
                                                                   {"role": "user", "content": user}]}, timeout)
        try:
            choice = reply["choices"][0]
            diagnostics.clear()
            diagnostics.update(stop_reason=choice.get("finish_reason"))
            return choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise AIRequestError(f"{url}: unexpected answer {str(reply)[:300]}") from exc

    return Provider("openai", model, ask, diagnostics)


# ---------------------------------------------------------------------------------------
# Prompts and answers
# ---------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You help the verifier Prodigy analyse probabilistic programs with loops. Prodigy represents a "
    "distribution over program states by its probability generating function (PGF) in the program "
    "variables: the coefficient of x^i*c^j is the probability of the state x = i, c = j. You propose a "
    "generating function; Prodigy checks it exactly and rejects it if it is wrong. Your candidate is a "
    "single closed-form expression in the program variables and parameters, using numbers, + - * / ^, "
    "parentheses, sqrt and exp. Keep any reasoning short and end your answer with the line\n"
    "ANSWER: <expression>\n"
    "If you cannot find a candidate, end with the line ANSWER: none")

_TASKS = {
    "evt_invariant": (
        "Find the generating function I of the expected visiting times at the head of the loop: the least "
        "solution of I = G + B(<I>_guard), where G is the PGF entering the loop, <F>_guard keeps the terms "
        "of F whose states satisfy the guard, and B is the PGF transformer of one execution of the loop "
        "body. Then <I>_(not guard) is the output distribution of the loop. I must have non-negative "
        "coefficients; Prodigy accepts it as exact only if I(1, ..., 1) is finite."),
    "evt_subinvariant": (
        "Find an occupation sub-invariant nu: a generating function with non-negative coefficients and "
        "finite mass nu(1, ..., 1) such that nu <= G + B(<nu>_guard) coefficient-wise, where G is the PGF "
        "entering the loop, <F>_guard keeps the terms of F whose states satisfy the guard, and B is the "
        "PGF transformer of one execution of the loop body. Then <nu>_(not guard) is a lower bound on the "
        "output distribution. Larger sub-invariants give tighter bounds: the mass that the lower bound "
        "misses shrinks as nu approaches the expected visiting times, so keep as much of the loop's "
        "behaviour (every branch of the body) as you can. The expected visiting times are the best choice "
        "if their mass is finite; otherwise choose a smaller nu with finite mass."),
}

_EXAMPLE = ("Example: for `nat x; nat c; x := 1; while (x = 1) { {x := 0} [1/2] {c := c + 1} }` the loop "
            "is entered with G = x, and its expected visiting times are (x + 1/2)/(1 - c/2).")


@dataclass(frozen=True)
class Proven:
    """The best sub-invariant Prodigy has proven so far, for the request for a larger one."""
    candidate: str
    mass: str
    """Mass of the lower bound <nu>_(not guard): the loop terminates with at least this probability."""
    missing: str
    """Mass the lower bound misses at most."""


def prompt(form: str, program: str, loop: str, variables: Sequence[str], parameters: Sequence[str],
           entering: str, rejected: Sequence[Tuple[str, str]], proven: Optional[Proven] = None) -> str:
    """The request for a candidate of kind ``form`` (see :data:`FORMS`) for ``loop``.

    ``rejected`` lists earlier candidates with Prodigy's verdict; with ``proven``, the request asks for a
    sub-invariant larger than that one."""
    parts = [f"Program:\n{program}", f"The loop:\n{loop}",
             f"Program variables: {', '.join(variables)}",
             f"Parameters (unknown constants): {', '.join(parameters) if parameters else 'none'}",
             f"G, the PGF entering the loop: {entering}", _TASKS[form], _EXAMPLE]
    if rejected:
        parts.append("These earlier candidates did not help (Prodigy's verdict after each); do not propose "
                     "them again:\n" + "\n".join(f"- {candidate}: {reason}" for candidate, reason in rejected))
    if proven is not None:
        parts.append(f"Prodigy proved the sub-invariant nu = {proven.candidate}. Its lower bound "
                     f"<nu>_(not guard) has mass {proven.mass}, and mass {proven.missing} is not accounted "
                     "for. Find a larger sub-invariant, one whose lower bound has more mass, e.g. by keeping "
                     "branches of the body that nu leaves out. Prodigy checks it exactly and keeps the better "
                     "of the two.")
    return "\n\n".join(parts)


def _state(at: Any) -> str:
    if isinstance(at, Mapping):
        return ", ".join(f"{name} = {value}" for name, value in at.items())
    return str(at)


def _shorten(text: Any, limit: int = 300) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " …"


def describe_rejection(message: str, details: Optional[Mapping[str, Any]]) -> str:
    """Why Prodigy's check rejected a candidate, in words, from the exception's ``message`` and ``details``
    (e.g. "Φ(ν) − ν has the negative coefficient -1/4 at x = 0, c = 1, so ν ≤ Φ(ν) fails.")."""
    details = details or {}
    kind, verdict, condition = details.get("type"), details.get("verdict"), details.get("condition")
    negative = (f"the negative coefficient {details['coefficient']} at {_state(details.get('at'))}"
                if "coefficient" in details else None)
    if kind == "evt_subinvariant":
        if condition == "nonnegative":
            if negative:
                return f"ν has {negative}, but a sub-invariant needs non-negative coefficients."
            return f"Prodigy could not establish that ν has non-negative coefficients ({details.get('method')})."
        if condition == "subinvariance":
            if negative:
                return f"Φ(ν) − ν has {negative}, so ν ≤ Φ(ν) fails there."
            return (f"Prodigy could not establish ν ≤ Φ(ν) coefficient-wise ({details.get('method')}); "
                    f"Φ(ν) − ν = {_shorten(details.get('difference'))}.")
        if condition == "finite_mass":
            return "ν has infinite mass, but a sub-invariant needs finite mass."
    if kind == "evt_invariant":
        difference = details.get("difference")
        if verdict == "refuted" and difference is not None:
            return f"I is not a fixed point: Φ(I) − I = {_shorten(difference)} is not zero."
        if verdict == "unknown":
            reason = details.get("reason") or message
            if difference is not None:
                return f"Not proven: {reason} Φ(I) − I = {_shorten(difference)}."
            return f"Not proven: {reason}"
    return message


_FENCE = re.compile(r"```[A-Za-z]*\n?(.*?)```", re.S)
_ANSWER = re.compile(r"^[\s*_#>]*ANSWER[\s*_]*:[\s*_`$]*(.*?)[\s*_`$]*$", re.I | re.M)
_PREFIX = re.compile(r"^(?:[A-Za-z_]\w*\s*(?:\([^)]*\))?\s*=\s*)+")
_ALLOWED = re.compile(r"[0-9A-Za-z_+\-*/^().\s]+")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_CALL = re.compile(r"([A-Za-z_]\w*)\s*\(")


def to_syntax(expression: str, syntax: str) -> str:
    """``expression`` with the power operator of ``syntax`` (``^`` in pGCL, ``**`` in ``.ppy``)."""
    return expression.replace("^", "**") if syntax == "ppy" else expression.replace("**", "^")


def admissible_expression(text: str, names: Optional[Sequence[str]] = None, syntax: str = "pgcl",
                          functions: AbstractSet[str] = FUNCTIONS) -> str:
    """Checks that ``text`` is a generating function that is safe to give to a computer algebra
    system and returns it in the engines' spelling (powers as ``^``).

    ``text`` must consist of numbers, names, ``+ - * /``, powers (``^`` or ``**``, both
    syntaxes accept both), parentheses and calls of ``functions``, and parse as an expression of
    ``syntax``. With ``names``, every other name is rejected as well.

    Raises :class:`CandidateError` otherwise."""
    text = to_syntax(text, syntax)
    if not _ALLOWED.fullmatch(text):
        raise CandidateError(f"{text!r} contains characters other than numbers, names, "
                             f"+ - * / {'**' if syntax == 'ppy' else '^'} and parentheses.")
    if "//" in text:
        # floor division in .ppy and Python, a comment in pGCL: neither belongs into a generating function
        raise CandidateError(f"{text!r} contains '//', which is not an operation on generating functions.")
    called = sorted({match.group(1) for match in _CALL.finditer(text)} - set(functions))
    if called:
        raise CandidateError(f"{text!r} calls {', '.join(called)}; allowed functions are "
                             f"{', '.join(sorted(functions))}.")
    if any(name.startswith("__") for name in _IDENTIFIER.findall(text)):
        raise CandidateError(f"{text!r} uses a name that starts with '__'.")
    if names is not None:
        unknown = sorted(set(_IDENTIFIER.findall(text)) - set(names) - set(functions))
        if unknown:
            raise CandidateError(f"{text!r} uses {', '.join(unknown)}, which are neither program "
                                 "variables nor parameters.")
    try:
        if syntax == "ppy":
            # pylint: disable=import-outside-toplevel
            from prodigy.pgcl.ppy_parser import parse_ppy_expr
            parse_ppy_expr(text)
        else:
            parse_expectation(text)
    except Exception as exc:  # pylint: disable=broad-except  # lark raises several error types
        raise CandidateError(f"{text!r} is not a well-formed expression: {exc}") from exc
    return text.replace("**", "^")


def candidate_from_answer(answer: str, names: Sequence[str], syntax: str = "pgcl") -> str:
    """The candidate in a model's answer: its last ``ANSWER:`` line, else the last line of its last
    code block or of the answer, written with the power operator of ``syntax`` (``^`` for pGCL,
    ``**`` for ``.ppy``).

    Raises :class:`CandidateError` unless it is an expression over ``names``, numbers,
    ``+ - * / ^``, parentheses, ``sqrt`` and ``exp`` that parses as an expression of ``syntax``
    (:func:`admissible_expression`)."""
    marked = _ANSWER.findall(answer)
    if marked:
        line = marked[-1].strip()
        if line.lower().strip(".") in ("none", "unknown", ""):
            raise CandidateError("The model found no candidate (ANSWER: none).")
    else:
        blocks = _FENCE.findall(answer)
        lines = [line.strip() for line in (blocks[-1] if blocks else answer).splitlines() if line.strip()]
        if not lines:
            raise CandidateError("The answer contains no text.")
        line = lines[-1]
    candidate = to_syntax(_PREFIX.sub("", line.strip("`$ ")).rstrip(".;").strip(), syntax)
    admissible_expression(candidate, names, syntax)
    return candidate
