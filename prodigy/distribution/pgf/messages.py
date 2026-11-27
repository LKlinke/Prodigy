"""
Messages of the shared logic of generating functions (#511): the temporary variables of an update spelled as the
program wrote their terms.
"""
from __future__ import annotations

import re
from typing import Dict

from prodigy.language.engine_text import engine_text_of
from prodigy.distribution.pgf.rules.independent_comparison import ComparisonNotImplemented


def in_programs_terms(error: Exception, names: Dict[str, str]) -> Exception:
    """
    A copy of `error` with the temporary variables of an operator update (`_0`, `sym_1`, `x_2`: engine
    internals) replaced by the operands as the program wrote them (#349). `names` maps temporary to
    program text, in parentheses unless it is a name or a number (:func:`with_terms`); an unknown or
    unchanged message returns `error` itself.
    """
    if not names:
        return error
    message = with_terms(engine_text_of(error), names)
    if message == engine_text_of(error):
        return error
    if isinstance(error, ComparisonNotImplemented):
        return ComparisonNotImplemented(with_terms(error.condition, names), error.reason)
    return type(error)(message)


def with_terms(text: str, terms: Dict[str, str]) -> str:
    """
    `text` with every temporary variable of `terms` (temporary variable to the term as the program wrote it,
    #425) substituted by its term, in parentheses unless the term is a name or a number, so that the closed
    form of a distribution reads in the terms of the program (#221).
    """
    if not terms:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in terms) + r")\b")
    return pattern.sub(lambda match: terms[match.group(1)] if re.fullmatch(r"\w+", terms[match.group(1)])
                       else f"({terms[match.group(1)]})", text)
