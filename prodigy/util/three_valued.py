"""
Three-valued verdicts.

A verdict is ``True`` (proven), ``False`` (refuted, with a witness) or ``None`` (undecided), as everywhere in
Prodigy's checks (``CLAUDE.md``, "Principles"). :func:`all_three_valued` is the conjunction of such verdicts.
"""
from typing import Iterable, Optional

__all__ = ["all_three_valued"]


def all_three_valued(verdicts: Iterable[Optional[bool]]) -> Optional[bool]:
    """
    The conjunction of three-valued verdicts: ``False`` if one is ``False`` (a refutation decides on its own),
    ``True`` if all are ``True``, otherwise ``None``. Anything but ``True`` and ``False`` counts as undecided. The
    iterable is consumed lazily and only up to the first ``False``.

    >>> all_three_valued([True, True])
    True
    >>> all_three_valued([True, None, True])
    >>> all_three_valued([None, False, True])
    False
    >>> all_three_valued([])
    True
    """
    verdict: Optional[bool] = True
    for result in verdicts:
        if result is False:
            return False
        if result is not True:
            verdict = None
    return verdict
