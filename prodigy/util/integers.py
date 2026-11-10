"""
Exact rationals of any size as strings.

Since Python 3.11, ``str(int)`` and ``int(str)`` refuse integers with more than 4300 digits (a guard
against quadratic conversions of untrusted text). Prodigy's engines pass exact coefficients as
strings (results, the GiNaC bindings, log messages), and loop-free programs with many rational
probabilities easily have coefficients with more digits (#235). The analysis lifts the limit while it
runs (:func:`unlimited_int_digits`, entered by
:meth:`~prodigy.analysis.config.ForwardAnalysisConfig.activate`); the text it reads from outside has
passed the parser and a size limit before.
"""
import sys
import threading
from contextlib import contextmanager
from typing import Iterator

_lock = threading.Lock()
_depth = 0  # pylint: disable=invalid-name
_saved = 0  # pylint: disable=invalid-name


@contextmanager
def unlimited_int_digits() -> Iterator[None]:
    """
    Lifts Python's limit on the digits of integer string conversions within a ``with`` block.

    The limit is global to the interpreter, so nested and concurrent blocks (threads) share one
    counter: the first block lifts it, the last one restores the previous value.
    """
    global _depth, _saved  # pylint: disable=global-statement
    with _lock:
        if _depth == 0:
            _saved = sys.get_int_max_str_digits()
            sys.set_int_max_str_digits(0)
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                sys.set_int_max_str_digits(_saved)
