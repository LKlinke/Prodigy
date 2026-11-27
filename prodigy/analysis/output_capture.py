"""
The output of an analysis run: what it prints, also from native code, goes into the session's log.

:func:`captured_output` replaces ``sys.stdout`` by a writer that keeps a copy and forwards every chunk (to the
GUI's stream), and redirects file descriptor 1, to which the GiNaC bindings write directly, into the same log,
so that e.g. ``prodigy analyze --json`` stays clean.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
from typing import Any, Callable, Iterator, Optional

from prodigy.util.logger import log_setup

__all__ = ["captured_output"]

logger = log_setup(str(__name__).rsplit(".", maxsplit=1)[-1], logging.DEBUG)


@contextlib.contextmanager
def captured_output(log: io.StringIO, callback: Optional[Callable[[str], None]]) -> Iterator[None]:
    """Everything printed inside, by Python or by native code, is appended to ``log`` and passed to
    ``callback`` (if given)."""
    writer = _TeeWriter(log, callback)
    with _native_stdout_captured(writer.write), contextlib.redirect_stdout(writer):
        yield


def _flush_c_stdio() -> None:
    try:
        import ctypes  # pylint: disable=import-outside-toplevel
        ctypes.CDLL(None).fflush(None)
    except Exception:  # pylint: disable=broad-except
        # e.g. no C library to load: Python's own buffers are flushed by the caller
        logger.debug("Could not flush the C stdio buffers", exc_info=True)


@contextlib.contextmanager
def _native_stdout_captured(append: Callable[[str], Any]) -> Iterator[None]:
    """Redirects file descriptor 1 into a temporary file and hands its content to ``append``."""
    try:
        target = sys.__stdout__
        descriptor = target.fileno() if target is not None else 1
    except (AttributeError, OSError, ValueError):
        yield
        return
    try:
        if target is not None:
            target.flush()
        _flush_c_stdio()
        saved = os.dup(descriptor)
    except OSError:
        yield
        return
    with tempfile.TemporaryFile() as spool:
        os.dup2(spool.fileno(), descriptor)
        try:
            yield
        finally:
            _flush_c_stdio()
            os.dup2(saved, descriptor)
            os.close(saved)
            spool.seek(0)
            data = spool.read().decode("utf-8", "replace")
            if data:
                append(data)


class _TeeWriter(io.TextIOBase):
    """stdout replacement that keeps a copy and forwards chunks to a callback."""

    def __init__(self, buffer: io.StringIO, callback: Optional[Callable[[str], None]]):
        super().__init__()
        self._buffer = buffer
        self._callback = callback

    def write(self, text: str) -> int:
        self._buffer.write(text)
        if self._callback is not None and text:
            try:
                self._callback(text)
            except Exception:  # pylint: disable=broad-except
                # the output is kept in the buffer; a failing listener (e.g. a closed GUI stream) must not stop the run
                logger.debug("The output callback failed", exc_info=True)
        return len(text)

    def flush(self) -> None:
        pass
