"""A time limit that interrupts a block of (pure Python) computer algebra, e.g. a SymPy solver call."""
import contextlib
import signal
import threading
import time
from typing import Any, Callable, Iterator, Optional


class TimeLimitReached(BaseException):
    """Raised by the alarm of :func:`time_limit`. A ``BaseException`` so that no ``except Exception``
    inside SymPy or a heuristic swallows it."""


@contextlib.contextmanager
def time_limit(seconds: Optional[float],
               exception: type = TimeLimitReached) -> Iterator[Callable[[], bool]]:
    """Interrupts the block with ``exception`` (a :class:`TimeLimitReached`) after ``seconds`` (``None``:
    no limit).

    Yields a function that tells whether the time is up, for checks between steps. The interruption
    uses ``SIGALRM`` and therefore only works in the main thread on POSIX systems; elsewhere only these
    checks stop the block (after the step that is running). An alarm that is already set and fires
    earlier (e.g. the timeout of a test) is kept, a later one is restored afterwards."""
    deadline = None if seconds is None else time.monotonic() + seconds

    def expired() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    if seconds is None or not hasattr(signal, "setitimer") \
            or threading.current_thread() is not threading.main_thread():
        yield expired
        return
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    if old_delay and old_delay <= seconds:
        yield expired
        return
    armed = [True]

    def on_alarm(_signum: int, _frame: Any) -> None:
        if armed[0]:
            raise exception()

    start = time.monotonic()
    old_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield expired
    finally:
        armed[0] = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay:
            signal.setitimer(signal.ITIMER_REAL, max(old_delay - (time.monotonic() - start), 1e-3), old_interval)
