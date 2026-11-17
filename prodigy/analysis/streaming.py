"""
Batches of streamed coefficients (#405).

While the coefficients of the output distribution are computed, the GUI receives them in batches through
the progress callback of the analysis (:mod:`prodigy.web.jobs`), at most every :data:`INTERVAL` seconds,
so that the plots build up. A batch is a JSON event with a ``kind``:

* ``"coefficients"``: rows of the marginal of ``variable`` (as in
  :func:`prodigy.analysis.result.univariate_table`), appended to those sent before; ``done`` after the
  last row of the variable. The first event (``"start"``) names the ``variables``, the number of
  coefficients (``limit``) and the guarantee (``bound``, ``residual_mass``) — every streamed coefficient of
  a lower bound is itself a lower bound.
* ``"grid"``: cells ``[r, c, cell]`` of a joint grid (:func:`prodigy.analysis.joint_grid.joint_grid`); the
  first event carries the ``header`` (windows, guarantee).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Mapping, Sequence, Union

INTERVAL = 0.1
"""Seconds between two batches."""

JsonValue = Union[None, bool, int, float, str, Sequence["JsonValue"], Mapping[str, "JsonValue"]]
Event = Dict[str, JsonValue]
Sink = Callable[[Event], None]
"""Where the events go: the progress callback of the analysis."""


class Batcher:
    """Collects the items of one stream and sends them as events ``{**fields, key: [items]}`` to ``sink``
    when :data:`INTERVAL` seconds have passed since the last batch, and on :meth:`flush`."""

    def __init__(self, sink: Sink, key: str, fields: Event, *, clock: Callable[[], float] = time.monotonic,
                 interval: float = INTERVAL) -> None:
        self._sink = sink
        self._key = key
        self._fields = fields
        self._clock = clock
        self._interval = interval
        self._items: List[JsonValue] = []
        self._last = clock()

    def add(self, item: JsonValue) -> None:
        self._items.append(item)
        if self._clock() - self._last >= self._interval:
            self.flush()

    def flush(self, **extra: JsonValue) -> None:
        if self._items or extra:
            self._sink({**self._fields, self._key: self._items, **extra})
            self._items = []
        self._last = self._clock()
