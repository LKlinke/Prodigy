"""
Grids of the joint view kept across window changes (#469).

A grid job (:func:`prodigy.web.requests.run_joint`) computes the cells of one window in a new process.
When the user pans, zooms, swaps back or returns to a position of the timeline, most of these cells were
computed before. :class:`GridCache` keeps them in the server process, keyed by everything a cell depends
on except the window: the closed form of the distribution it comes from, its guarantee (``bound``,
``residual_mass``, the proven upper bound ``upper_closed_form``), the engine, the variables and the pair.
A cell is then identified by its values ``(i, j)``. Its JSON is a function of these (see
:func:`prodigy.analysis.joint_grid.joint_grid`), so a kept cell is exactly the cell a new job would
compute.

* A window that was shown before is answered from the cache at once, without a job process.
* For another window, the kept cells of that window go to the job (``known_cells``) with the part of a
  grid that does not depend on the window (``known_header``). If every cell is known, the job needs
  neither the distribution nor an expansion (:func:`~prodigy.analysis.joint_grid.grid_from_known`);
  otherwise it computes the window and takes the known cells as they are.

The cache is bounded: the grids of the most recently used :data:`MAX_RESULTS` results, at most
:data:`MAX_CELLS` cells and :data:`MAX_WINDOWS` whole grids each.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from prodigy.analysis.joint_grid import GRID_HEADER

MAX_RESULTS = 16
"""Results (closed form, guarantee, engine and pair) whose cells are kept."""
MAX_CELLS = 20_000
"""Cells kept per result; the oldest go first."""
MAX_WINDOWS = 8
"""Whole grids (windows) kept per result."""

_KEY_FIELDS = ("closed_form", "bound", "residual_mass", "upper_closed_form", "engine", "variables", "x", "y")

Window = Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]


def result_key(request: Dict[str, Any]) -> str:
    """What the cells of a grid request depend on, except the window, as one string."""
    return json.dumps([request.get(name) for name in _KEY_FIELDS], sort_keys=True, default=str)


def _window(request: Dict[str, Any]) -> Window:
    def pair(value: Any) -> Optional[Tuple[int, int]]:
        return None if value is None else (int(value[0]), int(value[1]))
    return pair(request.get("x_window")), pair(request.get("y_window"))


class _Entry:
    def __init__(self) -> None:
        self.cells: "OrderedDict[Tuple[int, int], Dict[str, Any]]" = OrderedDict()
        self.grids: "OrderedDict[Window, Dict[str, Any]]" = OrderedDict()
        self.header: Optional[Dict[str, Any]] = None


class GridCache:
    """The cells and grids computed for the joint view (see the module documentation); thread-safe."""

    def __init__(self, max_results: int = MAX_RESULTS, max_cells: int = MAX_CELLS,
                 max_windows: int = MAX_WINDOWS) -> None:
        self.max_results = max_results
        self.max_cells = max_cells
        self.max_windows = max_windows
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._lock = threading.Lock()

    def _entry(self, request: Dict[str, Any], create: bool = False) -> Optional[_Entry]:
        key = result_key(request)
        entry = self._entries.get(key)
        if entry is None and create:
            entry = self._entries[key] = _Entry()
            while len(self._entries) > self.max_results:
                self._entries.popitem(last=False)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def grid(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The whole grid of the request's window if it was computed before (the default window when the
        request names none), else None."""
        with self._lock:
            entry = self._entry(request)
            if entry is None:
                return None
            grid = entry.grids.get(_window(request))
            if grid is not None:
                entry.grids.move_to_end(_window(request))
            return grid

    def known_cells(self, request: Dict[str, Any]) -> List[List[Any]]:
        """The kept cells of the request's window as ``[i, j, cell]`` (none for the default window, which only
        the job knows)."""
        x_window, y_window = _window(request)
        if x_window is None or y_window is None:
            return []
        with self._lock:
            entry = self._entry(request)
            if entry is None:
                return []
            return [[i, j, cell] for (i, j), cell in entry.cells.items()
                    if x_window[0] <= i <= x_window[1] and y_window[0] <= j <= y_window[1]]

    def header(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The fields of the grids of the request's result that do not depend on the window, if one was kept."""
        with self._lock:
            entry = self._entry(request)
            return None if entry is None else entry.header

    def store(self, request: Dict[str, Any], grid: Dict[str, Any]) -> None:
        """Keeps a computed grid of ``request``: under the requested window and its actual one, and its cells."""
        x_values, y_values = grid.get("x_values") or [], grid.get("y_values") or []
        with self._lock:
            entry = self._entry(request, create=True)
            assert entry is not None
            if all(name in grid for name in GRID_HEADER):
                entry.header = {name: grid[name] for name in GRID_HEADER}
            windows = {_window(request)}
            if x_values and y_values:
                windows.add(((x_values[0], x_values[-1]), (y_values[0], y_values[-1])))
            for window in windows:
                entry.grids[window] = grid
                entry.grids.move_to_end(window)
            while len(entry.grids) > self.max_windows:
                entry.grids.popitem(last=False)
            for r, row in enumerate(grid.get("cells") or []):
                for c, cell in enumerate(row):
                    if cell is not None and c < len(x_values) and r < len(y_values):
                        entry.cells[(x_values[c], y_values[r])] = cell
                        entry.cells.move_to_end((x_values[c], y_values[r]))
            while len(entry.cells) > self.max_cells:
                entry.cells.popitem(last=False)
