"""Grids of the joint view kept across window changes (#469): :mod:`prodigy.web.grid_cache`, the cells
``known`` to :func:`prodigy.analysis.joint_grid.joint_grid` and the job manager's use of both."""
import pytest

from prodigy.analysis.joint_grid import GRID_HEADER
from prodigy.web.grid_cache import GridCache, result_key
from prodigy.web.jobs import JobManager
from prodigy.web.requests import joint_request, run_analysis, run_joint

from tests.web.test_joint import LOOP

POISSON = "x: nat;\ny: nat;\nx = poisson(2);\ny = iid(binomial(10, 1/2), x);\n"
PARAMETRIC = "param p: real;\nx: nat;\ny: nat;\nx = binomial(6, p);\ny = iid(bernoulli(1/3), x);\n"
# a lower bound: every cell has an upper end (lower end plus the residual mass)
UNROLLED = {"0": {"strategy": "unroll", "params": {"iterations": 6}}}


def _result(source, decisions=None):
    result = run_analysis({"source": source, "syntax": "ppy", "engine": "sympy", "decisions": decisions or {}})
    assert result["status"] == "ok", result.get("error")
    return result


def _grid(result, x_window, y_window, known=None, header=None, **changes):
    request = {**joint_request(result, {"x": "x", "y": "y", "x_window": x_window, "y_window": y_window}), **changes}
    if known is not None:
        request["known_cells"] = [[i, j, cell] for (i, j), cell in known.items()]
    if header is not None:
        request["known_header"] = header
    answer = run_joint(request)
    assert answer["status"] == "ok", answer.get("error")
    return answer["joint"]


def _cells(grid):
    return {(i, j): grid["cells"][r][c] for r, j in enumerate(grid["y_values"]) for c, i in enumerate(grid["x_values"])}


@pytest.mark.parametrize("source, decisions", [(POISSON, None), (PARAMETRIC, None), (LOOP, UNROLLED)],
                         ids=["poisson", "parametric", "lower-bound"])
def test_known_cells_give_the_grid_of_a_new_computation(source, decisions):
    """A window whose cells are all known, or some of them, is the same grid as one computed from scratch:
    cells, masses (the shown mass summed from the known exact values) and guarantee."""
    result = _result(source, decisions)
    first = _grid(result, [0, 5], [0, 5])
    known, header = _cells(first), {name: first[name] for name in GRID_HEADER}
    if decisions:
        assert all(cell.get("upper_source") for cell in known.values())
    for x_window, y_window in [([1, 4], [2, 5]), ([3, 8], [0, 3])]:  # inside the known window, and partly
        assert _grid(result, x_window, y_window, known, header) == _grid(result, x_window, y_window)


def test_a_known_window_needs_no_distribution():
    """With all cells and the header known, the closed form is not even read (it is the expensive part of a large
    one); with a cell missing, it is."""
    result = _result(POISSON)
    first = _grid(result, [0, 3], [0, 3])
    known, header = _cells(first), {name: first[name] for name in GRID_HEADER}
    assert _grid(result, [1, 2], [0, 3], known, header, closed_form="))") == _grid(result, [1, 2], [0, 3])
    answer = run_joint({**joint_request(result, {"x": "x", "y": "y", "x_window": [1, 4], "y_window": [0, 3]}),
                        "known_cells": [[i, j, cell] for (i, j), cell in known.items()], "known_header": header,
                        "closed_form": "))"})
    assert answer["status"] == "error"


def test_known_cells_are_taken_as_they_are():
    """The known cells are not computed again (a marker cell shows it), whether all of the window is known or not."""
    result = _result(POISSON)
    first = _grid(result, [0, 3], [0, 3])
    known, header = _cells(first), {name: first[name] for name in GRID_HEADER}
    marker = {"prob": "1/7", "p": 1 / 7}
    known[(1, 2)] = marker
    assert _grid(result, [0, 3], [0, 3], known, header)["cells"][2][1] == marker
    partly = _grid(result, [0, 5], [0, 3], known)
    assert partly["cells"][2][1] == marker and partly["cells"][0][5]["prob"] == _grid(result, [5, 5], [0, 0])["cells"][0][0]["prob"]


def test_masses_of_unreadable_known_values_are_computed():
    """If an exact value of a known cell cannot be read back, the window is expanded for its masses."""
    result = _result(POISSON)
    fresh = _grid(result, [0, 2], [0, 2])
    known = {key: {**cell, "prob": "))"} for key, cell in _cells(fresh).items()}
    grid = _grid(result, [0, 2], [0, 2], known, {name: fresh[name] for name in GRID_HEADER})
    assert _cells(grid) == known and {**grid, "cells": None} == {**fresh, "cells": None}


def _request(x_window=None, y_window=None, **changes):
    return {"closed_form": "x*y", "bound": "exact", "residual_mass": None, "upper_closed_form": None,
            "engine": "sympy", "variables": ["x", "y"], "x": "x", "y": "y", "x_window": x_window,
            "y_window": y_window, **changes}


def _fake_grid(x_window, y_window):
    xs, ys = list(range(x_window[0], x_window[1] + 1)), list(range(y_window[0], y_window[1] + 1))
    return {"x_values": xs, "y_values": ys, "cells": [[{"prob": f"{i}-{j}"} for i in xs] for j in ys],
            **{name: name for name in GRID_HEADER}}


def test_cache_keeps_grids_and_cells_per_result():
    cache = GridCache()
    grid = _fake_grid((0, 2), (0, 1))
    cache.store(_request(), grid)
    # the default window and the window it turned out to be
    assert cache.grid(_request()) is grid and cache.grid(_request((0, 2), (0, 1))) is grid
    assert cache.grid(_request((0, 1), (0, 1))) is None
    assert sorted((i, j) for i, j, _ in cache.known_cells(_request((1, 5), (1, 3)))) == [(1, 1), (2, 1)]
    assert cache.known_cells(_request()) == []  # the default window is only known to the job
    assert cache.header(_request((7, 9), (7, 9))) == {name: name for name in GRID_HEADER}
    # anything a cell depends on besides the window is part of the key
    for change in [{"closed_form": "x"}, {"bound": "lower"}, {"residual_mass": "1/2"}, {"upper_closed_form": "x"},
                   {"engine": "ginac"}, {"x": "y", "y": "x"}]:
        assert result_key(_request(**change)) != result_key(_request())
        assert cache.grid(_request(**change)) is None and cache.known_cells(_request((0, 2), (0, 1), **change)) == []
        assert cache.header(_request(**change)) is None


def test_cache_is_bounded():
    cache = GridCache(max_results=2, max_cells=5, max_windows=2)
    cache.store(_request(), _fake_grid((0, 2), (0, 1)))  # 6 cells: the oldest goes
    assert len(cache.known_cells(_request((0, 9), (0, 9)))) == 5
    cache.store(_request((5, 5), (5, 5)), _fake_grid((5, 5), (5, 5)))
    windows = [_request(), _request((0, 2), (0, 1)), _request((5, 5), (5, 5))]
    assert [cache.grid(request) is not None for request in windows].count(True) == 2
    assert cache.grid(windows[-1]) is not None
    cache.store(_request(closed_form="a"), _fake_grid((0, 0), (0, 0)))
    cache.store(_request(closed_form="b"), _fake_grid((0, 0), (0, 0)))
    assert cache.known_cells(_request((0, 9), (0, 9))) == []  # the least recently used result went


def test_job_manager_answers_known_windows_from_the_cache():
    jobs = JobManager(max_jobs=1)
    try:
        analysis = jobs.submit({"source": POISSON, "syntax": "ppy"})
        while jobs.get(analysis.id).running:
            pass
        request = joint_request(analysis.result, {"x": "x", "y": "y", "x_window": [0, 4], "y_window": [0, 4]})
        first = jobs.submit(request, kind="joint")
        assert first.grid_request is request and "known_cells" not in request
        while jobs.get(first.id).running:
            pass
        assert first.result["status"] == "ok"
        again = jobs.submit(request, kind="joint")
        assert not again.running and again.process is None and again.summary["cached"]
        assert again.result == first.result and jobs.get(again.id).to_json()["result"] == first.result
        # another window: the job gets the cells of it computed before
        inside = joint_request(analysis.result, {"x": "x", "y": "y", "x_window": [1, 3], "y_window": [2, 6]})
        assert len(jobs.grids.known_cells(inside)) == 3 * 3
        assert jobs.submit(inside, kind="joint").grid_request is inside
    finally:
        jobs.shutdown()
