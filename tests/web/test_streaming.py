"""Coefficients streamed to the GUI while they are computed (#405): the transport through the job and the
partial grids of the joint view. The values themselves are tested in tests/distribution/test_coefficient_stream.py."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from prodigy.web.jobs import Job, merge_stream
from prodigy.web.requests import run_analysis, run_joint

COMPONENTS = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components"
TWO = "x: nat;\ny: nat;\nx = geometric(1/3);\ny = poisson(2);\n"


def test_an_analysis_streams_the_rows_of_its_result():
    events = []
    result = run_analysis({"source": TWO, "syntax": "ppy", "engine": "sympy", "coefficients": 25},
                          progress=events.append)
    assert result["status"] == "ok"
    state = {}
    for event in events:
        assert event["kind"] == "coefficients"
        merge_stream(state, event)
    assert state["variables"] == ["x", "y"] and state["limit"] == 25 and state["bound"] == "exact"
    for variable in ("x", "y"):
        assert state["marginals"][variable]["done"]
        assert state["marginals"][variable]["rows"] == result["distribution"]["marginals"][variable]["rows"]


def test_a_grid_job_streams_its_cells():
    events = []
    request = {"closed_form": "exp(2*y - 2) / (3 - 2*x)", "variables": ["x", "y"], "engine": "sympy", "x": "x",
               "y": "y", "x_window": [0, 3], "y_window": [0, 2], "bound": "exact", "residual_mass": None}
    result = run_joint(request, progress=events.append)
    assert result["status"] == "ok"
    state = {}
    for event in events:
        merge_stream(state, event)
    assert state["grid"]["cells"] == result["joint"]["cells"]


def test_the_stream_is_shown_only_while_the_job_runs():
    job = Job(id="j", process=None, conn=None, time_limit=10, summary={}, state="running")
    merge_stream(job.stream, {"kind": "coefficients", "event": "start", "variables": ["x"], "limit": 3,
                              "bound": "lower", "residual_mass": "1/4", "residual_mass_float": 0.25})
    merge_stream(job.stream, {"kind": "coefficients", "variable": "x", "rows": [{"value": 0}]})
    merge_stream(job.stream, {"kind": "coefficients", "variable": "x", "rows": [{"value": 1}], "done": True})
    assert job.to_json()["stream"]["marginals"]["x"] == {"rows": [{"value": 0}, {"value": 1}], "done": True}
    job.state = "cancelled"
    assert job.to_json()["stream"] is None


def test_partial_grids_keep_cells_out_of_their_window():
    stream = {}
    merge_stream(stream, {"kind": "grid", "cells": [], "header": {"x_values": [0, 1], "y_values": [5]}})
    merge_stream(stream, {"kind": "grid", "cells": [[0, 1, {"prob": "1/2"}], [3, 0, {"prob": "x"}]]})
    assert stream["grid"]["cells"] == [[None, {"prob": "1/2"}]]
    merge_stream(stream, {"kind": "grid", "cells": [[0, 0, {"prob": "1"}]]})
    assert stream["grid"]["cells"] == [[{"prob": "1"}, {"prob": "1/2"}]]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_pending_cells_say_computing(tmp_path):
    """A cell that is still being computed is marked as such in words (status never by colour alone)."""
    script = tmp_path / "check.mjs"
    stream = {"x": "x", "y": "y", "x_values": [0, 1], "y_values": [0], "bound": "lower",
              "cells": [[{"prob": "1/2", "p": 0.5}, None]]}
    script.write_text(
        f"const j = await import({(COMPONENTS / 'joint-view.js').as_uri()!r});\n"
        f"const grid = j.pendingGrid({json.dumps(stream)});\n"
        "console.log(JSON.stringify([grid.cells[0][1].pending, j.cellParts(grid, 0, 1), j.cellParts(grid, 0, 0)]));\n",
        encoding="utf-8")
    output = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(output) == [True, {"where": "x = 1, y = 0", "value": "computing…"},
                                  {"where": "x = 0, y = 0", "value": "at least 0.5 (1/2)"}]
