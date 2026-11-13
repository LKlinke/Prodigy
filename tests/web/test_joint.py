"""The coefficient grid of two variables (issue #197): ``POST /api/jobs/<id>/joint`` and
:mod:`prodigy.analysis.joint_grid`."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.joint_grid import MAX_AXIS, GridError, check_window
from prodigy.web.app import create_app
from prodigy.web.jobs import JobManager
from prodigy.web.requests import joint_request, run_analysis, run_joint

from tests.web.test_app import HOST, wait_for

# x counts the failures before c becomes 0 (geometric), y collects x with probability 1/3 per round
LOOP = """x: nat;
y: nat;
c: nat;
c = 1;
while (c == 1) {
    { c = 0; } [1/2] { x = x + 1; }
    { y = y + x; } [1/3] { skip; }
}
"""

INDEPENDENT = """x: nat;
y: nat;
z: nat;
x = poisson(2);
y = binomial(3, 1/2);
z = bernoulli(1/4);
"""

ENGINES = ["sympy", "ginac", "symengine"]


def _engine(engine):
    if engine == "ginac":
        pytest.importorskip("prodigy_ginac")
    return engine


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=60, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def _analyze(client, source, engine, decisions=None):
    response = client.post("/api/analyze", json={"source": source, "syntax": "ppy", "engine": engine,
                                                 "decisions": decisions or {}}, headers=HOST)
    assert response.status_code == 202
    job = response.get_json()["job"]
    assert wait_for(client, job)["result"]["status"] == "ok"
    return job


def _grid(client, job, **payload):
    response = client.post(f"/api/jobs/{job}/joint", json=payload, headers=HOST)
    assert response.status_code == 202, response.get_json()
    status = wait_for(client, response.get_json()["job"])
    assert status["summary"]["kind"] == "joint"
    return status["result"]


@pytest.mark.parametrize("engine", ENGINES)
def test_grid_of_a_lower_bound_on_every_engine(client, engine):
    """Eight unrollings: every cell is a lower bound, 1/256 is not accounted for, and the cells,
    the mass outside the window and the residual add up to 1."""
    job = _analyze(client, LOOP, _engine(engine), {"0": {"strategy": "unroll", "params": {"iterations": 8}}})
    result = _grid(client, job, x="x", y="y", x_window=[0, 3], y_window=[0, 3])
    assert result["status"] == "ok"
    grid = result["joint"]
    assert (grid["x"], grid["y"], grid["x_values"], grid["y_values"]) == ("x", "y", [0, 1, 2, 3], [0, 1, 2, 3])
    assert [[cell["prob"] for cell in row] for row in grid["cells"][:2]] == [
        ["1/2", "1/9", "1/27", "1/81"], ["0", "1/9", "1/54", "1/162"]]
    assert grid["bound"] == "lower" and grid["residual_mass"] == "1/256"
    assert grid["cells"][0][0]["upper"] == "129/256"
    assert (grid["mass"], grid["shown_mass"], grid["outside_mass"]) == ("255/256", "73/81", "1967/20736")
    assert grid["finite_support"] and grid["support"] == {"x": [0, 7], "y": [0, 35]}
    shown = sum(sympy.Rational(cell["prob"]) for row in grid["cells"] for cell in row)
    assert shown + sympy.Rational(grid["outside_mass"]) + sympy.Rational(grid["residual_mass"]) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_grid_marginalises_the_other_variables_on_every_engine(client, engine):
    """Pr[x = i, y = j] of independent x ~ Poisson(2), y ~ Binomial(3, 1/2) with z marginalised;
    the default window of an infinite distribution starts at 0."""
    job = _analyze(client, INDEPENDENT, _engine(engine))
    grid = _grid(client, job, x="x", y="y")["joint"]
    assert grid["bound"] == "exact" and grid["residual_mass"] is None and not grid["finite_support"]
    assert grid["x_values"] == list(range(16)) and grid["y_values"] == list(range(16))
    for j, binomial in enumerate([sympy.Rational(1, 8), sympy.Rational(3, 8), sympy.Rational(3, 8),
                                  sympy.Rational(1, 8)]):
        for i in range(4):
            expected = binomial * sympy.exp(-2) * 2 ** i / sympy.factorial(i)
            assert sympy.simplify(sympy.sympify(grid["cells"][j][i]["prob"]) - expected) == 0
    assert all(cell["prob"] == "0" for cell in grid["cells"][4])
    swapped = _grid(client, job, x="z", y="x", x_window=[0, 1], y_window=[0, 0])["joint"]
    assert [cell["prob"] for cell in swapped["cells"][0]] == ["3*exp(-2)/4", "exp(-2)/4"]


def test_grid_requests_are_checked(client):
    job = _analyze(client, INDEPENDENT, "sympy")
    for payload, message in [
            ({"x": "x", "y": "x"}, "two different"),
            ({"x": "x", "y": "w"}, "variables of the program"),
            ({"x": "x", "y": "y", "x_window": [0, MAX_AXIS]}, f"At most {MAX_AXIS}"),
            ({"x": "x", "y": "y", "y_window": [3, 2]}, "from ≤ to"),
            ({"x": "x", "y": "y", "y_window": [0, 10 ** 6]}, "from ≤ to"),
            ({"x": "x", "y": "y", "x_window": ["0", "3"]}, "whole numbers"),
            ({"x": "x", "y": "y", "x_window": [True, 3]}, "whole numbers")]:
        response = client.post(f"/api/jobs/{job}/joint", json=payload, headers=HOST)
        assert response.status_code == 400 and message in response.get_json()["error"], payload
    assert client.post("/api/jobs/nope/joint", json={"x": "x", "y": "y"}, headers=HOST).status_code == 404
    grid = client.post(f"/api/jobs/{job}/joint", json={"x": "x", "y": "y"}, headers=HOST).get_json()["job"]
    wait_for(client, grid)
    # a grid is not an analysis that grids can be computed from
    assert client.post(f"/api/jobs/{grid}/joint", json={"x": "x", "y": "y"}, headers=HOST).status_code == 404
    # the distribution never comes from the client
    response = client.post(f"/api/jobs/{job}/joint", json={"x": "x", "y": "y", "closed_form": "__import__('os')"},
                           headers=HOST)
    assert response.status_code == 202
    assert wait_for(client, response.get_json()["job"])["result"]["status"] == "ok"


def test_failed_analyses_have_no_grid(client):
    source = "x: nat;\ny: nat;\nwhile (x == 0) { { x = 1; } [1/2] { y = y + 1; } }\n"
    response = client.post("/api/analyze", json={"source": source, "syntax": "ppy"}, headers=HOST)
    job = response.get_json()["job"]
    assert wait_for(client, job)["result"]["status"] == "needs_decision"
    response = client.post(f"/api/jobs/{job}/joint", json={"x": "x", "y": "y"}, headers=HOST)
    assert response.status_code == 400 and "no output distribution" in response.get_json()["error"]


def test_grid_with_parameters_and_finite_support():
    """Symbolic cells have no float; the default window of a finite distribution is its support."""
    source = "param p: real;\nx: nat;\ny: nat;\n{ x = 2; } [p] { y = 3; }\n"
    result = run_analysis({"source": source, "syntax": "ppy", "engine": "sympy"})
    assert result["status"] == "ok"
    grid = run_joint(joint_request(result, {"x": "x", "y": "y"}))["joint"]
    assert grid["support"] == {"x": [0, 2], "y": [0, 3]}
    assert grid["x_values"] == [0, 1, 2] and grid["y_values"] == [0, 1, 2, 3]
    assert grid["cells"][0][2] == {"prob": "p", "p": None}
    assert grid["cells"][3][0] == {"prob": "1 - p", "p": None}
    assert grid["outside_mass"] == "0"


def test_grid_at_a_position_of_the_timeline():
    """The analysis card's timeline (#311) asks for the grid of a step or of an iterate of a loop
    (#270, #289): the distribution comes from the trace that the server keeps, with the guarantee
    of that position (an iterate is a lower bound: the mass still inside the loop is not accounted
    for)."""
    result = run_analysis({"source": LOOP, "syntax": "ppy", "engine": "sympy", "record_steps": True,
                           "step_iterations": 4, "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 4}}}})
    assert result["status"] == "ok"
    loop_step = next(step["index"] for step in result["steps"] if step["loop"])
    first = run_joint(joint_request(result, {"x": "x", "y": "y", "step": loop_step, "iterate": 0}))["joint"]
    assert first["bound"] == "lower"
    after = run_joint(joint_request(result, {"x": "x", "y": "y", "step": loop_step, "x_window": [0, 3],
                                             "y_window": [0, 3]}))["joint"]
    final = run_joint(joint_request(result, {"x": "x", "y": "y", "x_window": [0, 3], "y_window": [0, 3]}))["joint"]
    # the loop is the last instruction: after it, the trace shows the output distribution
    assert [[c["prob"] for c in row] for row in after["cells"]] == [[c["prob"] for c in row] for row in final["cells"]]
    assert sympy.sympify(first["residual_mass"]) > sympy.sympify(after["residual_mass"])
    for payload, message in [({"step": 99}, "no such step"), ({"step": -1}, "whole number"), ({"step": "1"}, "whole number"),
                             ({"step": True}, "whole number"), ({"iterate": 0}, "needs 'step'"),
                             ({"step": loop_step, "iterate": 99}, "no such iterate")]:
        with pytest.raises(ValueError, match=message):
            joint_request(result, {"x": "x", "y": "y", **payload})
    truncated = json.loads(json.dumps(result))
    truncated["steps"][loop_step]["after"]["truncated"] = True
    with pytest.raises(ValueError, match="too large"):
        joint_request(truncated, {"x": "x", "y": "y", "step": loop_step})


def test_check_window():
    assert check_window(None, "x") is None
    assert check_window([2, 2], "x") == (2, 2)
    with pytest.raises(GridError):
        check_window([0], "x")


def test_grids_are_pruned_before_analyses():
    """Many grids of one result must not push the analysis they come from out of the server."""
    jobs = JobManager(max_jobs=1, keep_finished=2)
    try:
        analysis = jobs.submit({"source": "x: nat;\ny: nat;\n", "syntax": "ppy"})
        while jobs.get(analysis.id).running:
            pass
        assert jobs.get(analysis.id).result["status"] == "ok"
        grids = []
        for _ in range(3):
            grid = jobs.submit(joint_request(analysis.result, {"x": "x", "y": "y"}), kind="joint")
            while jobs.get(grid.id).running:
                pass
            grids.append(grid.id)
        assert jobs.get(analysis.id) is not None
        assert jobs.get(grids[0]) is None and jobs.get(grids[-1]) is not None
    finally:
        jobs.shutdown()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_heatmap_colour_scale(tmp_path):
    """The colour scale of the heatmap (viridis, colour-blind safe): linear from 0 to the largest
    value, logarithmic with both ends labelled (#307), no colour for 0 or unknown values."""
    module = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components" / "joint-view.js"
    script = tmp_path / "check.mjs"
    script.write_text(
        f"const j = await import({str(module.as_uri())!r});\n"
        "const lin = j.colourScale([0, 0.25, 0.5, null], 'linear');\n"
        "const log = j.colourScale([0.001, 0.1, 0], 'log');\n"
        "console.log(JSON.stringify([lin.t(0.5), lin.t(0.25), lin.t(0), lin.t(null), log.t(0.1), log.t(0.001),"
        " log.t(0), j.viridis(0), j.viridis(1), log.ticks[0] === log.min, log.ticks.at(-1) === log.max]));\n",
        encoding="utf-8")
    output = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(output) == [1, 0.5, None, None, 1, 0, None, "#440154", "#fde725", True, True]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_cell_text_of_heatmap_and_3d_view(tmp_path):
    """The heatmap's readout and the 3D view's tooltip (#287) name the values of both variables
    and the probability, with its exact form and, for a lower bound, the upper end."""
    module = Path(__file__).parents[2] / "prodigy" / "web" / "static" / "app" / "components" / "joint-view.js"
    script = tmp_path / "check.mjs"
    grid = {
        "x": "x", "y": "y", "x_values": [0, 1], "y_values": [2], "bound": "lower",
        "cells": [[{"prob": "1/36", "p": 1 / 36, "upper": "0.031684", "upper_p": 0.031684},
                   {"prob": "0", "p": 0}]],
    }
    script.write_text(
        f"const j = await import({str(module.as_uri())!r});\n"
        f"const grid = {json.dumps(grid)};\n"
        "console.log(JSON.stringify([j.cellParts(grid, 0, 0), j.cellParts(grid, 0, 1)]));\n",
        encoding="utf-8")
    output = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60, check=True).stdout
    assert json.loads(output) == [
        {"where": "x = 0, y = 2", "value": "at least 0.0277778 (1/36); at most 0.031684"},
        {"where": "x = 1, y = 2", "value": "0"},
    ]
