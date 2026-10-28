"""Tests for the web GUI's JSON API (prodigy web)."""
import time

import pytest

from prodigy.web.app import create_app

HOST = {"Host": "localhost:8080"}

GEOMETRIC = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Pr[c = 0]
"""


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def wait_for(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}", headers=HOST).get_json()
        if status["state"] not in ("starting", "running"):
            return status
        time.sleep(0.1)
    raise AssertionError("analysis did not finish")


def test_page_and_meta(client):
    page = client.get("/", headers=HOST)
    assert page.status_code == 200 and b'id="editor"' in page.data
    meta = client.get("/api/meta", headers=HOST).get_json()
    assert [s["name"] for s in meta["strategies"]][:6] == \
           ["invariant", "unroll", "mass_threshold", "evt_iterate", "evt_invariant", "evt_synthesis"]
    unroll = next(s for s in meta["strategies"] if s["name"] == "unroll")
    assert unroll["params"][0]["name"] == "iterations"
    assert all(p["kind"] != "file" for s in meta["strategies"] for p in s["params"])


def test_check_reports_syntax_errors_with_position(client):
    result = client.post("/api/check", json={"source": "nat x;\nwhile (x < 3 { x := x+1 }"}, headers=HOST).get_json()
    assert not result["ok"]
    error = result["diagnostics"][0]
    assert (error["severity"], error["line"], error["column"]) == ("error", 2, 14)
    assert result["outline"] is None


def test_check_outline_and_type_warnings(client):
    result = client.post("/api/check", json={"source": "nat x;\ny := 2\n" + GEOMETRIC.split("\n", 1)[1]},
                         headers=HOST).get_json()
    assert result["ok"]  # type-checker findings are warnings
    assert result["diagnostics"][0]["severity"] == "warning"
    assert result["diagnostics"][0]["line"] == 2
    assert result["outline"]["loops"][0]["line"] == 4


def test_requests_from_other_sites_are_rejected(client):
    assert client.post("/api/check", data="{}", headers={**HOST, "Content-Type": "text/plain"}).status_code == 415
    assert client.post("/api/check", json={"source": ""},
                       headers={**HOST, "Origin": "http://evil.example"}).status_code == 403
    assert client.post("/api/check", json={"source": ""}, headers={"Host": "evil.example"}).status_code == 403


def test_analysis_job_returns_result(client):
    response = client.post("/api/analyze", json={"source": GEOMETRIC, "engine": "sympy",
                                                 "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 4}}}},
                           headers=HOST)
    assert response.status_code == 202
    status = wait_for(client, response.get_json()["job"])
    assert status["state"] == "done"
    result = status["result"]
    assert result["status"] == "ok" and result["bounds"]["kind"] == "lower"
    assert (result["queries"][0]["lower"], result["queries"][0]["upper"]) == ("1/2", "9/16")


def test_invalid_requests_are_rejected_before_running(client):
    response = client.post("/api/analyze", json={"source": GEOMETRIC,
                                                 "decisions": {"0": "unroll:iterations=many"}}, headers=HOST)
    assert response.status_code == 400
    assert "whole number" in response.get_json()["error"]
    files = client.post("/api/analyze", json={"source": GEOMETRIC, "decisions": {
        "0": {"strategy": "invariant", "params": {"invariant_file": "/etc/passwd"}}}}, headers=HOST)
    status = wait_for(client, files.get_json()["job"])
    assert status["result"]["status"] == "needs_decision"


def test_cancel_and_time_limit(client):
    # does not terminate with probability 1/2, so the threshold 0.9 is never reached
    source = "nat x;\nx := 1\nwhile (x > 0) { {x := x - 1} [1/3] {x := x + 1} }\n?Pr[x = 0]\n"
    request = {"source": source, "decisions": {"0": "mass_threshold:threshold=0.9"}}
    job = client.post("/api/analyze", json=request, headers=HOST).get_json()["job"]
    time.sleep(1.5)
    cancelled = client.delete(f"/api/jobs/{job}", headers=HOST).get_json()
    assert cancelled["state"] == "cancelled"
    job = client.post("/api/analyze", json={**request, "time_limit": 2}, headers=HOST).get_json()["job"]
    status = wait_for(client, job)
    assert status["state"] == "timeout" and status["result"]["status"] == "timeout"


def test_examples(client):
    examples = client.get("/api/examples", headers=HOST).get_json()["examples"]
    ids = {e["id"] for e in examples}
    assert "equivalence/loopy/geometric.pgcl" in ids
    assert not any("/invariants/" in i for i in ids)
    loaded = client.get("/api/examples/equivalence/loopy/geometric.pgcl", headers=HOST).get_json()
    decision = loaded["suggestion"]["decisions"]["0"]
    assert decision["strategy"] == "invariant" and "geometric(1/2)" in decision["params"]["invariant"]
    assert client.get("/api/examples/../../pyproject.toml", headers=HOST).status_code == 404
