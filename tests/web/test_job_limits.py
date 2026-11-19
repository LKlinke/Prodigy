"""Limits of the job manager per group: analyses and checks (``max_jobs``) and the background jobs of the
GUI (``max_auxiliary``): grids and suggestions never take the slot of an analysis."""
from typing import Any, List

import pytest

from prodigy.web import jobs as jobs_module
from prodigy.web.app import create_app
from prodigy.web.jobs import BusyError, JobManager, describe_running

from tests.web.test_app import HOST


class FakeConnection:
    def poll(self) -> bool:
        return False

    def close(self) -> None:
        pass


class FakeProcess:
    """A process that runs until it is stopped."""

    def __init__(self, **_: Any) -> None:
        self.alive = False
        self.exitcode = None

    def start(self) -> None:
        self.alive = True

    def terminate(self) -> None:
        self.alive = False

    kill = terminate

    def join(self, timeout: Any = None) -> None:
        pass

    def is_alive(self) -> bool:
        return self.alive


class FakeContext:
    @staticmethod
    def Pipe(duplex: bool = True):  # pylint: disable=invalid-name,unused-argument
        return FakeConnection(), FakeConnection()

    Process = FakeProcess


@pytest.fixture(name="manager")
def fixture_manager():
    manager = JobManager(max_jobs=2, max_auxiliary=2)
    manager._context = FakeContext()  # pylint: disable=protected-access
    yield manager
    manager.shutdown()


def submit(manager: JobManager, kind: str, **request: Any):
    return manager.submit({"source": "", **request}, kind=kind)


def test_background_jobs_do_not_take_the_slots_of_analyses(manager):
    submit(manager, "joint")
    submit(manager, "suggest")
    first = submit(manager, "analysis")
    second = submit(manager, "equivalence")
    assert all(job.running for job in (first, second))
    with pytest.raises(BusyError):
        submit(manager, "analysis")


def test_analyses_do_not_take_the_slots_of_background_jobs(manager):
    submit(manager, "analysis")
    submit(manager, "analysis")
    assert submit(manager, "joint").running and submit(manager, "suggest").running


def test_a_new_grid_replaces_the_oldest_grid(manager):
    old, newer = submit(manager, "joint"), submit(manager, "joint")
    newest = submit(manager, "joint")
    assert old.state == "cancelled" and old.result["status"] == "cancelled"
    assert newer.running and newest.running
    assert [job["id"] for job in manager.running(auxiliary=True)] == [newer.id, newest.id]


def test_a_suggestion_is_refused_when_background_jobs_are_full(manager):
    grids = [submit(manager, "joint"), submit(manager, "joint")]
    with pytest.raises(BusyError) as error:
        submit(manager, "suggest")
    assert "joint grid" in str(error.value) and all(job.running for job in grids)
    assert len(error.value.running) == 2


def test_busy_error_says_what_is_running(manager):
    submit(manager, "analysis", file_name="a.ppy")
    submit(manager, "equivalence")
    with pytest.raises(BusyError) as error:
        submit(manager, "analysis")
    assert str(error.value).startswith("An analysis of a.ppy and an equivalence check are running")
    assert [job["kind"] for job in error.value.running] == ["analysis", "equivalence"]
    assert not any(job["auxiliary"] for job in error.value.running)


def test_describe_running():
    assert describe_running([{"kind": "joint"}]) == "A joint grid is running"
    assert describe_running([{"kind": "analysis", "file_name": "x.ppy"}, {"kind": "joint"}]) == \
        "An analysis of x.ppy and a joint grid are running"


def test_cancel_frees_the_slot(manager):
    jobs: List[Any] = [submit(manager, "analysis"), submit(manager, "analysis")]
    manager.cancel(jobs[0].id)
    assert submit(manager, "analysis").running


def test_abandoned_background_jobs_are_cancelled(manager, monkeypatch):
    grid = submit(manager, "joint")
    analysis = submit(manager, "analysis")
    grid.touched -= jobs_module.STALE_SECONDS + 1
    analysis.touched -= jobs_module.STALE_SECONDS + 1
    monkeypatch.setattr(jobs_module.time, "monotonic", jobs_module.time.monotonic)
    for _ in range(50):
        if not grid.running:
            break
        manager._stop.wait(0.1)  # pylint: disable=protected-access
    assert grid.state == "cancelled" and analysis.running


# -- endpoints ----------------------------------------------------------------------------

@pytest.fixture(name="client")
def fixture_client():
    app = create_app(time_limit=30, max_jobs=1, max_background_jobs=1)
    manager = app.extensions["prodigy_jobs"]
    manager._context = FakeContext()  # pylint: disable=protected-access
    yield app.test_client()
    manager.shutdown()


def test_busy_response_lists_running_jobs_and_they_can_be_cancelled(client):
    body = {"source": "x: nat;\n", "syntax": "ppy", "file_name": "a.ppy"}
    first = client.post("/api/analyze", json=body, headers=HOST)
    assert first.status_code == 202
    busy = client.post("/api/analyze", json=body, headers=HOST)
    assert busy.status_code == 429
    data = busy.get_json()
    assert data["error"].startswith("An analysis of a.ppy is running")
    assert [job["id"] for job in data["running"]] == [first.get_json()["job"]]
    listing = client.get("/api/jobs", headers=HOST).get_json()
    assert [job["kind"] for job in listing["jobs"]] == ["analysis"] and listing["background"] == []
    assert client.delete(f"/api/jobs/{data['running'][0]['id']}", headers=HOST).get_json()["state"] == "cancelled"
    assert client.post("/api/analyze", json=body, headers=HOST).status_code == 202


def test_meta_names_both_limits(client):
    limits = client.get("/api/meta", headers=HOST).get_json()["limits"]
    assert limits["max_jobs"] == 1 and limits["max_background_jobs"] == 1
