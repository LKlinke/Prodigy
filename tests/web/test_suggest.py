"""The "Suggest" button of the GUI: one candidate from a language model (no network: fake providers)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from prodigy.analysis import ai_candidates
from prodigy.analysis.ai_candidates import Provider
from prodigy.web.app import create_app
from prodigy.web.requests import ai_status, run_suggestion

HOST = {"Host": "localhost:8080"}
AI_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "OPENAI_API_KEY", "PRODIGY_AI_BASE_URL",
          "PRODIGY_AI_MODEL", "PRODIGY_AI_PROVIDER", "PRODIGY_AI_TIMEOUT")

GEOMETRIC = """nat x;
nat c;
x := 1;
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Ex[c]
"""

TWO_LOOPS = """nat x;
nat c;
nat d;
x := 1;
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
x := 1;
while (x = 1) { {x := 0} [1/3] {d := d + 1} }
"""


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Tests never see a language model configured on the machine that runs them."""
    for name in AI_ENV:
        monkeypatch.delenv(name, raising=False)


def fake_provider(monkeypatch, answer):
    prompts = []

    def ask(system, user):
        assert "ANSWER:" in system
        prompts.append(user)
        return answer

    monkeypatch.setattr(ai_candidates, "get_provider", lambda env=None: Provider("fake", "fake-model", ask))
    return prompts


def test_candidate_for_the_first_loop(monkeypatch):
    prompts = fake_provider(monkeypatch, "The visiting times are geometric.\nANSWER: (x + 1/2)/(1 - c/2)")
    result = run_suggestion({"source": GEOMETRIC, "loop": 0, "form": "evt_invariant"})
    assert result["status"] == "ok", result
    suggestion = result["suggestion"]
    assert suggestion["candidate"] == "(x + 1/2)/(1 - c/2)"
    assert suggestion["param"] == "invariant" and suggestion["model"] == "fake-model"
    assert suggestion["entering"] == "x" and "ANSWER:" in suggestion["answer"]
    assert "G, the PGF entering the loop: x" in prompts[0] and "expected visiting times" in prompts[0]


def test_earlier_loops_are_analysed_with_their_decisions(monkeypatch):
    prompts = fake_provider(monkeypatch, "ANSWER: nu = (x + 2/3)*c/(1 - d/3)")
    result = run_suggestion({
        "source": TWO_LOOPS, "loop": "1", "form": "evt_subinvariant",
        "decisions": {"0": {"strategy": "evt_invariant", "params": {"invariant": "(x + 1/2)/(1 - c/2)"}},
                      "1": {"strategy": "evt_subinvariant", "params": {"subinvariant": "not parsed"}}}})
    assert result["status"] == "ok", result
    assert result["suggestion"]["candidate"] == "(x + 2/3)*c/(1 - d/3)"
    assert result["suggestion"]["param"] == "subinvariant"
    entering = result["suggestion"]["entering"]
    assert "c" in entering and "x" in entering and "d" not in entering
    assert "occupation sub-invariant" in prompts[0]


def test_untrusted_answer_is_not_a_candidate(monkeypatch):
    fake_provider(monkeypatch, "ANSWER: __import__('os').system('true')")
    result = run_suggestion({"source": GEOMETRIC, "loop": 0, "form": "evt_invariant"})
    assert result["status"] == "error" and result["error"]["type"] == "candidate"
    assert "candidate" not in result["error"]["suggestion"]
    assert "__import__" in result["error"]["suggestion"]["answer"]


def test_failing_earlier_loop_and_bad_requests(monkeypatch):
    fake_provider(monkeypatch, "ANSWER: x")
    failing = run_suggestion({"source": TWO_LOOPS, "loop": 1, "form": "evt_invariant",
                              "decisions": {"0": {"strategy": "evt_invariant", "params": {"invariant": "x"}}}})
    assert failing["status"] == "error" and "entering loop 1" in failing["error"]["message"]
    assert run_suggestion({"source": GEOMETRIC, "loop": 3, "form": "evt_invariant"})["error"]["type"] == "request"
    assert run_suggestion({"source": GEOMETRIC, "loop": 0, "form": "unroll"})["error"]["type"] == "request"
    assert run_suggestion({"source": GEOMETRIC, "loop": True, "form": "evt_invariant"})["error"]["type"] == "request"


def test_no_model_configured():
    assert not ai_status()["available"] and "ANTHROPIC_API_KEY" in ai_status()["reason"]
    result = run_suggestion({"source": GEOMETRIC, "loop": 0, "form": "evt_invariant"})
    assert result["error"]["type"] == "configuration"


# ---- through the API, with the job process and a fake OpenAI-compatible server on localhost ----

class _FakeModel(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # pylint: disable=invalid-name
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append({"path": self.path, "authorization": self.headers.get("authorization"),
                                    "body": body})
        reply = json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": "Geometric.\nANSWER: (x + 1/2)/(1 - c/2)"}}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):  # pylint: disable=arguments-differ
        pass


@pytest.fixture()
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeModel.requests = []
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def client():
    app = create_app(time_limit=60, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def wait_for(client, job_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}", headers=HOST).get_json()
        if status["state"] not in ("starting", "running"):
            return status
        time.sleep(0.1)
    raise AssertionError("suggestion did not finish")


def test_api_without_model(client):
    meta = client.get("/api/meta", headers=HOST).get_json()
    assert meta["ai"]["available"] is False and meta["ai"]["forms"] == ["evt_invariant", "evt_subinvariant"]
    response = client.post("/api/suggest", json={"source": GEOMETRIC, "loop": 0, "form": "evt_invariant"},
                           headers=HOST)
    assert response.status_code == 503 and "ANTHROPIC_API_KEY" in response.get_json()["error"]
    response = client.post("/api/suggest", json={"source": GEOMETRIC, "loop": 0, "form": "invariant"},
                           headers=HOST)
    assert response.status_code == 400


def test_api_suggestion_runs_in_a_job_and_keeps_the_key(client, fake_server, monkeypatch):
    secret = "sk-test-not-for-the-browser"
    monkeypatch.setenv("PRODIGY_AI_BASE_URL", fake_server)
    monkeypatch.setenv("PRODIGY_AI_MODEL", "fake-local")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    meta = client.get("/api/meta", headers=HOST)
    assert meta.get_json()["ai"] == {"available": True, "provider": "openai", "model": "fake-local",
                                     "forms": ["evt_invariant", "evt_subinvariant"]}
    response = client.post("/api/suggest", json={"source": GEOMETRIC, "loop": 0, "form": "evt_invariant",
                                                 "engine": "sympy", "time_limit": 30}, headers=HOST)
    assert response.status_code == 202 and response.get_json()["time_limit"] == 30
    status = wait_for(client, response.get_json()["job"])
    assert status["state"] == "done" and status["summary"]["kind"] == "suggest"
    assert status["result"]["status"] == "ok", status["result"]
    assert status["result"]["suggestion"]["candidate"] == "(x + 1/2)/(1 - c/2)"
    assert _FakeModel.requests[0]["authorization"] == f"Bearer {secret}"
    assert _FakeModel.requests[0]["body"]["model"] == "fake-local"
    assert secret not in meta.get_data(as_text=True) and secret not in json.dumps(status)
