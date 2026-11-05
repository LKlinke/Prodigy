"""The .ppy syntax in the web GUI's API: checking, converting, analysing, examples and the
generating functions typed into the setup."""
import pytest

from prodigy.analysis import ai_candidates
from prodigy.analysis.ai_candidates import CandidateError, Provider, admissible_expression, candidate_from_answer
from prodigy.web.app import create_app
from prodigy.web.requests import prepare_request, run_analysis, run_suggestion, validate_request

HOST = {"Host": "localhost:8080"}

GEOMETRIC = """# c counts the tails
x: nat = 1;
c: nat;
while (x == 1) {
    { x = 0; } [1/2] { c = c + 1; }
}
query Pr[c == 0];
query Ex[c];
"""


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


def test_check_ppy_by_syntax_or_file_name(client):
    for extra in ({"syntax": "ppy"}, {"file_name": "geometric.ppy"}):
        result = client.post("/api/check", json={"source": GEOMETRIC, **extra}, headers=HOST).get_json()
        assert result["ok"] and result["syntax"] == "ppy" and result["diagnostics"] == []
        loop = result["outline"]["loops"][0]
        assert (loop["line"], loop["cond"]) == (4, "x == 1")
        assert [q["line"] for q in result["outline"]["queries"]] == [7, 8]
    # the same text is no pGCL program
    assert not client.post("/api/check", json={"source": GEOMETRIC}, headers=HOST).get_json()["ok"]
    response = client.post("/api/check", json={"source": GEOMETRIC, "syntax": "python"}, headers=HOST)
    assert response.status_code == 400


def test_ppy_syntax_errors_have_positions(client):
    result = client.post("/api/check", json={"source": "x: nat;\nx := 1;\n", "syntax": "ppy"},
                         headers=HOST).get_json()
    error = result["diagnostics"][0]
    assert (error["severity"], error["line"], error["column"]) == ("error", 2, 4)
    assert "x = e" in error["message"]


def test_not_implemented_features_are_errors(client):
    source = "x: nat;\nif (x == 0) { y: nat = 1; x = y; }\nx = x // 2;\nquery Ex[x];\n"
    result = client.post("/api/check", json={"source": source, "syntax": "ppy"}, headers=HOST).get_json()
    assert not result["ok"]
    error = result["diagnostics"][0]
    assert error["severity"] == "error" and "not implemented" in error["message"] and error["line"] == 2
    floor = "x: nat;\nx = x // 2;\nquery Ex[x];\n"
    error = client.post("/api/check", json={"source": floor, "syntax": "ppy"}, headers=HOST).get_json()["diagnostics"][0]
    assert (error["severity"], error["line"]) == ("error", 2) and "'//'" in error["message"]


def test_convert(client):
    response = client.post("/api/convert", json={"source": "nat switch;\nswitch := 1 // one\n?Pr[switch = 1]\n"},
                           headers=HOST)
    data = response.get_json()
    assert response.status_code == 200 and data["syntax"] == "ppy"
    assert data["source"] == "switch_: nat;\nswitch_ = 1;  # one\nquery Pr[switch_ == 1];\n"
    assert data["renamings"] == [{"from": "switch", "to": "switch_"}]
    response = client.post("/api/convert", json={"source": "nat x;\nx := := 1"}, headers=HOST)
    assert response.status_code == 422 and response.get_json()["line"] == 2


def test_analysis_of_a_ppy_program(client):
    result = run_analysis({"source": GEOMETRIC, "syntax": "ppy",
                           "decisions": {"0": {"strategy": "unroll", "params": {"iterations": 5}}}})
    assert result["status"] == "ok"
    assert [q["expr"] for q in result["queries"]] == ["c == 0", "c"]
    assert [q["line"] for q in result["queries"]] == [7, 8]
    assert result["loops"][0]["cond"] == "x == 1"
    # an invariant program is written in the program's syntax
    source = "x: nat = 1;\nc: nat;\nwhile (x == 1) { x = 0; c = c + 1; }\nquery Pr[c == 1];\n"
    invariant = "x: nat;\nc: nat;\nif (x == 1) { x = 0; c = c + 1; }\n"
    result = run_analysis({"source": source, "syntax": "ppy",
                           "decisions": {"0": {"strategy": "invariant", "params": {"invariant": invariant}}}})
    assert result["status"] == "ok", result.get("error")
    assert result["queries"][0]["value"] == "1"


@pytest.mark.parametrize("engine", ["sympy", "ginac"])
def test_generating_functions_in_both_spellings(engine):
    pytest.importorskip("prodigy_ginac") if engine == "ginac" else None
    request = {"source": GEOMETRIC, "syntax": "ppy", "engine": engine, "input_distribution": "x**1",
               "decisions": {"0": {"strategy": "evt_invariant", "params": {"invariant": "(x**1 + 1/2)/(1 - c/2)"}}}}
    assert validate_request(request) == []
    prepared = prepare_request(request)
    assert prepared["input_distribution"] == "x^1" and request["input_distribution"] == "x**1"
    result = run_analysis(request)
    assert result["status"] == "ok", result.get("error")
    assert result["queries"][0]["value"] == "1/2"


def test_untrusted_generating_functions_are_rejected():
    for text in ["open(x)", "__import__(x)", "x.real", "'a'", "x // 2", "lambda: x", "exit(0)"]:
        problems = validate_request({"source": GEOMETRIC, "syntax": "ppy", "input_distribution": text})
        assert problems and problems[0].startswith("Initial distribution"), text
        problems = validate_request({"source": "nat x;", "decisions": {
            "*": {"strategy": "evt_invariant", "params": {"invariant": text}}}})
        assert problems and "EVT invariant" in problems[0], text
    assert validate_request({"source": "nat x;", "input_distribution": "exp(x - 1) + sin(x)^2"}) == []
    with pytest.raises(CandidateError, match="calls sqrt"):
        admissible_expression("sqrt(x) ^ 2", syntax="ppy", functions=frozenset())
    with pytest.raises(CandidateError, match="well-formed"):
        admissible_expression("x + * 2", syntax="ppy")


def test_candidates_in_the_program_syntax():
    assert candidate_from_answer("ANSWER: x^2 + sqrt(1 - c)", ["x", "c"], "ppy") == "x**2 + sqrt(1 - c)"
    assert candidate_from_answer("ANSWER: x**2", ["x"], "pgcl") == "x^2"
    assert admissible_expression("x**2 + c^3", syntax="ppy") == "x^2 + c^3"


def test_suggestion_for_a_ppy_program(monkeypatch):
    monkeypatch.setattr(ai_candidates, "get_provider",
                        lambda env=None: Provider("fake", "fake-model", lambda system, user: "ANSWER: 1/(2 - c)^1"))
    result = run_suggestion({"source": GEOMETRIC, "syntax": "ppy", "loop": 0, "form": "evt_invariant"})
    assert result["status"] == "ok", result
    assert result["suggestion"]["candidate"] == "1/(2 - c)**1"


def test_examples_as_ppy(client):
    data = client.get("/api/examples/inference/loop_free/conditioning/monty_hall_nested.pgcl?syntax=ppy",
                      headers=HOST).get_json()
    assert data["syntax"] == "ppy" and data["file_name"] == "monty_hall_nested.ppy"
    assert data["renamings"] == [{"from": "switch", "to": "switch_"}]
    assert client.post("/api/check", json={"source": data["source"], "syntax": "ppy"}, headers=HOST).get_json()["ok"]
    examples = client.get("/api/examples", headers=HOST).get_json()["examples"]
    with_invariant = next(e for e in examples if e["id"].endswith("loopy/dep_bern.pgcl"))
    data = client.get(f"/api/examples/{with_invariant['id']}?syntax=ppy", headers=HOST).get_json()
    invariant = data["suggestion"]["decisions"]["0"]["params"]["invariant"]
    assert "if (c > 0) {" in invariant and ":=" not in invariant
    original = client.get(f"/api/examples/{with_invariant['id']}", headers=HOST).get_json()
    assert original["syntax"] == "pgcl" and original["file_name"] == "dep_bern.pgcl"
    assert client.get(f"/api/examples/{with_invariant['id']}?syntax=java", headers=HOST).status_code == 400


def test_block_comments_keep_positions(client):
    source = "/* while (x == 1) {\n   query Pr[x == 0];\n*/\n" + GEOMETRIC
    result = client.post("/api/check", json={"source": source, "syntax": "ppy"}, headers=HOST).get_json()
    assert result["ok"]
    assert [loop["line"] for loop in result["outline"]["loops"]] == [7]
    assert [q["line"] for q in result["outline"]["queries"]] == [10, 11]
    error = client.post("/api/check", json={"source": "x: nat;\n/* open\nx = 1;", "syntax": "ppy"},
                        headers=HOST).get_json()["diagnostics"][0]
    assert (error["line"], error["column"]) == (2, 1) and "never closed" in error["message"]
