"""Candidates from a language model: checked by Prodigy, never trusted (no network: fake providers)."""
from fractions import Fraction

import pytest

from prodigy.analysis import ai_candidates
from prodigy.analysis.ai_candidates import (AIConfigurationError, AIRequestError, CandidateError, Provider,
                                            candidate_from_answer, describe_rejection, get_provider)
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl import parse_pgcl

GEOMETRIC = """nat x;
nat c;
x := 1;
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Ex[c]
"""


def fake_provider(monkeypatch, answers):
    prompts = []

    def ask(system, user):
        assert "generating function" in system
        prompts.append(user)
        return answers[len(prompts) - 1]

    monkeypatch.setattr(ai_candidates, "get_provider", lambda env=None: Provider("fake", "fake-model", ask))
    return prompts


def analyse(decision):
    session = AnalysisSession(parse_pgcl(GEOMETRIC), source=GEOMETRIC, config=ForwardAnalysisConfig(),
                              decisions={"0": decision}, capture_output=True)
    return session.run().to_json()


def test_rejected_candidate_goes_back_to_the_model_and_the_checked_one_is_used(monkeypatch):
    prompts = fake_provider(monkeypatch, ["I = x/(1 - c/2)", "Here it is:\n```\n(x + 1/2)/(1 - c/2)\n```"])
    result = analyse("ai_invariant:form=evt_invariant")
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == "exact"
    ai = loop["certificates"][0]["ai"]
    assert ai["model"] == "fake-model" and ai["attempts"] == 2 and ai["candidate"] == "(x + 1/2)/(1 - c/2)"
    assert ai["history"][0]["candidate"] == "x/(1 - c/2)" and ai["history"][0]["verdict"] == "refuted"
    assert ai["history"][0]["reason"].startswith("I is not a fixed point: Φ(I) − I = ")
    assert "x/(1 - c/2)" in prompts[1] and "G, the PGF entering the loop: x" in prompts[0]
    assert Fraction(result["queries"][0]["value"]) == 1


def test_no_accepted_candidate_fails_the_analysis(monkeypatch):
    fake_provider(monkeypatch, ["__import__('os').system('true')", "x"])
    result = analyse({"strategy": "ai_invariant", "params": {"form": "evt_invariant", "attempts": "2"}})
    assert result["status"] != "ok"
    assert "None of 2 answers" in str(result.get("error"))
    log = result["loops"][0]["certificates"][0]
    assert log["type"] == "ai_attempts" and [a["attempt"] for a in log["attempts"]] == [1, 2]
    assert log["attempts"][0]["candidate"] is None and "__import__" in log["attempts"][0]["answer"]


def test_subinvariant_candidates_give_lower_bounds(monkeypatch):
    prompts = fake_provider(monkeypatch, ["(x + 1/2)/(1 - c/2)/2"])
    result = analyse({"strategy": "ai_invariant", "params": {"form": "evt_subinvariant", "attempts": "1"}})
    assert result["status"] == "ok", result.get("error")
    assert result["loops"][0]["kind"] == "lower"
    assert "Larger sub-invariants give tighter bounds" in prompts[0]


def test_a_proven_subinvariant_is_improved_and_the_largest_is_kept(monkeypatch):
    prompts = fake_provider(monkeypatch, ["(x + 1/2)/(1 - c/2)/2", "x + 2*c", "(x + 1/2)/(1 - c/2)/4",
                                          "(x + 1/2)/(1 - c/2)/(4/3)", "(x + 1/2)/(1 - c/2)/3"])
    result = analyse({"strategy": "ai_invariant", "params": {"form": "evt_subinvariant", "attempts": "5"}})
    assert result["status"] == "ok", result.get("error")
    loop = result["loops"][0]
    assert loop["kind"] == "lower" and loop["residual_mass"] == "1/4"
    ai = loop["certificates"][0]["ai"]
    assert ai["attempts"] == 4 and ai["asked"] == 5 and ai["candidate"] == "(x + 1/2)/(1 - c/2)/(4/3)"
    assert [a["verdict"] for a in ai["history"]] == ["proven", "refuted", "not_larger", "not_larger"]
    assert ai["history"][0]["mass"] == "1/2" and ai["history"][0]["missing_mass"] == "1/2"
    # the mass reached goes back to the model, together with the full details of a rejection
    assert "Prodigy proved the sub-invariant" not in prompts[0]
    assert "nu = (x + 1/2)/(1 - c/2)/2" in prompts[1] and "has mass 1/2, and mass 1/2 is not accounted" in prompts[1]
    assert '"coefficient": "-2"' in prompts[2] and "Φ(ν) − ν has the negative coefficient -2 at x = 0, c = 1" in prompts[2]
    assert "nu = (x + 1/2)/(1 - c/2)/(4/3)" in prompts[4] and "not more than 1/2 (attempt 1)" in prompts[4]
    # the GUI gets the reason in words; the details stay available
    assert ai["history"][1]["reason"] == "Φ(ν) − ν has the negative coefficient -2 at x = 0, c = 1, so ν ≤ Φ(ν) fails there."
    assert ai["history"][1]["details"]["at"] == {"x": 0, "c": 1}
    assert "largest of 4 proven sub-invariants" in loop["messages"][0]


def test_an_exact_subinvariant_ends_the_search(monkeypatch):
    prompts = fake_provider(monkeypatch, ["(x + 1/2)/(1 - c/2)/2", "(x + 1/2)/(1 - c/2)", "x"])
    result = analyse({"strategy": "ai_invariant", "params": {"form": "evt_subinvariant", "attempts": "3"}})
    assert result["loops"][0]["kind"] == "exact" and len(prompts) == 2
    assert Fraction(result["queries"][0]["value"]) == 1


def test_a_failed_request_keeps_the_proven_subinvariant(monkeypatch):
    calls = []

    def ask(system, user):
        calls.append(user)
        if len(calls) > 1:
            raise AIRequestError("rate limited")
        return "(x + 1/2)/(1 - c/2)/2"

    monkeypatch.setattr(ai_candidates, "get_provider", lambda env=None: Provider("fake", "fake-model", ask))
    result = analyse({"strategy": "ai_invariant", "params": {"form": "evt_subinvariant", "attempts": "3"}})
    assert result["status"] == "ok", result.get("error")
    ai = result["loops"][0]["certificates"][0]["ai"]
    assert result["loops"][0]["kind"] == "lower" and ai["attempts"] == 1 and len(calls) == 2
    assert ai["history"] == [{"attempt": 2, "answer": "", "candidate": None, "verdict": "error", "reason": "rate limited"}]


def test_rejections_in_words():
    negative = {"coefficient": "-1/4", "at": {"x": 0, "c": 1}, "verdict": "refuted"}
    assert describe_rejection("m", {"type": "evt_subinvariant", "condition": "nonnegative", **negative}) \
        == "ν has the negative coefficient -1/4 at x = 0, c = 1, but a sub-invariant needs non-negative coefficients."
    assert describe_rejection("m", {"type": "evt_subinvariant", "condition": "subinvariance", "verdict": "unknown",
                                    "method": "positivity heuristic of Prodigy", "difference": "x - c"}) \
        == "Prodigy could not establish ν ≤ Φ(ν) coefficient-wise (positivity heuristic of Prodigy); Φ(ν) − ν = x - c."
    assert "infinite mass" in describe_rejection("m", {"type": "evt_subinvariant", "condition": "finite_mass"})
    assert describe_rejection("m", {"type": "evt_invariant", "verdict": "unknown", "reason": "Not decided.",
                                    "difference": "x"}) == "Not proven: Not decided. Φ(I) − I = x."
    assert describe_rejection("Something else.", None) == "Something else."
    assert describe_rejection("m", {"type": "evt_invariant", "verdict": "refuted", "difference": "c" * 400}).endswith(" … is not zero.")


def test_answers_are_untrusted():
    names = ["x", "c"]
    assert candidate_from_answer("I(x, c) = (x + 1/2)/(1 - c/2).", names) == "(x + 1/2)/(1 - c/2)"
    assert candidate_from_answer("so\n$x**2 + sqrt(1 - c)$", names) == "x^2 + sqrt(1 - c)"
    assert candidate_from_answer("Reasoning.\n**ANSWER:** `(x + 1/2)/(1 - c/2)`\nThis is exact.", names) \
        == "(x + 1/2)/(1 - c/2)"
    with pytest.raises(CandidateError, match="no candidate"):
        candidate_from_answer("I do not know.\nANSWER: none", names)
    for answer in ["__import__('os').system('ls')", "x + y", "lambda: x", "(x + 1", "", "x.real"]:
        with pytest.raises(CandidateError):
            candidate_from_answer(answer, names)


def test_configuration(monkeypatch):
    with pytest.raises(AIConfigurationError, match="ANTHROPIC_API_KEY"):
        get_provider({})
    with pytest.raises(AIConfigurationError, match="PRODIGY_AI_MODEL"):
        get_provider({"PRODIGY_AI_BASE_URL": "http://localhost:11434/v1"})
    requests = []

    def post(url, headers, body, timeout):
        requests.append((url, headers, body))
        if "anthropic" in url and body["messages"][0]["content"] == "think":
            return {"content": [{"type": "thinking", "thinking": "..."}], "stop_reason": "max_tokens"}
        if "anthropic" in url:
            return {"content": [{"type": "text", "text": "x/2"}], "stop_reason": "end_turn"}
        return {"choices": [{"message": {"content": "x/3"}}]}

    monkeypatch.setattr(ai_candidates, "_post", post)
    claude = get_provider({"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_WORKSPACE_ID": "ws"})
    assert claude.model == ai_candidates.DEFAULT_ANTHROPIC_MODEL and claude.ask("s", "u") == "x/2"
    url, headers, body = requests[-1]
    assert url == "https://api.anthropic.com/v1/messages" and headers["authorization"] == "Bearer sk-ant-test"
    assert headers["anthropic-workspace-id"] == "ws" and body["system"] == "s"
    assert claude.ask("s", "think") == "" and claude.diagnostics == {"stop_reason": "max_tokens", "content": ["thinking"]}
    local = get_provider({"PRODIGY_AI_BASE_URL": "http://localhost:11434/v1/", "PRODIGY_AI_MODEL": "m"})
    assert local.ask("s", "u") == "x/3"
    url, headers, body = requests[-1]
    assert url == "http://localhost:11434/v1/chat/completions" and "authorization" not in headers
    assert body["messages"][0] == {"role": "system", "content": "s"}
