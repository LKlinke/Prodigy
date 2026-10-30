"""Candidates from a language model: checked by Prodigy, never trusted (no network: fake providers)."""
from fractions import Fraction

import pytest

from prodigy.analysis import ai_candidates
from prodigy.analysis.ai_candidates import (AIConfigurationError, CandidateError, Provider, candidate_from_answer,
                                            get_provider)
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
    assert ai["rejected"][0]["candidate"] == "x/(1 - c/2)"
    assert "x/(1 - c/2)" in prompts[1] and "G, the PGF entering the loop: x" in prompts[0]
    assert Fraction(result["queries"][0]["value"]) == 1


def test_no_accepted_candidate_fails_the_analysis(monkeypatch):
    fake_provider(monkeypatch, ["__import__('os').system('true')", "x"])
    result = analyse({"strategy": "ai_invariant", "params": {"form": "evt_invariant", "attempts": "2"}})
    assert result["status"] != "ok"
    assert "None of 2 candidates" in str(result.get("error"))


def test_subinvariant_candidates_give_lower_bounds(monkeypatch):
    fake_provider(monkeypatch, ["(x + 1/2)/(1 - c/2)/2"])
    result = analyse("ai_invariant:form=evt_subinvariant")
    assert result["status"] == "ok", result.get("error")
    assert result["loops"][0]["kind"] == "lower"


def test_answers_are_untrusted():
    names = ["x", "c"]
    assert candidate_from_answer("I(x, c) = (x + 1/2)/(1 - c/2).", names) == "(x + 1/2)/(1 - c/2)"
    assert candidate_from_answer("so\n$x**2 + sqrt(1 - c)$", names) == "x^2 + sqrt(1 - c)"
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
        if "anthropic" in url:
            return {"content": [{"type": "text", "text": "x/2"}]}
        return {"choices": [{"message": {"content": "x/3"}}]}

    monkeypatch.setattr(ai_candidates, "_post", post)
    claude = get_provider({"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_WORKSPACE_ID": "ws"})
    assert claude.model == ai_candidates.DEFAULT_ANTHROPIC_MODEL and claude.ask("s", "u") == "x/2"
    url, headers, body = requests[-1]
    assert url == "https://api.anthropic.com/v1/messages" and headers["authorization"] == "Bearer sk-ant-test"
    assert headers["anthropic-workspace-id"] == "ws" and body["system"] == "s"
    local = get_provider({"PRODIGY_AI_BASE_URL": "http://localhost:11434/v1/", "PRODIGY_AI_MODEL": "m"})
    assert local.ask("s", "u") == "x/3"
    url, headers, body = requests[-1]
    assert url == "http://localhost:11434/v1/chat/completions" and "authorization" not in headers
    assert body["messages"][0] == {"role": "system", "content": "s"}
