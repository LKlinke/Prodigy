"""
`current_session()` without an active session (#195): every call gives a new detached session, so the entry
points of an analysis activate one session for the whole run, and all parts of the run share it.
"""
from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession, current_session, has_current_session
from prodigy.pgcl import parse_pgcl

PROGRAM = """
x: nat;
y: nat;
x = bernoulli(1/2);
observe(x == 1);
y = bernoulli(1/2);
observe(y == 1);
query Pr[x == 1];
query Ex[y];
"""


def test_without_active_session_each_call_is_a_new_detached_session():
    assert not has_current_session()
    first, second = current_session(), current_session()
    assert first is not second
    assert first.detached_session and second.detached_session


def test_active_session_is_the_current_one():
    session = AnalysisSession.detached()
    with session.activate():
        assert has_current_session()
        assert current_session() is session
        inner = AnalysisSession.detached()
        with inner.activate():
            assert current_session() is inner
        assert current_session() is session
    assert not has_current_session()


def test_one_session_for_a_whole_run_without_an_active_session(monkeypatch):
    # compute_discrete_distribution activates a detached session of its own: both observe statements and
    # both queries are booked in it
    seen = []
    original = AnalysisSession.on_normalize

    def on_normalize(self, error_prob):
        seen.append(self)
        return original(self, error_prob)

    monkeypatch.setattr(AnalysisSession, "on_normalize", on_normalize)
    config = ForwardAnalysisConfig()
    program = parse_pgcl(PROGRAM, "ppy")
    compute_discrete_distribution(program, config.factory.one(*program.variables), config)
    assert len(seen) >= 2
    assert all(session is seen[0] for session in seen)
    assert len(seen[0].queries) == 2
    assert not has_current_session()


def test_an_active_session_is_used_for_the_run():
    config = ForwardAnalysisConfig()
    program = parse_pgcl(PROGRAM, "ppy")
    session = AnalysisSession.detached("ppy")
    with session.activate():
        compute_discrete_distribution(program, config.factory.one(*program.variables), config)
    assert len(session.queries) == 2
