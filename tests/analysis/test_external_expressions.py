"""Generating functions given from outside (CLI options, loop decisions, the GUI) are checked before any engine
sees them (#304): SymPy's ``sympify``/``parse_expr`` evaluate Python code."""
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from prodigy.analysis.ai_candidates import MATH_FUNCTIONS, CandidateError, admissible_expression
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound, get_loop_strategy
from prodigy.analysis.session import AnalysisSession, InteractiveDecisions, parse_decisions, validate_decisions
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl

GEOMETRIC = """nat x; nat c;
x := 1
while (x = 1) { {x := 0} [1/2] {c := c + 1} }
?Pr[c = 0]
"""
INVARIANT = "(x + 1/2)/(1 - c/2)"
SUBINVARIANT = "x/2"


def _malicious(tmp_path):
    """Strings that run code (or reach Python objects) if SymPy's parser sees them; the first ones create
    ``tmp_path / "pwned"``."""
    marker = tmp_path / "pwned"
    return [
        f"__import__('pathlib').Path({str(marker)!r}).touch()",
        f"exec(\"open({str(marker)!r}, 'w')\")",
        "__import__",
        "x.__class__.__mro__",
        "x.conjugate()",
        "x.real",
        "(lambda: 1)()",
        "lambda x: x",
        "[x for x in (1, 2)]",
        "eval(x)",
        "Symbol(x)",
        "x if c else 1",
        "not x",
        "x and c",
        "1 // 2",
    ]


def _assert_untouched(tmp_path):
    assert not (tmp_path / "pwned").exists(), "a malicious expression was evaluated"


@pytest.mark.parametrize("syntax", ["pgcl", "ppy"])
def test_expression_parameters_refuse_malicious_text(tmp_path, syntax):
    spec = get_loop_strategy("evt_invariant").param("invariant")
    for text in _malicious(tmp_path):
        with pytest.raises(ValueError):
            spec.parse(text, syntax)
    _assert_untouched(tmp_path)


@pytest.mark.parametrize("syntax", ["pgcl", "ppy"])
@pytest.mark.parametrize("text, spelled", [
    ("(x + 1/2)/(1 - c/2)", "(x + 1/2)/(1 - c/2)"),
    ("x**2/2 + 1/2", "x^2/2 + 1/2"),
    ("x^2/2 + 1/2", "x^2/2 + 1/2"),
    ("exp(x - 1)", "exp(x - 1)"),
    (" sqrt(1/4)*x + 0.5 ", "sqrt(1/4)*x + 0.5"),
])
def test_expression_parameters_accept_generating_functions(syntax, text, spelled):
    assert get_loop_strategy("evt_subinvariant").param("subinvariant").parse(text, syntax) == spelled


def test_boolean_words_are_not_generating_functions():
    for text in ["not x", "x and c", "x or c", "true", "False*x"]:
        for syntax in ("pgcl", "ppy"):
            with pytest.raises(CandidateError, match="Boolean"):
                admissible_expression(text, syntax=syntax, functions=MATH_FUNCTIONS)


def test_validate_decisions_reports_malicious_expressions(tmp_path):
    for text in _malicious(tmp_path):
        decisions = parse_decisions({"0": {"strategy": "evt_invariant", "params": {"invariant": text}},
                                     "*": {"strategy": "evt_subinvariant", "params": {"subinvariant": text}}})
        problems = validate_decisions(decisions, syntax="ppy")
        assert len(problems) == 2, problems
        assert problems[0].startswith("loop 0: EVT invariant I: ")
        assert problems[1].startswith("default decision: EVT sub-invariant nu: ")
    _assert_untouched(tmp_path)
    good = parse_decisions({"0": {"strategy": "evt_invariant", "params": {"invariant": "x**2"}}})
    assert validate_decisions(good, syntax="ppy") == []


@pytest.mark.parametrize("engine", list(ForwardAnalysisConfig.Engine), ids=lambda e: e.name.lower())
@pytest.mark.parametrize("strategy, param", [("evt_invariant", "invariant"), ("evt_subinvariant", "subinvariant")])
def test_session_decisions_from_python_are_checked(tmp_path, engine, strategy, param):
    """An ``AnalysisSession`` built from Python reports a malicious decision as an error naming the parameter."""
    for text in _malicious(tmp_path)[:4]:
        session = AnalysisSession(parse_pgcl(GEOMETRIC), source=GEOMETRIC, config=ForwardAnalysisConfig(engine=engine),
                                  decisions={"0": {"strategy": strategy, "params": {param: text}}},
                                  capture_output=True)
        result = session.run().to_json()
        assert result["status"] == "error", result
        assert f"parameter {param!r} of strategy {strategy}" in result["error"]["message"]
    _assert_untouched(tmp_path)


@pytest.mark.parametrize("engine", list(ForwardAnalysisConfig.Engine), ids=lambda e: e.name.lower())
def test_session_decisions_accept_valid_expressions(engine):
    for invariant in (INVARIANT, "(x + 1/2)/(1 - c/2)".replace("/", " / ")):
        session = AnalysisSession(parse_pgcl(GEOMETRIC), source=GEOMETRIC, config=ForwardAnalysisConfig(engine=engine),
                                  decisions={"0": {"strategy": "evt_invariant", "params": {"invariant": invariant}}},
                                  capture_output=True)
        result = session.run().to_json()
        assert result["status"] == "ok", result.get("error")
        assert result["bounds"]["kind"] == Bound.EXACT


def test_interactive_answers_are_checked(tmp_path, monkeypatch, capsys):
    """The interactive CLI asks again after a malicious answer."""
    answers = iter(_malicious(tmp_path)[:3] + [INVARIANT])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    strategy = get_loop_strategy("evt_invariant")
    task = SimpleNamespace(strategy=strategy)
    value = InteractiveDecisions().ask_param(AnalysisSession.detached(), task, strategy.param("invariant"))
    assert value == INVARIANT
    assert capsys.readouterr().out.count("Invalid input") == 3
    _assert_untouched(tmp_path)


@pytest.fixture(name="program")
def _program(tmp_path):
    path = tmp_path / "program.pgcl"
    path.write_text(GEOMETRIC)
    return path


@pytest.mark.parametrize("command, invariant", [("invariant_synthesis", INVARIANT),
                                                ("evt_lower_bound", SUBINVARIANT)])
def test_cli_invariant_option_is_checked(tmp_path, program, command, invariant):
    runner = CliRunner()
    for text in _malicious(tmp_path) + ["y + 1"]:
        refused = runner.invoke(cli, [command, str(program), "--invariant", text])
        assert refused.exit_code == 2, refused.output
        assert "Invalid value for '--invariant'" in refused.output
        assert "Traceback" not in refused.output
    _assert_untouched(tmp_path)
    accepted = runner.invoke(cli, [command, str(program), "--invariant", invariant])
    assert accepted.exit_code == 0, accepted.output


@pytest.mark.parametrize("command", ["analyze", "main"])
def test_cli_loop_option_is_checked(tmp_path, program, command):
    runner = CliRunner()
    for text in _malicious(tmp_path):
        refused = runner.invoke(cli, [command, str(program), "--loop", f"0=evt_invariant:invariant={text}"])
        assert refused.exit_code == 2, refused.output
        # texts with ':', ',' or '=' do not even split into a strategy and its parameters
        assert "loop 0: EVT invariant I:" in refused.output or "Invalid value for '--loop'" in refused.output
        assert "Traceback" not in refused.output
    _assert_untouched(tmp_path)
    accepted = runner.invoke(cli, [command, str(program), "--loop", f"0=evt_invariant:invariant={INVARIANT}"])
    assert accepted.exit_code == 0, accepted.output


def test_cli_decisions_file_is_checked(tmp_path, program):
    decisions = tmp_path / "decisions.json"
    decisions.write_text('{"0": {"strategy": "evt_subinvariant", "params": {"subinvariant": "x.__class__"}}}')
    refused = CliRunner().invoke(cli, ["analyze", str(program), "--decisions", str(decisions)])
    assert refused.exit_code == 2, refused.output
    assert "loop 0: EVT sub-invariant nu:" in refused.output


def test_web_requests_report_each_malicious_expression_once(tmp_path):
    # pylint: disable=import-outside-toplevel
    from prodigy.web.requests import validate_request
    for text in _malicious(tmp_path):
        problems = validate_request({"source": GEOMETRIC, "decisions": {
            "0": {"strategy": "evt_invariant", "params": {"invariant": text}}}})
        assert len(problems) == 1 and problems[0].startswith("loop 0: EVT invariant I: "), problems
    assert validate_request({"source": GEOMETRIC, "decisions": {
        "0": {"strategy": "evt_invariant", "params": {"invariant": INVARIANT}}}}) == []
