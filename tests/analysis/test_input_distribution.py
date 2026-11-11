"""Input distributions in closed form are accepted only when they are proven to be distributions (#296):
a power series at 0 with non-negative coefficients and finite mass."""
import time

import pytest
from click.testing import CliRunner

from prodigy.analysis.analyzer import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.input_distribution import InputDistributionError, check_input_expression, \
    input_distribution_from_text, validate_input_distribution
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl
from prodigy.web.app import create_app

ENGINES = list(ForwardAnalysisConfig.Engine)

PROGRAM = "nat x; nat y; x := x + 1;\n?Pr[x = 1];\n"

VALID = {
    "finite": "1/2*x + 1/4*x^2*y + 1/4",
    "geometric": "1/(2 - x)",
    "poisson": "exp(3*(x - 1))",
    "two geometrics": "1/((2 - x)*(3 - y))*2",
    "joint": "x*y/(3 - x - y)",
    "poisson times finite": "exp(2*x - 2)*(1/2 + y/2)",
}

REFUTED = {
    "signed": ("1 - x", "negative"),
    "signed rational": ("x/(2 - x) - x^2/2", "negative"),
    "divergent": ("1/(1 - x)", "infinite"),
    "pole inside the unit disc": ("1/(1 - 2*x)", "infinite"),
    "pole at 0": ("1/x", "pole at 0"),
}

UNPROVEN = {
    "alternating": "1/(1 + x)",
    "exp with negative coefficients": "exp(-x)",
    "root": "sqrt(x)",
    "real parameter": "p*x + 1 - p",
}


@pytest.fixture(params=ENGINES, ids=lambda e: e.name.lower())
def config(request):
    return ForwardAnalysisConfig(engine=request.param)


def _program(parameters=""):
    return parse_pgcl(parameters + PROGRAM)


@pytest.mark.parametrize("text", VALID.values(), ids=VALID.keys())
def test_valid_inputs_are_accepted(config, text):
    program = _program()
    dist = input_distribution_from_text(config, text, program)
    result, _ = compute_discrete_distribution(program, dist, config)
    assert not result.is_zero_dist()


@pytest.mark.parametrize("text,reason", REFUTED.values(), ids=REFUTED.keys())
def test_refuted_inputs_are_refused(config, text, reason):
    program = _program()
    dist = config.factory.from_expr(text, *program.variables)
    with pytest.raises(InputDistributionError, match=reason) as info:
        compute_discrete_distribution(program, dist, config)
    assert info.value.details["proof"] == "refuted"
    assert "is not a distribution" in str(info.value)


@pytest.mark.parametrize("text", UNPROVEN.values(), ids=UNPROVEN.keys())
def test_unproven_inputs_are_refused(config, text):
    program = _program("rparam p;\n")
    dist = config.factory.from_expr(text, *program.variables)
    with pytest.raises(InputDistributionError, match="Prodigy accepts an input distribution only") as info:
        compute_discrete_distribution(program, dist, config)
    assert info.value.details["proof"] == "not_proven"


def test_parameters_count_only_with_a_proven_sign():
    program = _program("nparam n;\nrparam p;\n")
    config = ForwardAnalysisConfig()
    # an nparam is a natural number
    validate_input_distribution(config.factory.from_expr("(n*x + 1)/(n + 1)", *program.variables), program)
    # an rparam has no known sign, even where it is meant as a probability
    for text in ("p*x + 1 - p", "(1 - p)/(1 - p*x)", "x^n"):
        with pytest.raises(InputDistributionError) as info:
            validate_input_distribution(config.factory.from_expr(text, *program.variables), program)
        assert info.value.details["proof"] == "not_proven"
    # a name that is no parameter has no sign either
    assert check_input_expression("q*x + 1", ["x"])[0] is None


def test_unsafe_text_is_refused_before_any_engine_sees_it():
    program = _program()
    with pytest.raises(InputDistributionError, match="contains characters"):
        input_distribution_from_text(ForwardAnalysisConfig(), "__import__('os').system('true')", program)
    with pytest.raises(InputDistributionError, match="neither program variables nor parameters"):
        input_distribution_from_text(ForwardAnalysisConfig(), "z/2 + 1/2", program)


@pytest.mark.parametrize("text", ["1 - x", "1/(1 - x)"])
def test_session_reports_a_refused_input_as_error(config, text):
    session = AnalysisSession(parse_pgcl(PROGRAM), source=PROGRAM, config=config, input_distribution=text,
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "error"
    assert result["error"]["type"] == "InputDistributionError"
    assert result["error"]["details"]["proof"] == "refuted"
    assert "verdict" not in result["error"]  # explained as the verdict on an invariant
    assert result["queries"] == []


@pytest.mark.parametrize("engine", ["sympy", "ginac", "symengine"])
@pytest.mark.parametrize("command", ["main", "analyze"])
def test_cli_refuses_invalid_inputs(tmp_path, engine, command):
    program = tmp_path / "program.pgcl"
    program.write_text(PROGRAM)
    runner = CliRunner()
    accepted = runner.invoke(cli, ["--engine", engine, command, str(program), "1/(2 - x)"])
    assert accepted.exit_code == 0, accepted.output
    for text, reason in [("1 - x", "negative"), ("1/(1 - x)", "infinite")]:
        refused = runner.invoke(cli, ["--engine", engine, command, str(program), text])
        assert refused.exit_code == 1, refused.output
        assert reason in refused.output and "is not a distribution" in refused.output
        assert "Traceback" not in refused.output


@pytest.mark.parametrize("command", ["invariant_synthesis", "evt_lower_bound"])
def test_cli_loop_commands_refuse_invalid_inputs(tmp_path, command):
    program = tmp_path / "program.pgcl"
    program.write_text("nat x; nat c;\nwhile (c = 0) { {c := 1} [1/2] {x := x + 1} }\n")
    refused = CliRunner().invoke(cli, [command, str(program), "--invariant", "1", "1 - x"])
    assert refused.exit_code == 1, refused.output
    assert "is not a distribution" in refused.output


def _wait_for(client, job_id, host, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}", headers=host).get_json()
        if status["state"] not in ("starting", "running"):
            return status
        time.sleep(0.1)
    raise AssertionError("analysis did not finish")


@pytest.fixture(scope="module")
def client():
    app = create_app(time_limit=30, max_jobs=2)
    yield app.test_client()
    app.extensions["prodigy_jobs"].shutdown()


@pytest.mark.parametrize("engine", ["sympy", "ginac", "symengine"])
def test_web_refuses_invalid_inputs(client, engine):
    host = {"Host": "localhost:8080"}
    expected = {"1/(2 - x)": "ok", "1 - x": "error", "1/(1 - x)": "error", "1/(1 - 2*x)": "error"}
    for text, status in expected.items():
        response = client.post("/api/analyze", json={"source": PROGRAM, "engine": engine,
                                                     "input_distribution": text}, headers=host)
        assert response.status_code == 202, response.get_json()
        result = _wait_for(client, response.get_json()["job"], host)["result"]
        assert result["status"] == status, (text, result.get("error"))
        if status == "error":
            assert result["error"]["type"] == "InputDistributionError"
            assert "is not a distribution" in result["error"]["message"]
