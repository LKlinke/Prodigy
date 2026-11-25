"""
The engine settings of a ForwardAnalysisConfig (LaTeX output) hold for its own
analysis only, not for the rest of the process.
"""
import threading

from click.testing import CliRunner

from prodigy.analysis.session import compute_discrete_distribution
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.distribution.generating_function import GeneratingFunction, GeneratingFunctionSettings
from prodigy.language.compiler import parse_program

DEFAULTS = GeneratingFunctionSettings()

PROGRAM = """x: nat;
y: nat;
{
    x = x + 1;
} [1/2] {
    y = y + 1;
}"""


def analyse(config):
    program = parse_program(PROGRAM)
    dist, _ = compute_discrete_distribution(
        program, config.factory.from_expr("1/((2-x)*(2-y))", "x", "y"), config)
    return repr(dist), GeneratingFunction.settings()


def test_constructing_a_config_changes_nothing():
    ForwardAnalysisConfig(use_simplification=True, use_latex=True)
    assert GeneratingFunction.settings() == DEFAULTS


def test_two_analyses_in_one_process_do_not_influence_each_other():
    simplifying, plain = ForwardAnalysisConfig(use_simplification=True), ForwardAnalysisConfig()
    first = [analyse(simplifying), analyse(plain)]
    second = [analyse(plain), analyse(simplifying)]
    assert first == second[::-1]
    # simplification only concerns the shown result (#404): the computed distribution is the same, and
    # no setting outlives its analysis
    assert first[0][0] == first[1][0]
    assert all(settings == DEFAULTS for _, settings in first + second)


def test_activate_restores_the_enclosing_settings():
    latex = ForwardAnalysisConfig(use_latex=True)
    with latex.activate():
        assert GeneratingFunction.settings() == GeneratingFunctionSettings(latex_output=True)
        analyse(ForwardAnalysisConfig(use_simplification=True))
        assert GeneratingFunction.settings() == GeneratingFunctionSettings(latex_output=True)
        assert str(GeneratingFunction("x/2", "x")) == r"\frac{x}{2}"
    assert GeneratingFunction.settings() == DEFAULTS
    assert str(GeneratingFunction("x/2", "x")) == "x/2"


def test_settings_do_not_leak_into_other_threads():
    seen = []
    with ForwardAnalysisConfig(use_simplification=True, use_latex=True).activate():
        thread = threading.Thread(target=lambda: seen.append(GeneratingFunction.settings()))
        thread.start()
        thread.join()
    assert seen == [DEFAULTS]


def test_session_result_uses_its_own_settings():
    source = """x: nat;
x = 1;"""
    for use_latex in (True, False, True):
        session = AnalysisSession(parse_program(source), source=source,
                                  config=ForwardAnalysisConfig(use_latex=use_latex), capture_output=True)
        assert session.run().to_json()["status"] == "ok"
        assert GeneratingFunction.settings() == DEFAULTS


def test_cli_prints_latex_and_leaves_the_defaults(tmp_path):
    program = tmp_path / "program.ppy"
    program.write_text("""x: nat;
{
    x = 1;
} [1/2] {
    x = 2;
}""")
    runner = CliRunner()
    latex = runner.invoke(cli, ["--use-latex", "--no-simplification", "main", str(program)])
    assert latex.exit_code == 0, latex.output
    assert r"\frac{x^{2}}{2} + \frac{x}{2}" in latex.output
    plain = runner.invoke(cli, ["--no-simplification", "main", str(program)])
    assert plain.exit_code == 0, plain.output
    assert "x**2/2 + x/2" in plain.output
    # by default only the shown result is simplified (#404)
    shown = runner.invoke(cli, ["main", str(program)])
    assert shown.exit_code == 0, shown.output
    assert "x*(x + 1)/2" in shown.output
    assert GeneratingFunction.settings() == DEFAULTS
