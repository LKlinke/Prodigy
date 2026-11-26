"""Plots (T7 of docs/source/architecture.md): the analysis records the data of a ``plot`` statement in the result,
also when the CLI runs it interactively; ``prodigy.plotting`` draws that data and the CLI calls it."""
from fractions import Fraction

import matplotlib
import pytest
from click.testing import CliRunner

from prodigy.analysis.session import AnalysisSession
from prodigy.cli import cli
from prodigy.language import parse_program
from prodigy.plotting import plotter
from prodigy.plotting.plotter import PlotError, plot

matplotlib.use("Agg")

FINITE = "x: nat;\ny: nat;\n{ x = 1; } [1/4] { x = 2; y = 1; }\nplot(x);\nplot(x, y);\n"
GEOMETRIC = "x: nat;\ny: nat;\nx = geometric(1/2);\nplot(x, 3);\nplot(x, y);\n"
PARAMETRIC = "x: nat;\nparam p: real;\n{ x = 1; } [p] { x = 2; }\nplot(x);\n"


def _plots(source: str, *, interactive: bool):
    program = parse_program(source)
    make = AnalysisSession.interactive if interactive else AnalysisSession
    session = make(program, source=source, capture_output=True)
    result = session.run().to_json()
    return [query["data"] for query in result["queries"] if query["kind"] == "plot"]


@pytest.fixture(name="shown")
def fixture_shown(monkeypatch):
    """The figures that plt.show would have opened, as (bar heights, image) per call."""
    figures = []

    def show():
        axes = matplotlib.pyplot.gca()
        heights = [patch.get_height() for patch in axes.patches]
        images = [image.get_array().tolist() for image in axes.get_images()]
        figures.append((heights, images))
        matplotlib.pyplot.close("all")

    monkeypatch.setattr(plotter.plt, "show", show)
    return figures


@pytest.mark.parametrize("interactive", [False, True])
def test_the_analysis_records_plot_data_for_every_front_end(interactive):
    marginal, joint = _plots(FINITE, interactive=interactive)
    assert marginal["variables"] == ["x"]
    assert [(row["value"], Fraction(row["prob"])) for row in marginal["marginal"]["rows"]] == [
        (1, Fraction(1, 4)), (2, Fraction(3, 4))]
    assert joint["variables"] == ["x", "y"] and not joint["joint"]["truncated"]


def test_a_marginal_is_drawn_as_a_histogram(shown):
    marginal, _ = _plots(FINITE, interactive=False)
    plot(marginal)
    assert shown == [([0.25, 0.75], [])]


def test_the_term_count_of_the_statement_cuts_the_histogram(shown):
    marginal, _ = _plots(GEOMETRIC, interactive=False)
    plot(marginal)
    assert shown == [([0.5, 0.25, 0.125], [])]


def test_a_joint_distribution_is_drawn_as_a_grid(shown):
    _, joint = _plots(FINITE, interactive=False)
    plot(joint)
    # rows are y = 0, 1 and columns x = 0, 1, 2
    assert shown == [([], [[[0.0, 0.25, 0.0], [0.0, 0.0, 0.75]]])]


@pytest.mark.parametrize("source, index, reason", [
    (GEOMETRIC, 1, "infinite-support"),
    (PARAMETRIC, 0, "parametrized"),
])
def test_data_that_cannot_be_drawn_says_why(shown, source, index, reason):
    data = _plots(source, interactive=False)[index]
    with pytest.raises(PlotError, match=reason):
        plot(data)
    assert not shown


def test_the_cli_draws_the_plots_of_the_program(shown, tmp_path):
    path = tmp_path / "plots.ppy"
    path.write_text(GEOMETRIC, encoding="utf-8")
    result = CliRunner().invoke(cli, ["main", str(path)])
    assert result.exit_code == 0, result.output
    assert shown == [([0.5, 0.25, 0.125], [])]
    assert "Plot of x, y:" in result.output and "infinite-support" in result.output
    assert result.output.index("Plot of x, y:") < result.output.index("Result:")
