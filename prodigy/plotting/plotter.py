"""
Draws the data of a ``plot`` query with matplotlib (T7 of ``docs/source/architecture.md``).

The analysis does not draw: a ``plot`` statement records its data in the result (``QueryResult.data`` of a query
of kind ``plot``), the GUI draws it in the browser and the CLI with this module. The data is

- for one variable: ``{"variables": [x], "marginal": table}`` with the rows ``{"value", "prob", "p"}`` of the
  marginal (``prodigy.analysis.values.univariate_table``), already cut at the statement's term count or
  probability;
- for two variables: ``{"variables": [x, y], "joint": table}`` with the rows ``{"state", "prob", "p"}``
  (``joint_table``), or ``"joint": None`` if the support is infinite.

``p`` is the probability as a float, ``None`` if it depends on a parameter. Data that cannot be drawn raises a
:class:`PlotError` with the reason, which the front end shows instead of a figure.
"""
from typing import List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PlotData = Mapping[str, object]
"""``QueryResult.data`` of a ``plot`` query, as described above."""


class PlotError(ValueError):
    """The data of a plot query cannot be drawn; the message says why."""


def _rows(table: object) -> Sequence[Mapping[str, object]]:
    rows = table.get("rows") if isinstance(table, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise PlotError("The plot data has no table of probabilities.")
    return rows


def _probability(row: Mapping[str, object]) -> float:
    p = row.get("p")
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise PlotError("Cannot plot parametrized distributions.")
    return float(p)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlotError(f"The plot data has a value {value!r} that is not a natural number.")
    return value


def histogram(variable: str, table: object) -> None:
    """A bar chart of the marginal of ``variable`` (rows ``{"value", "p"}``), coloured by probability."""
    rows = _rows(table)
    if not rows:
        raise PlotError(f"The marginal of {variable} has no states to plot.")
    values = [float(_integer(row.get("value"))) for row in rows]
    probabilities = [_probability(row) for row in rows]
    top = max(probabilities) or 1.0
    colours = plt.get_cmap("Blues")
    mappable = ScalarMappable(cmap=colours, norm=Normalize(0, top))
    mappable.set_array([])
    axis = plt.subplot()
    axis.bar(values, probabilities, 1, linewidth=.5, ec=(0, 0, 0), color=colours([p / top for p in probabilities]))
    axis.set_xlabel(variable)
    axis.set_xticks(values)
    axis.set_ylabel(f"Probability p({variable})")
    manager = plt.get_current_fig_manager()
    if manager:
        manager.set_window_title("Histogram Plot")
    plt.gcf().suptitle("Histogram")
    # pass the axes (https://github.com/matplotlib/matplotlib/issues/23973)
    plt.colorbar(mappable, ax=plt.gca())
    plt.show()


def heatmap(x: str, y: str, table: object) -> None:
    """The joint probabilities of ``x`` and ``y`` (rows ``{"state", "p"}``) as a coloured grid."""
    rows = _rows(table)
    if isinstance(table, Mapping) and table.get("truncated"):
        raise PlotError(f"The joint distribution of {x} and {y} has too many states to plot.")
    cells: List[Tuple[int, int, float]] = []
    for row in rows:
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise PlotError("The plot data has a row without a state.")
        cells.append((_integer(state.get(x, 0)), _integer(state.get(y, 0)), _probability(row)))
    if not cells:
        raise PlotError(f"The joint distribution of {x} and {y} has no states to plot.")
    width, height = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1
    grid = [[0.0] * width for _ in range(height)]
    for i, j, p in cells:
        grid[j][i] += p
    image = plt.imshow(grid, vmin=0, origin="lower", interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(image, ax=plt.gca())
    plt.gca().set_xlabel(x)
    plt.gca().set_xticks(range(width))
    plt.gca().set_ylabel(y)
    plt.gca().set_yticks(range(height))
    plt.show()


def plot(data: PlotData) -> None:
    """Draws the data of one ``plot`` query; raises :class:`PlotError` if it cannot be drawn."""
    variables = data.get("variables")
    if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
        raise PlotError("The plot data names no variables.")
    if len(variables) == 1 and "marginal" in data:
        histogram(variables[0], data["marginal"])
    elif len(variables) == 2 and "joint" in data:
        if data["joint"] is None:
            raise PlotError("Joint plots of infinite-support distributions are not available yet.")
        heatmap(variables[0], variables[1], data["joint"])
    else:
        raise PlotError("The plot data has no table for its variables.")
