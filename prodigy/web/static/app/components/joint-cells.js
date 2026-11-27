// The cells of a joint distribution (joint-view.js) and their colours: the colour scale, a cell's interval
// and readout, and moving between cells with the keyboard. Shared by the heatmap and the 3D bars.
import { DEFAULT_VIEW } from "./charts.js";
import { probabilityText, upperSourceText } from "./format.js";
import { COLOUR_MAPS, colourAt, probabilityScale, viridis } from "./plot-scale.js";

export { viridis };

// ---- colour scale ---------------------------------------------------------------------------

/**
 * Maps probabilities to [0, 1] and to colours (plot-scale.js: probabilityScale, colourAt) with the
 * view options `view` (λ of the scale from linear to logarithmic, the colour map). `t(p)` is null for
 * 0 and for values without a number (parameters), which get no colour. `atLeast` raises the largest
 * value, so that the frames of an animation share one scale (#289).
 */
export function colourScale(values, view = DEFAULT_VIEW, atLeast = 0) {
  const scale = probabilityScale(values, { lambda: view.lambda, atLeast });
  const colours = COLOUR_MAPS[view.colours] ? view.colours : "viridis";
  return { ...scale, colours, colour: (t) => colourAt(colours, t) };
}

// ---- helpers ----------------------------------------------------------------------------------

/**
 * The values of a cell and its probability as an interval (#405, format.js: probabilityText),
 * alike in the heatmap's readout and the 3D view's tooltip:
 * {where: "x = 1, y = 2", value: "[1/36, 0.0316840] ([0.0277777, 0.031684])"}.
 */
export function cellParts(grid, r, c) {
  const cell = grid.cells[r][c];
  const where = `${grid.x} = ${grid.x_values[c]}, ${grid.y} = ${grid.y_values[r]}`;
  if (cell.pending) return { where, value: "computing…" };
  return { where, value: probabilityText(grid.bound, cell) };
}

/** The readout of a cell: its interval and where its upper end comes from (#451), like the bar charts' readout. */
export function cellText(grid, r, c) {
  const { where, value } = cellParts(grid, r, c);
  const cell = grid.cells[r][c];
  return [`${where}: ${value}`, cell.pending ? null : upperSourceText(grid.bound, cell)].filter(Boolean).join("; ");
}

/** The cell [r, c] after an arrow, Home/End or Page Up/Down key (y grows upwards), or null. */
export function cellMove(key, [r, c], rows, cols) {
  const moves = { ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1], ArrowUp: [r + 1, c], ArrowDown: [r - 1, c],
    Home: [r, 0], End: [r, cols - 1], PageUp: [rows - 1, c], PageDown: [0, c] };
  const target = moves[key];
  return target ? [Math.max(0, Math.min(rows - 1, target[0])), Math.max(0, Math.min(cols - 1, target[1]))] : null;
}

/** The active cell of `view` if it lies in a rows × cols window, else [0, 0]. */
export function activeCell(view, rows, cols) {
  return view.active && view.active[0] < rows && view.active[1] < cols ? view.active : [0, 0];
}

export function axisLabels(values, every) {
  return values.map((v, i) => (i % every === 0 || i === values.length - 1 ? String(v) : null));
}
