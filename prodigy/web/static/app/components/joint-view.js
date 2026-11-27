// Joint distribution of two variables (issue #197): heatmap, 3D bars and a table of the exact
// coefficients Pr[x = i, y = j] for a window. The coefficients come from Prodigy
// (POST api/jobs/<id>/joint, prodigy/analysis/joint_grid.py); the browser only draws them.
//
// Like result-view.js this needs no GUI module: `load(payload, signal)` fetches a grid.
// Colours: viridis by default, cividis or greyscale in the view options (#452, view-options.js). Values
// are never shown by colour alone: the readout below the heatmap, the tooltips and the table give them.
// The colour scale and the cells are in joint-cells.js, the 3D bars in joint-bars3d.js.
import { DEFAULT_VIEW, exactInterval, hatchPattern, ownUpperEnds } from "./charts.js";
import { h, s } from "./dom.js";
import { formatNumber, formatValue, PROBABILITY_GUARANTEE, probabilityText } from "./format.js";
import { bars3d } from "./joint-bars3d.js";
import { activeCell, axisLabels, cellMove, cellText, colourScale } from "./joint-cells.js";
import { scaleName, textOn } from "./plot-scale.js";

let idCounter = 0;

// ---- heatmap ----------------------------------------------------------------------------------

function heatmap(grid, colours, view, readout, width) {
  const cols = grid.x_values.length;
  const rows = grid.y_values.length;
  const left = 34;
  const bottom = 30;
  const top = 6;
  // cells fill the width (at most 56 px wide) and at most about 440 px of height
  const cw = Math.max(8, Math.min(56, Math.floor((Math.max(260, width) - left - 12) / cols)));
  const ch = Math.max(8, Math.min(32, Math.floor(440 / rows), cw));
  const svgW = left + cols * cw + 12;
  const svgH = top + rows * ch + bottom;
  const hatch = `jv-hatch-${++idCounter}`;
  const every = (n) => Math.max(1, Math.ceil(n / Math.max(1, Math.floor((n * cw) / 28))));
  const xLabels = axisLabels(grid.x_values, every(cols));
  const yLabels = axisLabels(grid.y_values, Math.max(1, Math.ceil(16 / ch)));
  // y grows upwards, like a plot: the last row of the grid is drawn at the top
  const cy = (r) => top + (rows - 1 - r) * ch;
  // numbers inside the cells only where they fit (the readout and the table always have them)
  const cellNumber = (p) => (p >= 0.01 ? p.toFixed(2).replace(/^0/, "") : null);
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid.cells[r][c];
      const t = colours.t(cell.p);
      const kind = cell.pending ? "pending" : cell.p === null ? "symbolic" : t === null ? "zero" : "value";
      const fill = { value: colours.colour(t), symbolic: `url(#${hatch})`, pending: `url(#${hatch}-pending)` }[kind] ?? null;
      cells.push(s("rect", { class: `jv-cell jv-${kind}`, x: left + c * cw, y: cy(r), width: cw, height: ch,
        fill, "data-r": r, "data-c": c }));
      const number = kind === "value" && cw >= 30 && ch >= 16 ? cellNumber(cell.p) : null;
      if (number) {
        cells.push(s("text", { class: "jv-cell-text", x: left + c * cw + cw / 2, y: cy(r) + ch / 2 + 4, fill: textOn(colours.colour(t)),
          "aria-hidden": "true" }, number));
      }
    }
  }
  const marker = s("rect", { class: "jv-active", width: cw, height: ch, rx: 2 });
  const svg = s("svg", { class: "jv-heatmap", "data-colours": colours.colours, viewBox: `0 0 ${svgW} ${svgH}`, width: svgW, height: svgH, "aria-hidden": "true" },
    s("defs", {}, hatchPattern(hatch), hatchPattern(`${hatch}-pending`, "pending")),
    ...cells,
    ...xLabels.map((label, c) => (label === null ? null
      : s("text", { class: "jv-tick", x: left + c * cw + cw / 2, y: top + rows * ch + 13, "text-anchor": "middle" }, label))),
    ...yLabels.map((label, r) => (label === null ? null
      : s("text", { class: "jv-tick", x: left - 6, y: cy(r) + ch / 2 + 4, "text-anchor": "end" }, label))),
    s("text", { class: "jv-axis-title", x: left + (cols * cw) / 2, y: svgH - 3, "text-anchor": "middle" }, grid.x),
    s("text", { class: "jv-axis-title", x: 10, y: top + (rows * ch) / 2, "text-anchor": "middle",
      transform: `rotate(-90 10 ${top + (rows * ch) / 2})` }, grid.y),
    marker);

  const holder = h("div", { class: "jv-plot", tabindex: "0", role: "application", "aria-roledescription": "heatmap",
    "aria-label": `Heatmap of the joint distribution of ${grid.x} and ${grid.y}, ${cols} by ${rows} cells. ` +
      "Arrow keys move between cells, Home and End go to the first and last column, Page Up and Page Down to the top and bottom row.",
    "aria-describedby": readout.id }, svg);

  const select = (r, c) => {
    view.active = [Math.max(0, Math.min(rows - 1, r)), Math.max(0, Math.min(cols - 1, c))];
    const [ar, ac] = view.active;
    marker.setAttribute("x", left + ac * cw);
    marker.setAttribute("y", cy(ar));
    readout.textContent = cellText(grid, ar, ac);
  };
  select(...activeCell(view, rows, cols));
  holder.addEventListener("keydown", (event) => {
    const target = cellMove(event.key, view.active, rows, cols);
    if (!target) return;
    event.preventDefault();
    select(...target);
  });
  const fromPointer = (event) => {
    const target = event.target.closest?.(".jv-cell");
    if (target) select(Number(target.dataset.r), Number(target.dataset.c));
  };
  svg.addEventListener("pointermove", fromPointer);
  svg.addEventListener("click", (event) => { fromPointer(event); holder.focus({ preventScroll: true }); });
  return holder;
}


// ---- table ------------------------------------------------------------------------------------

function gridTable(grid) {
  const head = h("tr", {}, h("th", { scope: "col" }, `${grid.y} \\ ${grid.x}`),
    ...grid.x_values.map((v) => h("th", { scope: "col", class: "num" }, String(v))));
  const body = [...grid.y_values.keys()].reverse().map((r) => h("tr", {},
    h("th", { scope: "row", class: "num" }, String(grid.y_values[r])),
    ...grid.cells[r].map((cell) => (cell.pending ? h("td", { class: "num rv-muted" }, "computing…")
      : h("td", { class: "num", title: exactInterval(grid.bound, cell) },
        grid.bound === "exact" || cell.p === null ? (cell.p === null ? cell.prob : formatValue(cell.prob, cell.p, 4))
          : probabilityText(grid.bound, { ...cell, prob: null, upper: null }, 4))))));
  return h("div", { class: "table-wrap jv-table" }, h("table", { class: "rv-table" },
    h("caption", { class: "rv-note" }, `Pr[${grid.x} = column, ${grid.y} = row]; hover a value for its exact form. `
      + (PROBABILITY_GUARANTEE[grid.bound] || PROBABILITY_GUARANTEE.unknown)),
    h("thead", {}, head), h("tbody", {}, body)));
}

// ---- colour key -----------------------------------------------------------------------------------
// The colour key says what colour one cell has for its own probability. The mass of the window is a
// different quantity (#286); the result view shows it in its mass line (charts.js: massSummary),
// with its own heading and a grey bar, so it never reads as part of the colour scale.

function keyHeading(id, title, detail) {
  return h("h4", { class: "jv-key-title", id }, title, h("span", { class: "jv-key-detail" }, detail));
}

const COLOUR_END = { lower: " (the lower end of its interval)", upper: " (the upper end of its interval)" };

function legend(colours, bound) {
  const stops = Array.from({ length: 11 }, (_, i) => `${colours.colour(i / 10)} ${i * 10}%`).join(", ");
  // the position of a tick on the key is its place on the scale (0 has none on a scale with a floor)
  const position = (value) => (colours.t(value) ?? 0) * 100;
  const id = `jv-legend-${++idCounter}`;
  const detail = colours.max > 0
    ? ` — ${scaleName(colours.lambda).replace(/^\w/, (c) => c.toLowerCase())}, from ${formatNumber(colours.min, 2)} to ${formatNumber(colours.max, 2)}`
    : "";
  return h("section", { class: "jv-key-block jv-legend", "aria-labelledby": id },
    keyHeading(id, `Colour: probability of one cell${COLOUR_END[bound] || ""}`, detail),
    h("div", { class: "jv-legend-row" },
      h("span", { class: "jv-legend-zero" }, h("span", { class: "jv-swatch jv-zero", "data-colours": colours.colours, "aria-hidden": "true" }), "0"),
      h("div", { class: "jv-gradient-wrap" },
        h("div", { class: "jv-gradient", style: `background: linear-gradient(to right, ${stops})`, "aria-hidden": "true" }),
        h("div", { class: "jv-gradient-ticks" }, ...colours.ticks.map((tick) => h("span", {
          style: `left: ${Math.max(0, Math.min(100, position(tick)))}%` }, formatNumber(tick, 2)))))));
}

/**
 * The key of the 3D view's interval marks (#452), in words: the box above a bar and the "?" tower,
 * so that neither is told apart by colour or hatching alone.
 */
function intervalKey(grid, options) {
  if (grid.bound !== "lower") return null;
  const own = ownUpperEnds(grid.cells.flat());
  const style = options.intervals === "glass" ? "translucent box with dashed edges" : "dashed outline";
  const items = [h("li", {}, h("span", { class: `jv-key-glyph vo-glyph vo-glyph-${options.intervals}`, "aria-hidden": "true" }),
    own ? `Upper ends: a ${style} above a bar reaches the upper end of its interval; drawn for the cells with an upper end of their own, for the others on hover or selection.`
      : `Upper ends: a ${style} above the selected bar reaches the upper end of its interval (its lower end plus the mass not yet accounted for).`)];
  if (options.unaccounted && grid.residual_mass_float > 0) {
    items.push(h("li", {}, h("span", { class: "jv-key-glyph vo-glyph vo-glyph-residual", "aria-hidden": "true" }, "?"),
      `The hatched “?” tower beside the grid: the mass not yet accounted for, ${formatValue(grid.residual_mass, grid.residual_mass_float)}, on the same scale.`));
  }
  return h("ul", { class: "jv-interval-key rv-note" }, ...items);
}

// ---- the section ----------------------------------------------------------------------------------

const PENDING = Object.freeze({ pending: true, prob: null, p: null });

/** A grid of the cells streamed so far (prodigy/web/jobs.py: merge_stream): the missing ones pending. */
export function pendingGrid(stream) {
  return { ...stream, cells: stream.cells.map((row) => row.map((cell) => cell || PENDING)) };
}

const VIEWS = { heatmap: "Heatmap", bars: "3D", table: "Table" };

function numberField(label, value, max) {
  return h("label", { class: "jv-field" }, h("span", {}, label),
    h("input", { class: "field num", type: "number", min: "0", max: String(max), step: "1", value: String(value), inputmode: "numeric" }));
}

/** The kinds of view of a joint distribution, for the view switch of the result view. */
export const JOINT_VIEWS = VIEWS;

/**
 * The joint view of two variables. `load({x, y, x_window, y_window}, signal)` resolves to a grid
 * (prodigy/analysis/joint_grid.py) or rejects with an Error. `view` keeps the choices across
 * re-renders of the result; `view.mode` is one of JOINT_VIEWS. `options()` gives the view options
 * shared with the bar charts (view-options.js). The switches for the view and the options belong to the caller (the
 * result view has one control row for all its figures), which calls `draw()` after a change.
 * `onGrid(grid | null)` reports each grid shown (null while none is), for the caller's mass line.
 * `refresh()` loads the grid again for the same pair (the analysis card calls it when its timeline
 * moves to another position, and `load` then asks for that position's grid), in the window the user
 * chose with "Show window" or else in the position's default window. While the
 * pair and the window stay the same, the colour scale keeps the largest value seen so far, so that
 * frames can be compared (#289). Returns {element, draw, refresh}.
 */
export function jointView({ variables, load, view, maxAxis = 50, options = () => DEFAULT_VIEW, onGrid = () => {} }) {
  if (!view.x || !variables.includes(view.x)) view.x = variables[0];
  if (!view.y || !variables.includes(view.y) || view.y === view.x) view.y = variables.find((v) => v !== view.x);

  const readout = h("p", { class: "jv-readout mono", id: `jv-readout-${++idCounter}`, "aria-live": "polite" });
  const body = h("div", { class: "jv-body" });
  const status = h("p", { class: "rv-note jv-status", role: "status" });
  const xSelect = h("select", { "aria-label": "Variable on the horizontal axis" }, ...variables.map((v) => h("option", { value: v }, v)));
  const ySelect = h("select", { "aria-label": "Variable on the vertical axis" }, ...variables.map((v) => h("option", { value: v }, v)));
  xSelect.value = view.x;
  ySelect.value = view.y;
  const xFrom = numberField("from", view.xWindow?.[0] ?? 0, 1000);
  const xTo = numberField("to", view.xWindow?.[1] ?? 15, 1000);
  const yFrom = numberField("from", view.yWindow?.[0] ?? 0, 1000);
  const yTo = numberField("to", view.yWindow?.[1] ?? 15, 1000);
  let grid = null;
  let controller = null;

  const fieldValue = (field) => Number(field.querySelector("input").value);
  const setFields = (g) => {
    xFrom.querySelector("input").value = g.x_values[0];
    xTo.querySelector("input").value = g.x_values[g.x_values.length - 1];
    yFrom.querySelector("input").value = g.y_values[0];
    yTo.querySelector("input").value = g.y_values[g.y_values.length - 1];
  };

  // the largest probability seen for this pair and window, over all positions of the timeline
  const peakKey = (g) => `${g.x} ${g.y} ${g.x_values[0]}-${g.x_values.at(-1)} ${g.y_values[0]}-${g.y_values.at(-1)}`;
  const draw = () => {
    if (!grid) return;
    const values = grid.cells.flat().map((cell) => cell.p);
    const key = peakKey(grid);
    if (view.peak?.key !== key) view.peak = { key, max: 0 };
    view.peak.max = Math.max(view.peak.max, ...values.filter((p) => Number.isFinite(p)));
    const chosen = options();
    const colours = colourScale(values, chosen, view.peak.max);
    const width = body.clientWidth || 560;
    const symbolic = grid.cells.some((row) => row.some((cell) => cell.p === null && !cell.pending));
    // the readout gives the interval of the active cell
    const notes = [];
    if (symbolic) notes.push("Hatched cells depend on parameters; the table lists them.");
    const parts = [];
    if (view.mode === "heatmap") parts.push(heatmap(grid, colours, view, readout, width), readout);
    else if (view.mode === "bars") parts.push(bars3d(grid, colours, view, readout, width, chosen), readout);
    else parts.push(gridTable(grid));
    if (view.mode !== "table") parts.push(legend(colours, grid.bound));
    if (view.mode === "bars") parts.push(intervalKey(grid, chosen));
    // the guarantee in words (#405); the readout gives each cell's interval
    if (view.mode !== "table") notes.push(PROBABILITY_GUARANTEE[grid.bound] || PROBABILITY_GUARANTEE.unknown);
    if (notes.length) parts.push(h("p", { class: "rv-note" }, notes.join(" ")));
    body.replaceChildren(...parts.filter(Boolean));
  };

  const fetchGrid = async (useWindow) => {
    controller?.abort();
    controller = new AbortController();
    const { signal } = controller;
    view.x = xSelect.value;
    view.y = ySelect.value;
    if (view.x === view.y) {
      status.textContent = "Choose two different variables.";
      grid = null;
      body.replaceChildren();
      onGrid(null);
      return;
    }
    const payload = { x: view.x, y: view.y };
    if (useWindow) {
      const xw = [fieldValue(xFrom), fieldValue(xTo)];
      const yw = [fieldValue(yFrom), fieldValue(yTo)];
      for (const [name, w] of [[view.x, xw], [view.y, yw]]) {
        if (!w.every(Number.isInteger) || w[0] < 0 || w[1] < w[0]) {
          status.textContent = `The window of ${name} needs whole numbers 0 ≤ from ≤ to.`;
          return;
        }
        if (w[1] - w[0] + 1 > maxAxis) {
          status.textContent = `At most ${maxAxis} values of ${name} at once.`;
          return;
        }
      }
      payload.x_window = xw;
      payload.y_window = yw;
    }
    status.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }), " Computing the exact coefficients…");
    body.setAttribute("aria-busy", "true");
    // the cells computed so far (#405): the grid grows from its corner, the others are hatched
    const partial = (stream) => {
      if (signal.aborted) return;
      grid = pendingGrid(stream);
      body.dataset.streaming = "true";
      const total = grid.x_values.length * grid.y_values.length;
      const done = grid.cells.flat().filter((cell) => !cell.pending).length;
      status.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }),
        ` Computing… ${done} of ${total} cells; hatched cells are still being computed.`);
      draw();
    };
    try {
      const result = await load(payload, signal, partial);
      if (signal.aborted) return;
      grid = result;
      view.xWindow = [grid.x_values[0], grid.x_values[grid.x_values.length - 1]];
      view.yWindow = [grid.y_values[0], grid.y_values[grid.y_values.length - 1]];
      setFields(grid);
      onGrid(grid);
      status.textContent = grid.finite_support && grid.support
        ? `Support: ${grid.x} ${grid.support[grid.x][0]}–${grid.support[grid.x][1]}, ${grid.y} ${grid.support[grid.y][0]}–${grid.support[grid.y][1]}.`
        : "Infinite support; adjust the window to see other values.";
      draw();
    } catch (error) {
      if (signal.aborted || error.name === "AbortError") return;
      status.textContent = `Not available: ${error.message}`;
      grid = null;
      body.replaceChildren();
      onGrid(null);
    } finally {
      if (!signal.aborted) {
        body.removeAttribute("aria-busy");
        delete body.dataset.streaming;
      }
    }
  };

  const pairChanged = () => {
    view.xWindow = null;
    view.yWindow = null;
    view.windowChosen = false;
    view.active = null;
    fetchGrid(false);
  };
  xSelect.addEventListener("change", pairChanged);
  ySelect.addEventListener("change", pairChanged);
  const swap = h("button", { type: "button", class: "button small", title: "Swap the axes", "aria-label": "Swap the axes",
    onclick: () => {
      [xSelect.value, ySelect.value] = [ySelect.value, xSelect.value];
      [view.xWindow, view.yWindow] = [view.yWindow, view.xWindow];
      view.active = null;
      if (view.xWindow) {
        xFrom.querySelector("input").value = view.xWindow[0]; xTo.querySelector("input").value = view.xWindow[1];
        yFrom.querySelector("input").value = view.yWindow[0]; yTo.querySelector("input").value = view.yWindow[1];
      }
      fetchGrid(Boolean(view.xWindow));
    } }, "⇄");
  const form = h("form", { class: "jv-controls", onsubmit: (event) => { event.preventDefault(); view.windowChosen = true; fetchGrid(true); } },
    h("fieldset", { class: "jv-axis" }, h("legend", {}, "Horizontal"), xSelect, xFrom, xTo),
    swap,
    h("fieldset", { class: "jv-axis" }, h("legend", {}, "Vertical"), ySelect, yFrom, yTo),
    h("button", { type: "submit", class: "button small" }, "Show window"));

  if (typeof ResizeObserver !== "undefined") {
    let drawnWidth = 0;
    new ResizeObserver(() => {
      if (grid && view.mode !== "table" && body.clientWidth && Math.abs(body.clientWidth - drawnWidth) > 24) {
        drawnWidth = body.clientWidth;
        draw();
      }
    }).observe(body);
  }
  if (!VIEWS[view.mode]) view.mode = "heatmap";
  onGrid(null);
  fetchGrid(Boolean(view.xWindow && view.yWindow));
  return { element: h("div", { class: "jv" }, form, status, body), draw,
    // a window the user chose stays from position to position; else each position gets its default
    refresh: () => fetchGrid(Boolean(view.windowChosen && view.xWindow && view.yWindow)) };
}
