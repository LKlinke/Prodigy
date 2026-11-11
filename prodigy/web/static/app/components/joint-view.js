// Joint distribution of two variables (issue #197): heatmap, 3D bars and a table of the exact
// coefficients Pr[x = i, y = j] for a window. The coefficients come from Prodigy
// (POST api/jobs/<id>/joint, prodigy/analysis/joint_grid.py); the browser only draws them.
//
// Like result-view.js this needs no GUI module: `load(payload, signal)` fetches a grid.
// Colours: viridis (perceptually uniform, readable with colour-vision deficiencies). Values are
// never shown by colour alone: the readout below the heatmap, the tooltips and the table give them.
import { h, s } from "./dom.js";
import { hatchPattern } from "./charts.js";
import { formatNumber, formatValue } from "./format.js";

// ---- colour scale ---------------------------------------------------------------------------

const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];

const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const toHex = (rgb) => `#${rgb.map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("")}`;

/** The viridis colour at t in [0, 1] as #rrggbb. */
export function viridis(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const [a, b] = [hex(VIRIDIS[i]), hex(VIRIDIS[i + 1])];
  return toHex(a.map((v, k) => v + (b[k] - v) * (x - i)));
}

/** Text colour (black or white) with the higher contrast on a viridis background. */
function onColour(t) {
  const [r, g, b] = hex(viridis(t)).map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return (luminance + 0.05) / 0.05 > 1.05 / (luminance + 0.05) ? "#000000" : "#ffffff";
}

const LOG_DECADES = 12;

/**
 * Maps probabilities to [0, 1]: linear from 0 to the largest value, or logarithmic from the
 * smallest to the largest positive value (at most 12 decades). `t(p)` is null for 0 and for values
 * without a number (parameters), which get no colour.
 */
export function colourScale(values, scale = "linear") {
  const positive = values.filter((p) => Number.isFinite(p) && p > 0);
  const max = positive.length ? Math.max(...positive) : 0;
  if (scale === "log" && positive.length) {
    const hi = Math.log10(max);
    const lo = Math.max(Math.log10(Math.min(...positive)), hi - LOG_DECADES);
    const t = (p) => (Number.isFinite(p) && p > 0 ? (hi === lo ? 1 : Math.max(0, Math.min(1, (Math.log10(p) - lo) / (hi - lo)))) : null);
    const ticks = [];
    for (let k = Math.ceil(lo - 1e-9); k <= Math.floor(hi + 1e-9); k += Math.max(1, Math.ceil((hi - lo) / 4))) ticks.push(10 ** k);
    return { scale, min: 10 ** lo, max, t, ticks: ticks.length ? ticks : [max] };
  }
  const t = (p) => (Number.isFinite(p) && p > 0 && max > 0 ? Math.min(1, p / max) : null);
  return { scale: "linear", min: 0, max, t, ticks: max > 0 ? [0, max / 2, max] : [0] };
}

// ---- helpers ----------------------------------------------------------------------------------

let idCounter = 0;
const BOUND_WORDS = { lower: "at least ", upper: "at most " };

function cellText(grid, r, c) {
  const cell = grid.cells[r][c];
  const where = `${grid.x} = ${grid.x_values[c]}, ${grid.y} = ${grid.y_values[r]}`;
  const value = cell.p === null ? cell.prob : formatValue(cell.prob, cell.p, 6);
  const qualifier = cell.prob === "0" && grid.bound !== "upper" ? "" : (BOUND_WORDS[grid.bound] || "");
  const exact = cell.p !== null && cell.prob && cell.prob.length < 40 && cell.prob !== String(cell.p) ? ` (${cell.prob})` : "";
  const upper = cell.upper ? `; at most ${formatValue(cell.upper, cell.upper_p, 6)}` : "";
  return `${where}: ${qualifier}${value}${exact}${upper}`;
}

function axisLabels(values, every) {
  return values.map((v, i) => (i % every === 0 || i === values.length - 1 ? String(v) : null));
}

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
      const kind = cell.p === null ? "symbolic" : t === null ? "zero" : "value";
      cells.push(s("rect", { class: `jv-cell jv-${kind}`, x: left + c * cw, y: cy(r), width: cw, height: ch,
        fill: kind === "value" ? viridis(t) : kind === "symbolic" ? `url(#${hatch})` : null, "data-r": r, "data-c": c }));
      const number = kind === "value" && cw >= 30 && ch >= 16 ? cellNumber(cell.p) : null;
      if (number) {
        cells.push(s("text", { class: "jv-cell-text", x: left + c * cw + cw / 2, y: cy(r) + ch / 2 + 4, fill: onColour(t),
          "aria-hidden": "true" }, number));
      }
    }
  }
  const marker = s("rect", { class: "jv-active", width: cw, height: ch, rx: 2 });
  const svg = s("svg", { class: "jv-heatmap", viewBox: `0 0 ${svgW} ${svgH}`, width: svgW, height: svgH, "aria-hidden": "true" },
    s("defs", {}, hatchPattern(hatch)),
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
  const [r0, c0] = view.active && view.active[0] < rows && view.active[1] < cols ? view.active : [0, 0];
  select(r0, c0);
  holder.addEventListener("keydown", (event) => {
    const [r, c] = view.active;
    const moves = { ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1], ArrowUp: [r + 1, c], ArrowDown: [r - 1, c],
      Home: [r, 0], End: [r, cols - 1], PageUp: [rows - 1, c], PageDown: [0, c] };
    const target = moves[event.key];
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

// ---- 3D bars ----------------------------------------------------------------------------------

const DEFAULT_CAMERA = { yaw: -35, pitch: 32, zoom: 1 };

function shade(colour, factor) {
  return toHex(hex(colour).map((v) => v * factor));
}

function bars3d(grid, colours, view, width) {
  const camera = view.camera || (view.camera = { ...DEFAULT_CAMERA });
  const cols = grid.x_values.length;
  const rows = grid.y_values.length;
  const cssW = Math.max(280, Math.min(720, Math.round(width)));
  const cssH = Math.round(cssW * 0.62);
  const canvas = h("canvas", { class: "jv-canvas", width: cssW, height: cssH, style: `width:${cssW}px;height:${cssH}px`,
    role: "img", "aria-label": `3D bar chart of the joint distribution of ${grid.x} and ${grid.y}; the heatmap and the table give the values.` });

  const draw = () => {
    const ratio = globalThis.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * ratio);
    canvas.height = Math.round(cssH * ratio);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const style = getComputedStyle(canvas);
    const text = style.getPropertyValue("--rv-text-2").trim() || "#6E6E73";
    const line = style.getPropertyValue("--jv-floor").trim() || "rgba(128,128,128,0.45)";
    const yaw = (camera.yaw * Math.PI) / 180;
    const pitch = (camera.pitch * Math.PI) / 180;
    // the window is drawn on a square floor of side SIDE, whatever its numbers of rows and columns
    const SIDE = 10;
    const scale = (camera.zoom * Math.min(cssW, cssH * 1.4)) / (SIDE * 1.75);
    const heightUnits = SIDE * 0.55;
    const cx = cssW / 2;
    const cy = cssH * 0.6;
    // ground coordinates in cells (gx along x, gy along y), centred on the window; z up
    const project = (gx, gy, z) => {
      const X = (gx / cols - 0.5) * SIDE;
      const Y = (gy / rows - 0.5) * SIDE;
      const u = X * Math.cos(yaw) - Y * Math.sin(yaw);
      const v = X * Math.sin(yaw) + Y * Math.cos(yaw);
      return { x: cx + u * scale, y: cy - (z * Math.cos(pitch) + v * Math.sin(pitch)) * scale, depth: v };
    };
    const polygon = (points, fill, stroke) => {
      ctx.beginPath();
      points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
    };
    // floor and its labels
    polygon([project(0, 0, 0), project(cols, 0, 0), project(cols, rows, 0), project(0, rows, 0)], null, line);
    // bars, the farthest first (painter's algorithm)
    const list = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = colours.t(grid.cells[r][c].p);
        if (t !== null) list.push({ r, c, t, depth: project(c + 0.5, r + 0.5, 0).depth });
      }
    }
    list.sort((a, b) => b.depth - a.depth);
    const inset = 0.12;
    for (const bar of list) {
      const z = Math.max(0.02, bar.t) * heightUnits;
      const [x0, x1, y0, y1] = [bar.c + inset, bar.c + 1 - inset, bar.r + inset, bar.r + 1 - inset];
      const colour = viridis(bar.t);
      const faces = [
        [[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]],
      ].map(([[ax, ay], [bx, by]], i) => ({
        points: [project(ax, ay, 0), project(bx, by, 0), project(bx, by, z), project(ax, ay, z)],
        depth: project((ax + bx) / 2, (ay + by) / 2, 0).depth,
        factor: i % 2 ? 0.62 : 0.78,
      })).sort((a, b) => b.depth - a.depth);
      for (const face of faces.slice(2)) polygon(face.points, shade(colour, face.factor), null);
      polygon([project(x0, y0, z), project(x1, y0, z), project(x1, y1, z), project(x0, y1, z)], colour, shade(colour, 0.55));
    }
    // axis labels on top, so that bars never hide them
    ctx.fillStyle = text;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    const label = (gx, gy, value) => { const p = project(gx, gy, 0); ctx.fillText(value, p.x, p.y + 4); };
    const every = (n) => Math.max(1, Math.ceil(n / 6));
    grid.x_values.forEach((v, c) => { if (c % every(cols) === 0) label(c + 0.5, -0.07 * rows, String(v)); });
    grid.y_values.forEach((v, r) => { if (r % every(rows) === 0) label(-0.07 * cols, r + 0.5, String(v)); });
    ctx.font = "italic 12px -apple-system, BlinkMacSystemFont, sans-serif";
    label(cols / 2, -0.17 * rows, grid.x);
    label(-0.17 * cols, rows / 2, grid.y);

  };

  const holder = h("div", { class: "jv-plot jv-plot-3d", tabindex: "0", role: "application", "aria-roledescription": "3D view",
    "aria-label": "3D view. Left and right arrows rotate, up and down tilt, plus and minus zoom, 0 resets the view." }, canvas);
  const change = (yaw, pitch, zoom) => {
    camera.yaw = ((camera.yaw + yaw) % 360 + 360) % 360;
    camera.pitch = Math.max(5, Math.min(85, camera.pitch + pitch));
    camera.zoom = Math.max(0.4, Math.min(4, camera.zoom * zoom));
    draw();
  };
  const reset = () => { Object.assign(camera, DEFAULT_CAMERA); draw(); };
  holder.addEventListener("keydown", (event) => {
    const keys = { ArrowLeft: [-15, 0, 1], ArrowRight: [15, 0, 1], ArrowUp: [0, 5, 1], ArrowDown: [0, -5, 1],
      "+": [0, 0, 1.2], "=": [0, 0, 1.2], "-": [0, 0, 1 / 1.2] };
    if (event.key === "0") { event.preventDefault(); reset(); return; }
    const move = keys[event.key];
    if (!move) return;
    event.preventDefault();
    change(...move);
  });
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    change((event.clientX - drag.x) * 0.5, (event.clientY - drag.y) * 0.3, 1);
    drag = { x: event.clientX, y: event.clientY };
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", (event) => {
    // pinch on a trackpad (ctrl + wheel) zooms; a plain wheel keeps scrolling the page
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    change(0, 0, event.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, { passive: false });
  const button = (label, title, action) => h("button", { type: "button", class: "button small", title, "aria-label": title, onclick: action }, label);
  const controls = h("div", { class: "jv-3d-controls", role: "group", "aria-label": "3D view" },
    button("↺", "Rotate left", () => change(-15, 0, 1)), button("↻", "Rotate right", () => change(15, 0, 1)),
    button("▲", "Tilt up", () => change(0, 5, 1)), button("▼", "Tilt down", () => change(0, -5, 1)),
    button("+", "Zoom in", () => change(0, 0, 1.2)), button("−", "Zoom out", () => change(0, 0, 1 / 1.2)),
    button("Reset", "Reset the view", reset));
  // follow light/dark mode
  const scheme = globalThis.matchMedia?.("(prefers-color-scheme: dark)");
  scheme?.addEventListener?.("change", () => { if (canvas.isConnected) draw(); });
  queueMicrotask(draw);
  return h("div", { class: "jv-3d" }, controls, holder);
}

// ---- table ------------------------------------------------------------------------------------

function gridTable(grid) {
  const head = h("tr", {}, h("th", { scope: "col" }, `${grid.y} \\ ${grid.x}`),
    ...grid.x_values.map((v) => h("th", { scope: "col", class: "num" }, String(v))));
  const body = [...grid.y_values.keys()].reverse().map((r) => h("tr", {},
    h("th", { scope: "row", class: "num" }, String(grid.y_values[r])),
    ...grid.cells[r].map((cell) => h("td", { class: "num", title: cell.prob + (cell.upper ? ` (at most ${cell.upper})` : "") },
      cell.p === null ? cell.prob : formatValue(cell.prob, cell.p, 4)))));
  return h("div", { class: "table-wrap jv-table" }, h("table", { class: "rv-table" },
    h("caption", { class: "rv-note" }, `Pr[${grid.x} = column, ${grid.y} = row]; hover a value for its exact form.`),
    h("thead", {}, head), h("tbody", {}, body)));
}

// ---- legend and mass ----------------------------------------------------------------------------

function legend(colours) {
  const stops = Array.from({ length: 11 }, (_, i) => `${viridis(i / 10)} ${i * 10}%`).join(", ");
  const position = (value) => {
    if (colours.scale === "log") return colours.max === colours.min ? 100 : ((Math.log10(value) - Math.log10(colours.min)) / (Math.log10(colours.max) - Math.log10(colours.min))) * 100;
    return colours.max > 0 ? (value / colours.max) * 100 : 0;
  };
  return h("div", { class: "jv-legend" },
    h("span", { class: "jv-swatch jv-zero", "aria-hidden": "true" }), h("span", {}, "0"),
    h("div", { class: "jv-gradient-wrap" },
      h("div", { class: "jv-gradient", style: `background: linear-gradient(to right, ${stops})`, "aria-hidden": "true" }),
      h("div", { class: "jv-gradient-ticks" }, ...colours.ticks.map((tick) => h("span", {
        style: `left: ${Math.max(0, Math.min(100, position(tick)))}%` }, formatNumber(tick, 2))))),
    h("span", { class: "rv-muted" }, colours.scale === "log" ? "log scale" : "linear scale"));
}

function massBar(grid) {
  const residual = grid.residual_mass_float || 0;
  const shown = grid.shown_mass_float;
  const outside = grid.outside_mass_float;
  const parts = [];
  const width = 300;
  const hatch = `jv-hatch-${++idCounter}`;
  let x = 0;
  const total = (shown ?? 0) + Math.max(0, outside ?? 0) + residual;
  for (const [kind, value] of [["shown", shown], ["outside", outside], ["residual", residual]]) {
    if (!(value > 0) || !(total > 0)) continue;
    const w = Math.max(1.5, (value / Math.max(1, total)) * width);
    parts.push(s("rect", { class: `jv-mass-${kind}`, x, y: 0, width: w, height: 10, fill: kind === "residual" ? `url(#${hatch})` : null }));
    x += w;
  }
  const items = [
    h("span", {}, h("span", { class: "jv-key jv-mass-shown", "aria-hidden": "true" }), `In the window: ${formatValue(grid.shown_mass, shown)}`),
    h("span", {}, h("span", { class: "jv-key jv-mass-outside", "aria-hidden": "true" }), `Outside the window: ${formatValue(grid.outside_mass, outside)}`),
  ];
  if (grid.residual_mass && residual > 0) {
    items.push(h("span", {}, h("span", { class: "jv-key jv-mass-residual", "aria-hidden": "true" }),
      `? Not yet accounted for: ${formatValue(grid.residual_mass, residual)} — may end up in any cell, outside the window, or never terminate`));
  }
  return h("div", { class: "jv-mass" },
    s("svg", { class: "jv-mass-bar", viewBox: `0 0 ${width} 10`, preserveAspectRatio: "none", "aria-hidden": "true" },
      s("defs", {}, hatchPattern(hatch)), s("rect", { class: "jv-mass-track", width, height: 10, rx: 3 }), ...parts),
    h("div", { class: "jv-mass-items" }, ...items));
}

// ---- the section ----------------------------------------------------------------------------------

const VIEWS = { heatmap: "Heatmap", bars: "3D", table: "Table" };

function numberField(label, value, max) {
  return h("label", { class: "jv-field" }, h("span", {}, label),
    h("input", { class: "field num", type: "number", min: "0", max: String(max), step: "1", value: String(value), inputmode: "numeric" }));
}

/**
 * The joint view of two variables. `load({x, y, x_window, y_window}, signal)` resolves to a grid
 * (prodigy/analysis/joint_grid.py) or rejects with an Error. `view` keeps the choices across
 * re-renders of the result. `scale()`/`setScale()` share the linear/log choice with the bar charts.
 */
export function jointView({ variables, load, view, maxAxis = 50, scale, setScale }) {
  if (!view.x || !variables.includes(view.x)) view.x = variables[0];
  if (!view.y || !variables.includes(view.y) || view.y === view.x) view.y = variables.find((v) => v !== view.x);
  view.mode = view.mode || "heatmap";

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
  const modeToggle = h("div", { class: "segmented small", role: "group", "aria-label": "Show the joint distribution as" });
  const scaleToggle = h("div", { class: "segmented small", role: "group", "aria-label": "Colour scale" });
  let grid = null;
  let controller = null;

  const fieldValue = (field) => Number(field.querySelector("input").value);
  const setFields = (g) => {
    xFrom.querySelector("input").value = g.x_values[0];
    xTo.querySelector("input").value = g.x_values[g.x_values.length - 1];
    yFrom.querySelector("input").value = g.y_values[0];
    yTo.querySelector("input").value = g.y_values[g.y_values.length - 1];
  };

  const draw = () => {
    for (const button of modeToggle.children) button.setAttribute("aria-pressed", String(button.dataset.mode === view.mode));
    for (const button of scaleToggle.children) button.setAttribute("aria-pressed", String(button.dataset.scale === scale()));
    scaleToggle.hidden = view.mode === "table";
    if (!grid) return;
    const colours = colourScale(grid.cells.flat().map((cell) => cell.p), scale());
    const width = body.clientWidth || 560;
    const symbolic = grid.cells.some((row) => row.some((cell) => cell.p === null));
    const notes = [];
    if (grid.bound === "lower") notes.push("Every value is a lower bound.");
    if (grid.bound === "upper") notes.push("Every value is an upper bound (Park's lemma).");
    if (grid.bound === "unknown") notes.push("Values without a guarantee.");
    if (symbolic) notes.push("Hatched cells depend on parameters; the table lists them.");
    const parts = [];
    if (view.mode === "heatmap") parts.push(heatmap(grid, colours, view, readout, width), readout);
    else if (view.mode === "bars") parts.push(bars3d(grid, colours, view, width));
    else parts.push(gridTable(grid));
    if (view.mode !== "table") parts.push(legend(colours));
    parts.push(massBar(grid));
    if (notes.length) parts.push(h("p", { class: "rv-note" }, notes.join(" ")));
    body.replaceChildren(...parts);
  };

  const fetchGrid = async (useWindow) => {
    controller?.abort();
    controller = new AbortController();
    const { signal } = controller;
    view.x = xSelect.value;
    view.y = ySelect.value;
    if (view.x === view.y) {
      status.textContent = "Choose two different variables.";
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
    try {
      const result = await load(payload, signal);
      if (signal.aborted) return;
      grid = result;
      view.xWindow = [grid.x_values[0], grid.x_values[grid.x_values.length - 1]];
      view.yWindow = [grid.y_values[0], grid.y_values[grid.y_values.length - 1]];
      setFields(grid);
      status.textContent = grid.finite_support && grid.support
        ? `Support: ${grid.x} ${grid.support[grid.x][0]}–${grid.support[grid.x][1]}, ${grid.y} ${grid.support[grid.y][0]}–${grid.support[grid.y][1]}.`
        : "Infinite support; adjust the window to see other values.";
      draw();
    } catch (error) {
      if (signal.aborted || error.name === "AbortError") return;
      status.textContent = `Not available: ${error.message}`;
    } finally {
      if (!signal.aborted) body.removeAttribute("aria-busy");
    }
  };

  for (const [key, label] of Object.entries(VIEWS)) {
    modeToggle.append(h("button", { type: "button", "data-mode": key, "aria-pressed": "false", onclick: () => { view.mode = key; draw(); } }, label));
  }
  for (const [key, label] of Object.entries({ linear: "Linear", log: "Log" })) {
    scaleToggle.append(h("button", { type: "button", "data-scale": key, "aria-pressed": "false",
      title: key === "log" ? "Logarithmic colour scale: small probabilities stay visible" : "Linear colour scale",
      onclick: () => { setScale(key); draw(); } }, label));
  }
  const pairChanged = () => {
    view.xWindow = null;
    view.yWindow = null;
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
  const form = h("form", { class: "jv-controls", onsubmit: (event) => { event.preventDefault(); fetchGrid(true); } },
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
  draw();
  fetchGrid(Boolean(view.xWindow && view.yWindow));
  return h("div", { class: "jv" },
    h("div", { class: "rv-subhead" }, h("h4", {}, "Joint distribution"), h("div", { class: "jv-toggles" }, scaleToggle, modeToggle)),
    form, status, body);
}
