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
 * without a number (parameters), which get no colour. `atLeast` raises the largest value, so that
 * the frames of an animation share one scale (#289).
 */
export function colourScale(values, scale = "linear", atLeast = 0) {
  const positive = values.filter((p) => Number.isFinite(p) && p > 0);
  const max = positive.length ? Math.max(atLeast, ...positive) : 0;
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

/**
 * The values of a cell and its probability, formatted alike in the heatmap's readout and the 3D
 * view's tooltip: {where: "x = 1, y = 2", value: "at least 0.0277778 (1/36); at most 0.031684"}.
 */
export function cellParts(grid, r, c) {
  const cell = grid.cells[r][c];
  const where = `${grid.x} = ${grid.x_values[c]}, ${grid.y} = ${grid.y_values[r]}`;
  const value = cell.p === null ? cell.prob : formatValue(cell.prob, cell.p, 6);
  const qualifier = cell.prob === "0" && grid.bound !== "upper" ? "" : (BOUND_WORDS[grid.bound] || "");
  const exact = cell.p !== null && cell.prob && cell.prob.length < 40 && cell.prob !== String(cell.p) ? ` (${cell.prob})` : "";
  const upper = cell.upper ? `; at most ${formatValue(cell.upper, cell.upper_p, 6)}` : "";
  return { where, value: `${qualifier}${value}${exact}${upper}` };
}

function cellText(grid, r, c) {
  const { where, value } = cellParts(grid, r, c);
  return `${where}: ${value}`;
}

/** The cell [r, c] after an arrow, Home/End or Page Up/Down key (y grows upwards), or null. */
function cellMove(key, [r, c], rows, cols) {
  const moves = { ArrowLeft: [r, c - 1], ArrowRight: [r, c + 1], ArrowUp: [r + 1, c], ArrowDown: [r - 1, c],
    Home: [r, 0], End: [r, cols - 1], PageUp: [rows - 1, c], PageDown: [0, c] };
  const target = moves[key];
  return target ? [Math.max(0, Math.min(rows - 1, target[0])), Math.max(0, Math.min(cols - 1, target[1]))] : null;
}

/** The active cell of `view` if it lies in a rows × cols window, else [0, 0]. */
function activeCell(view, rows, cols) {
  return view.active && view.active[0] < rows && view.active[1] < cols ? view.active : [0, 0];
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

// ---- 3D bars ----------------------------------------------------------------------------------

const DEFAULT_CAMERA = { yaw: -35, pitch: 32, zoom: 1 };

function shade(colour, factor) {
  return toHex(hex(colour).map((v) => v * factor));
}

/** Whether the point (x, y) lies inside the convex or simple polygon `points` (even-odd rule). */
function inside(points, x, y) {
  let hit = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [a, b] = [points[i], points[j]];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) hit = !hit;
  }
  return hit;
}

function bars3d(grid, colours, view, readout, width) {
  const camera = view.camera || (view.camera = { ...DEFAULT_CAMERA });
  const cols = grid.x_values.length;
  const rows = grid.y_values.length;
  const cssW = Math.max(280, Math.min(720, Math.round(width)));
  const cssH = Math.round(cssW * 0.62);
  const canvas = h("canvas", { class: "jv-canvas", width: cssW, height: cssH, style: `width:${cssW}px;height:${cssH}px`,
    "aria-hidden": "true" });
  // the tooltip of the bar under the pointer (or the bar chosen with the keyboard); never the only
  // place of a value: the line below the view says the same and is read by screen readers
  const tipWhere = h("span", { class: "jv-tip-where" });
  const tipValue = h("span", { class: "jv-tip-value" });
  const tip = h("div", { class: "jv-tip", "aria-hidden": "true", hidden: true }, tipWhere, tipValue);
  // what the last drawing put where, for the pointer: per cell its floor and, for a bar, its outline
  let shapes = { bars: [], floor: [] };
  let hover = false; // the pointer is over a bar or a floor cell
  let pinned = false; // the view has the focus (keyboard, click or tap)
  let drag = null;

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
    const strong = style.getPropertyValue("--rv-text").trim() || "#1D1D1F";
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
    const polygon = (points, fill, stroke, lineWidth = 1) => {
      ctx.beginPath();
      points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
    };
    // floor and its labels
    polygon([project(0, 0, 0), project(cols, 0, 0), project(cols, rows, 0), project(0, rows, 0)], null, line);
    const floor = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        floor.push({ r, c, points: [project(c, r, 0), project(c + 1, r, 0), project(c + 1, r + 1, 0), project(c, r + 1, 0)] });
      }
    }
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
    const bars = [];
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
      const visible = faces.slice(2);
      const top = [project(x0, y0, z), project(x1, y0, z), project(x1, y1, z), project(x0, y1, z)];
      for (const face of visible) polygon(face.points, shade(colour, face.factor), null);
      polygon(top, colour, shade(colour, 0.55));
      bars.push({ r: bar.r, c: bar.c, polygons: [top, ...visible.map((face) => face.points)] });
    }
    shapes = { bars, floor };
    // the active cell: its bar outlined, or its place on the floor for probability 0; drawn after
    // all bars, so that it shows through bars in front of it
    if (view.active) {
      const [ar, ac] = view.active;
      const bar = bars.find((b) => b.r === ar && b.c === ac);
      ctx.setLineDash([]);
      if (bar) for (const points of bar.polygons) polygon(points, null, strong, 2);
      else {
        ctx.setLineDash([3, 3]);
        polygon(floor[ar * cols + ac].points, null, strong, 2);
        ctx.setLineDash([]);
      }
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
    placeTip();
  };

  // the tooltip sits above the active bar (or its floor cell), kept inside the canvas
  function placeTip() {
    const shown = Boolean(view.active) && !drag?.moved && (hover || pinned);
    tip.hidden = !shown;
    if (!shown) return;
    const [ar, ac] = view.active;
    const { where, value } = cellParts(grid, ar, ac);
    tipWhere.textContent = where;
    tipValue.textContent = value;
    const bar = shapes.bars.find((b) => b.r === ar && b.c === ac);
    const points = bar ? bar.polygons.flat() : shapes.floor[ar * cols + ac]?.points || [];
    if (!points.length) return;
    const xMid = points.reduce((sum, p) => sum + p.x, 0) / points.length;
    const yTop = Math.min(...points.map((p) => p.y));
    const [w, hgt] = [tip.offsetWidth, tip.offsetHeight];
    const left = Math.max(4, Math.min(cssW - w - 4, xMid - w / 2));
    const above = yTop - hgt - 8;
    const below = Math.max(...points.map((p) => p.y)) + 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(4, above >= 4 ? above : Math.min(cssH - hgt - 4, below))}px`;
  }

  const select = (r, c) => {
    view.active = [r, c];
    readout.textContent = cellText(grid, r, c);
    draw();
  };
  // the cell under a point of the canvas: the frontmost bar, else the floor cell
  const cellAt = (event) => {
    const box = canvas.getBoundingClientRect();
    const [x, y] = [event.clientX - box.left, event.clientY - box.top];
    for (let i = shapes.bars.length - 1; i >= 0; i--) {
      const bar = shapes.bars[i];
      if (bar.polygons.some((points) => inside(points, x, y))) return [bar.r, bar.c];
    }
    const cell = shapes.floor.find((f) => inside(f.points, x, y));
    return cell ? [cell.r, cell.c] : null;
  };

  const stage = h("div", { class: "jv-stage" }, canvas, tip);
  const holder = h("div", { class: "jv-plot jv-plot-3d", tabindex: "0", role: "application", "aria-roledescription": "3D view",
    "aria-label": `3D bar chart of the joint distribution of ${grid.x} and ${grid.y}, ${cols} by ${rows} cells. ` +
      "Arrow keys move between bars, Home and End go to the first and last column, Page Up and Page Down to the top and bottom row. " +
      "Shift with the arrow keys rotates and tilts the view, plus and minus zoom, 0 resets the view.",
    "aria-describedby": readout.id }, stage);
  const change = (yaw, pitch, zoom) => {
    camera.yaw = ((camera.yaw + yaw) % 360 + 360) % 360;
    camera.pitch = Math.max(5, Math.min(85, camera.pitch + pitch));
    camera.zoom = Math.max(0.4, Math.min(4, camera.zoom * zoom));
    draw();
  };
  const reset = () => { Object.assign(camera, DEFAULT_CAMERA); draw(); };
  holder.addEventListener("keydown", (event) => {
    if (event.key === "0") { event.preventDefault(); reset(); return; }
    const camera3d = { ArrowLeft: [-15, 0, 1], ArrowRight: [15, 0, 1], ArrowUp: [0, 5, 1], ArrowDown: [0, -5, 1] };
    const zooms = { "+": [0, 0, 1.2], "=": [0, 0, 1.2], "-": [0, 0, 1 / 1.2] };
    const move = event.shiftKey ? camera3d[event.key] : null;
    if (move || zooms[event.key]) {
      event.preventDefault();
      change(...(move || zooms[event.key]));
      return;
    }
    const target = cellMove(event.key, view.active || [0, 0], rows, cols);
    if (!target) return;
    event.preventDefault();
    select(...target);
  });
  holder.addEventListener("focus", () => { pinned = true; placeTip(); });
  holder.addEventListener("blur", () => { pinned = false; placeTip(); });
  canvas.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, y: event.clientY, moved: false };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (drag) {
      const [dx, dy] = [event.clientX - drag.x, event.clientY - drag.y];
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag = { x: event.clientX, y: event.clientY, moved: true };
      hover = false;
      change(dx * 0.5, dy * 0.3, 1);
      return;
    }
    const cell = cellAt(event);
    hover = Boolean(cell);
    if (!cell) { placeTip(); return; }
    if (!view.active || cell[0] !== view.active[0] || cell[1] !== view.active[1]) select(...cell);
    else placeTip();
  });
  canvas.addEventListener("pointerup", (event) => {
    // a click without dragging selects the bar under the pointer and focuses the view
    if (drag && !drag.moved) {
      const cell = cellAt(event);
      if (cell) select(...cell);
      holder.focus({ preventScroll: true });
    }
    drag = null;
    placeTip();
  });
  canvas.addEventListener("pointerleave", () => { if (!drag) { hover = false; placeTip(); } });
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
  const [r0, c0] = activeCell(view, rows, cols);
  view.active = [r0, c0];
  readout.textContent = cellText(grid, r0, c0);
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

// ---- colour key -----------------------------------------------------------------------------------
// The colour key says what colour one cell has for its own probability. The mass of the window is a
// different quantity (#286); the result view shows it in its mass line (charts.js: massSummary),
// with its own heading and a grey bar, so it never reads as part of the colour scale.

function keyHeading(id, title, detail) {
  return h("h4", { class: "jv-key-title", id }, title, h("span", { class: "jv-key-detail" }, detail));
}

function legend(colours) {
  const stops = Array.from({ length: 11 }, (_, i) => `${viridis(i / 10)} ${i * 10}%`).join(", ");
  const position = (value) => {
    if (colours.scale === "log") return colours.max === colours.min ? 100 : ((Math.log10(value) - Math.log10(colours.min)) / (Math.log10(colours.max) - Math.log10(colours.min))) * 100;
    return colours.max > 0 ? (value / colours.max) * 100 : 0;
  };
  const id = `jv-legend-${++idCounter}`;
  const detail = colours.scale === "log"
    ? ` — logarithmic, from ${formatNumber(colours.min, 2)} to ${formatNumber(colours.max, 2)}`
    : ` — linear, from 0 to ${formatNumber(colours.max, 2)}`;
  return h("section", { class: "jv-key-block jv-legend", "aria-labelledby": id },
    keyHeading(id, "Colour: probability of one cell", detail),
    h("div", { class: "jv-legend-row" },
      h("span", { class: "jv-legend-zero" }, h("span", { class: "jv-swatch jv-zero", "aria-hidden": "true" }), "0"),
      h("div", { class: "jv-gradient-wrap" },
        h("div", { class: "jv-gradient", style: `background: linear-gradient(to right, ${stops})`, "aria-hidden": "true" }),
        h("div", { class: "jv-gradient-ticks" }, ...colours.ticks.map((tick) => h("span", {
          style: `left: ${Math.max(0, Math.min(100, position(tick)))}%` }, formatNumber(tick, 2)))))));
}

// ---- the section ----------------------------------------------------------------------------------

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
 * re-renders of the result; `view.mode` is one of JOINT_VIEWS. `scale()` is the linear/log choice
 * shared with the bar charts. The switches for the view and the scale belong to the caller (the
 * result view has one control row for all its figures), which calls `draw()` after a change.
 * `onGrid(grid | null)` reports each grid shown (null while none is), for the caller's mass line.
 * `refresh()` loads the grid again for the same pair (the analysis card calls it when its timeline
 * moves to another position, and `load` then asks for that position's grid), in the window the user
 * chose with "Show window" or else in the position's default window. While the
 * pair and the window stay the same, the colour scale keeps the largest value seen so far, so that
 * frames can be compared (#289). Returns {element, draw, refresh}.
 */
export function jointView({ variables, load, view, maxAxis = 50, scale, onGrid = () => {} }) {
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
    const colours = colourScale(values, scale(), view.peak.max);
    const width = body.clientWidth || 560;
    const symbolic = grid.cells.some((row) => row.some((cell) => cell.p === null));
    // the guarantee is stated once, in the answer card; the readout says "at least" or "at most"
    const notes = [];
    if (symbolic) notes.push("Hatched cells depend on parameters; the table lists them.");
    const parts = [];
    if (view.mode === "heatmap") parts.push(heatmap(grid, colours, view, readout, width), readout);
    else if (view.mode === "bars") parts.push(bars3d(grid, colours, view, readout, width), readout);
    else parts.push(gridTable(grid));
    if (view.mode !== "table") parts.push(legend(colours));
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
    try {
      const result = await load(payload, signal);
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
      if (!signal.aborted) body.removeAttribute("aria-busy");
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
