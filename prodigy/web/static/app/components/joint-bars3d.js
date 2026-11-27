// The 3D bars of a joint distribution (joint-view.js): a canvas with a camera that the pointer and the keys
// turn, bars coloured like the heatmap, and the same readout of the active cell.
import { DEFAULT_VIEW, ownUpperEnds } from "./charts.js";
import { h } from "./dom.js";
import { formatValue } from "./format.js";
import { activeCell, cellMove, cellParts, cellText } from "./joint-cells.js";
import { shade } from "./plot-scale.js";

// ---- 3D bars ----------------------------------------------------------------------------------

const DEFAULT_CAMERA = { yaw: -35, pitch: 32, zoom: 1 };

/** Whether the point (x, y) lies inside the convex or simple polygon `points` (even-odd rule). */
function inside(points, x, y) {
  let hit = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [a, b] = [points[i], points[j]];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) hit = !hit;
  }
  return hit;
}

/** `colour` (#rrggbb) with opacity `alpha` as rgba(). */
function translucent(colour, alpha) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(colour.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** A canvas pattern of diagonal lines, the hatching of the SVG charts (charts.js: hatchPattern). */
function canvasHatch(ctx, background, line) {
  const tile = document.createElement("canvas");
  tile.width = 8;
  tile.height = 8;
  const g = tile.getContext("2d");
  if (!g) return background;
  g.fillStyle = background;
  g.fillRect(0, 0, 8, 8);
  g.strokeStyle = line;
  g.lineWidth = 2;
  g.beginPath();
  for (const [x0, y0, x1, y1] of [[-2, 10, 10, -2], [-2, 2, 2, -2], [6, 10, 10, 6]]) { g.moveTo(x0, y0); g.lineTo(x1, y1); }
  g.stroke();
  return ctx.createPattern(tile, "repeat") || background;
}

/**
 * The 3D bars of a grid. With a lower bound (#452), the part of a cell's interval above its bar is a
 * glass or outlined box (`options.intervals`): for the cells with an upper end of their own (charts.js:
 * ownUpperEnds) always, for the others only while the cell is hovered or selected. The mass not yet
 * accounted for is one hatched "?" tower beside the grid on the same scale (`options.unaccounted`).
 */
export function bars3d(grid, colours, view, readout, width, options = DEFAULT_VIEW) {
  const camera = view.camera || (view.camera = { ...DEFAULT_CAMERA });
  const cols = grid.x_values.length;
  const rows = grid.y_values.length;
  const cssW = Math.max(280, Math.min(720, Math.round(width)));
  // the window is drawn on a square floor of side SIDE, whatever its numbers of rows and columns
  const SIDE = 10;
  const heightUnits = SIDE * 0.55;
  // the height of a probability in the scene (above the largest value for the "?" tower), null for none
  const heightOf = (p) => {
    const value = colours.height(p);
    return value === null ? null : Math.max(0.02, value) * heightUnits;
  };
  const lower = grid.bound === "lower";
  const own = lower && ownUpperEnds(grid.cells.flat());
  const upperOf = (r, c) => {
    const cell = grid.cells[r][c];
    return lower && Number.isFinite(cell.upper_p) && Number.isFinite(cell.p) && cell.upper_p > cell.p ? cell.upper_p : null;
  };
  const residual = lower && options.unaccounted && grid.residual_mass_float > 0 ? grid.residual_mass_float : null;
  // the "?" tower stands beside the front right corner of the grid
  const towerW = Math.max(0.8, cols * 0.08);
  const towerD = Math.max(0.8, rows * 0.08);
  const tower = residual === null ? null
    : { x0: cols + Math.max(0.5, cols * 0.05), y0: 0, z: heightOf(residual) };
  if (tower) Object.assign(tower, { x1: tower.x0 + towerW, y1: towerD });
  // the tops of the bars, which decide how much room the scene needs above the floor
  const tops = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const z = heightOf(grid.cells[r][c].p);
      const up = own && grid.cells[r][c].upper_source === "certificate" ? heightOf(upperOf(r, c)) : null;
      if (z !== null) tops.push({ x0: c, x1: c + 1, y0: r, y1: r + 1, z: Math.max(z, up ?? 0) });
    }
  }
  if (tower) tops.push({ ...tower, z: tower.z + 0.6 });
  const projection = (cam, scale, cx, cy) => {
    const yaw = (cam.yaw * Math.PI) / 180;
    const pitch = (cam.pitch * Math.PI) / 180;
    return (gx, gy, z) => {
      const X = (gx / cols - 0.5) * SIDE;
      const Y = (gy / rows - 0.5) * SIDE;
      const u = X * Math.cos(yaw) - Y * Math.sin(yaw);
      const v = X * Math.sin(yaw) + Y * Math.cos(yaw);
      return { x: cx + u * scale, y: cy - (z * Math.cos(pitch) + v * Math.sin(pitch)) * scale, depth: v };
    };
  };
  // the box around floor, bars and axis labels at scale 1, for a camera
  const extent = (cam) => {
    const project = projection(cam, 1, 0, 0);
    const points = [
      project(0, 0, 0), project(cols, 0, 0), project(cols, rows, 0), project(0, rows, 0),
      // where the axis labels are
      project(0, -0.2 * rows, 0), project(cols, -0.2 * rows, 0), project(-0.2 * cols, 0, 0), project(-0.2 * cols, rows, 0),
    ];
    for (const { x0, x1, y0, y1, z } of tops) points.push(project(x0, y0, z), project(x1, y0, z), project(x0, y1, z), project(x1, y1, z));
    if (tower) points.push(project(tower.x1, tower.y0, 0), project(tower.x1, tower.y1, 0));
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  };
  const PAD = 12;
  // the height follows the shape of the scene at the default camera, so that low bars leave no empty space
  const first = extent(view.camera || DEFAULT_CAMERA);
  const cssH = Math.max(160, Math.min(Math.round(cssW * 0.75), Math.round(((first.y1 - first.y0) / (first.x1 - first.x0)) * (cssW - 2 * PAD) + 2 * PAD)));
  const canvas = h("canvas", { class: "jv-canvas", width: cssW, height: cssH, style: `width:${cssW}px;height:${cssH}px`,
    "aria-hidden": "true" });
  // the tooltip of the bar under the pointer (or the bar chosen with the keyboard); never the only
  // place of a value: the line below the view says the same and is read by screen readers
  const tipWhere = h("span", { class: "jv-tip-where" });
  const tipValue = h("span", { class: "jv-tip-value" });
  const tip = h("div", { class: "jv-tip", "aria-hidden": "true", hidden: true }, tipWhere, tipValue);
  // what the last drawing put where, for the pointer: per cell its floor and, for a bar (and the "?"
  // tower, kind "tower"), its outline, in the order of drawing
  let shapes = { bars: [], floor: [] };
  let hover = false; // the pointer is over a bar or a floor cell
  let hoverTower = false; // the pointer is over the "?" tower
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
    const boundColour = style.getPropertyValue("--rv-bound").trim() || "#7A3FA8";
    const surface = style.getPropertyValue("--rv-surface").trim() || "#FFFFFF";
    // fit the scene into the canvas; zoom magnifies around its centre
    const box = extent(camera);
    const fit = Math.min((cssW - 2 * PAD) / (box.x1 - box.x0), (cssH - 2 * PAD) / (box.y1 - box.y0));
    const scale = camera.zoom * fit;
    const project = projection(camera, scale, cssW / 2 - ((box.x0 + box.x1) / 2) * scale, cssH / 2 - ((box.y0 + box.y1) / 2) * scale);
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
    // a box from z0 to z1 over [x0, x1] × [y0, y1]: its two sides that face the camera and its top
    const prism = (x0, x1, y0, y1, z0, z1) => {
      const faces = [
        [[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]],
      ].map(([[ax, ay], [bx, by]], i) => ({
        points: [project(ax, ay, z0), project(bx, by, z0), project(bx, by, z1), project(ax, ay, z1)],
        depth: project((ax + bx) / 2, (ay + by) / 2, 0).depth,
        factor: i % 2 ? 0.62 : 0.78,
      })).sort((a, b) => b.depth - a.depth);
      return { sides: faces.slice(2), top: [project(x0, y0, z1), project(x1, y0, z1), project(x1, y1, z1), project(x0, y1, z1)] };
    };
    const inset = 0.12;
    // the part of a cell's interval above its bar: glass (translucent, dashed edges) or outline (dashed edges)
    const intervalBox = (r, c, colour) => {
      const z0 = heightOf(grid.cells[r][c].p) ?? 0;
      const z1 = heightOf(upperOf(r, c));
      if (z1 === null || !(z1 > z0)) return;
      const { sides, top } = prism(c + inset, c + 1 - inset, r + inset, r + 1 - inset, z0, z1);
      ctx.setLineDash([4, 3]);
      for (const points of [...sides, { points: top }].map((face) => face.points)) {
        polygon(points, options.intervals === "glass" ? translucent(colour, 0.28) : null, boundColour, 1.25);
      }
      ctx.setLineDash([]);
    };
    // bars and the "?" tower, the farthest first (painter's algorithm)
    const list = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = colours.t(grid.cells[r][c].p);
        if (t !== null) list.push({ r, c, t, depth: project(c + 0.5, r + 0.5, 0).depth });
      }
    }
    if (tower) list.push({ kind: "tower", depth: project((tower.x0 + tower.x1) / 2, (tower.y0 + tower.y1) / 2, 0).depth });
    list.sort((a, b) => b.depth - a.depth);
    const bars = [];
    for (const bar of list) {
      if (bar.kind === "tower") {
        const { sides, top } = prism(tower.x0, tower.x1, tower.y0, tower.y1, 0, tower.z);
        const hatch = canvasHatch(ctx, surface, boundColour);
        for (const face of sides) {
          polygon(face.points, hatch, boundColour);
          polygon(face.points, `rgba(0, 0, 0, ${face.factor > 0.7 ? 0.06 : 0.14})`, null);
        }
        polygon(top, hatch, boundColour);
        // labelled, so that it is not told apart by its hatching alone
        const peak = project((tower.x0 + tower.x1) / 2, (tower.y0 + tower.y1) / 2, tower.z);
        ctx.fillStyle = boundColour;
        ctx.font = "600 13px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("?", peak.x, peak.y - 6);
        bars.push({ kind: "tower", polygons: [top, ...sides.map((face) => face.points)] });
        continue;
      }
      const z = heightOf(grid.cells[bar.r][bar.c].p);
      const colour = colours.colour(bar.t);
      const { sides, top } = prism(bar.c + inset, bar.c + 1 - inset, bar.r + inset, bar.r + 1 - inset, 0, z);
      for (const face of sides) polygon(face.points, shade(colour, face.factor), null);
      polygon(top, colour, shade(colour, 0.55));
      if (own && grid.cells[bar.r][bar.c].upper_source === "certificate") intervalBox(bar.r, bar.c, colour);
      bars.push({ r: bar.r, c: bar.c, polygons: [top, ...sides.map((face) => face.points)] });
    }
    shapes = { bars, floor };
    // the active cell: its bar outlined, or its place on the floor for probability 0; drawn after
    // all bars, so that it shows through bars in front of it; with its interval above it
    if (view.active) {
      const [ar, ac] = view.active;
      const bar = bars.find((b) => b.r === ar && b.c === ac);
      ctx.setLineDash([]);
      // a halo in the surface colour under the outline keeps it visible on dark bars (greyscale)
      if (bar) for (const points of bar.polygons) { polygon(points, null, surface, 4); polygon(points, null, strong, 2); }
      else {
        ctx.setLineDash([3, 3]);
        polygon(floor[ar * cols + ac].points, null, strong, 2);
        ctx.setLineDash([]);
      }
      const t = colours.t(grid.cells[ar][ac].p);
      if (!(own && grid.cells[ar][ac].upper_source === "certificate")) intervalBox(ar, ac, colours.colour(t ?? 0));
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
  // (or above the "?" tower while the pointer is over it)
  function placeTip() {
    const onTower = hoverTower && !drag?.moved && Boolean(tower);
    const shown = onTower || (Boolean(view.active) && !drag?.moved && (hover || pinned));
    tip.hidden = !shown;
    if (!shown) return;
    let points;
    if (onTower) {
      tipWhere.textContent = "? Not yet accounted for";
      tipValue.textContent = formatValue(grid.residual_mass, grid.residual_mass_float);
      points = shapes.bars.find((b) => b.kind === "tower")?.polygons.flat() || [];
    } else {
      const [ar, ac] = view.active;
      const { where, value } = cellParts(grid, ar, ac);
      tipWhere.textContent = where;
      tipValue.textContent = value;
      const bar = shapes.bars.find((b) => b.r === ar && b.c === ac);
      points = bar ? bar.polygons.flat() : shapes.floor[ar * cols + ac]?.points || [];
    }
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
  // the cell under a point of the canvas: the frontmost bar (or "tower"), else the floor cell
  const cellAt = (event) => {
    const box = canvas.getBoundingClientRect();
    const [x, y] = [event.clientX - box.left, event.clientY - box.top];
    for (let i = shapes.bars.length - 1; i >= 0; i--) {
      const bar = shapes.bars[i];
      if (bar.polygons.some((points) => inside(points, x, y))) return bar.kind === "tower" ? "tower" : [bar.r, bar.c];
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
    hoverTower = cell === "tower";
    hover = Boolean(cell) && !hoverTower;
    if (!hover) { placeTip(); return; }
    if (!view.active || cell[0] !== view.active[0] || cell[1] !== view.active[1]) select(...cell);
    else placeTip();
  });
  canvas.addEventListener("pointerup", (event) => {
    // a click without dragging selects the bar under the pointer and focuses the view
    if (drag && !drag.moved) {
      const cell = cellAt(event);
      if (Array.isArray(cell)) select(...cell);
      holder.focus({ preventScroll: true });
    }
    drag = null;
    placeTip();
  });
  canvas.addEventListener("pointerleave", () => { if (!drag) { hover = false; hoverTower = false; placeTip(); } });
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
