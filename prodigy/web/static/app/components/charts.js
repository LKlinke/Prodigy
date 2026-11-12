// Charts of the result view: the interval bar of a query and the bar chart of a marginal.
// Plain SVG; colours come from CSS custom properties, so light/dark/high contrast follow.
import { h, s } from "./dom.js";
import { formatNumber, formatValue } from "./format.js";

let patternCounter = 0;

export function hatchPattern(id) {
  return s("pattern", { id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    s("rect", { width: 6, height: 6, class: "hatch-bg" }),
    s("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "hatch-line" }));
}

/**
 * Horizontal 0–1 track for a probability: the certain part solid, the part that is only
 * bounded hatched. `bound` is exact | lower | upper | unknown.
 */
export function intervalBar({ value, lower, upper, bound }, label) {
  const width = 200;
  const height = 10;
  const x = (p) => Math.max(0, Math.min(1, p ?? 0)) * width;
  const id = `hatch-${++patternCounter}`;
  const parts = [
    s("defs", {}, hatchPattern(id)),
    s("rect", { class: "ib-track", x: 0, y: 0, width, height, rx: 3 }),
  ];
  if (bound === "exact" && value !== null) {
    parts.push(s("rect", { class: "ib-solid", x: 0, y: 0, width: x(value), height, rx: 3 }));
  } else if (bound === "lower" && lower !== null) {
    const hi = upper ?? 1;
    parts.push(s("rect", { class: "ib-solid", x: 0, y: 0, width: x(lower), height, rx: 3 }));
    parts.push(s("rect", { class: "ib-hatch", x: x(lower), y: 0, width: Math.max(1.5, x(hi) - x(lower)), height, fill: `url(#${id})` }));
  } else if (bound === "upper" && upper !== null) {
    parts.push(s("rect", { class: "ib-hatch", x: 0, y: 0, width: Math.max(1.5, x(upper)), height, fill: `url(#${id})` }));
    parts.push(s("rect", { class: "ib-cap", x: Math.max(0, x(upper) - 1.5), y: -2, width: 1.5, height: height + 4 }));
  } else if (value !== null) {
    parts.push(s("rect", { class: "ib-marker", x: Math.max(0, x(value) - 1), y: -2, width: 2, height: height + 4 }));
  }
  return s("svg", { class: "interval-bar", viewBox: `0 -2 ${width} ${height + 4}`, preserveAspectRatio: "none",
    role: "img", "aria-label": label }, ...parts);
}

export function niceMax(value) {
  if (!(value > 0)) return 1;
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * base) return step * base;
  }
  return 10 * base;
}

const SUPERSCRIPT = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

/** 10^k as text, e.g. "10⁻³" (1 for k = 0). */
export function powerOfTen(k) {
  return k === 0 ? "1" : `10${String(k).replace(/./g, (c) => SUPERSCRIPT[c])}`;
}

/** At most this many decades are shown; smaller values get the minimal bar height. */
export const LOG_DECADES = 12;

/**
 * The decades of a logarithmic axis for the positive values among `values`: the axis runs from
 * 10^lo to 10^hi (whole decades, at most LOG_DECADES of them), `ticks` are the exponents that get
 * a grid line. Values that are 0 have no place on the axis; null when no value is positive.
 */
export function logAxis(values) {
  const positive = values.filter((p) => Number.isFinite(p) && p > 0);
  if (!positive.length) return null;
  const hi = Math.ceil(Math.log10(Math.max(...positive)) - 1e-9);
  // strictly below the smallest value, so that it keeps a visible bar
  let lo = Math.floor(Math.log10(Math.min(...positive)) - 1e-9);
  lo = Math.max(lo, hi - LOG_DECADES);
  const step = Math.ceil((hi - lo) / 4);
  const ticks = [];
  for (let k = hi; k >= lo; k -= step) ticks.push(k);
  return { lo, hi, ticks };
}

/**
 * Bar chart of a univariate coefficient table (see prodigy/analysis/result.py:
 * univariate_table). Extra bars: the mass beyond the shown values ("tail") and, for lower
 * bounds, the mass not yet accounted for (hatched).
 */
export function marginalChart(table, { residual = null, residualExact = null, variable = "x", bound = "exact", width: pixelWidth = 560, scale = "linear" } = {}) {
  const rows = (table.rows || []).filter((row) => row.p !== null);
  const extras = [];
  if (table.truncated && table.tail_mass_float && table.tail_mass_float > 1e-12) {
    const next = rows.length ? rows[rows.length - 1].value + 1 : 0;
    extras.push({ kind: "tail", label: `≥${next}`, p: table.tail_mass_float, exact: table.tail_mass,
      title: `${variable} ≥ ${next}: ${formatValue(table.tail_mass, table.tail_mass_float)} (not shown individually)` });
  }
  if (residual && residual > 0) {
    extras.push({ kind: "residual", label: "?", p: residual, exact: residualExact,
      title: `Not yet accounted for: ${formatValue(residualExact, residual)} — may end up at any value, or never terminate` });
  }
  if (!rows.length && !extras.length) {
    return h("p", { class: "rv-muted" }, table.rows?.length ? "The probabilities depend on parameters; see the table." : "No values.");
  }
  const bars = [...rows.map((row) => ({ kind: "value", label: String(row.value), p: row.p, exact: row.prob,
    title: `${variable} = ${row.value}: ${formatValue(row.prob, row.p)}${row.prob && row.prob.length < 30 ? ` (${row.prob})` : ""}` })),
  ...extras];

  const width = Math.max(260, Math.round(pixelWidth));
  const height = 168;
  const top = 14;
  const bottom = 22;
  const left = 4;
  const right = 40;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const id = `hatch-${++patternCounter}`;
  const labelEvery = Math.max(1, Math.ceil(bars.length / Math.max(1, Math.floor(plotW / 26))));
  const axis = scale === "log" ? logAxis(bars.map((b) => b.p)) : null;
  const max = axis ? 10 ** axis.hi : niceMax(Math.max(...bars.map((b) => b.p)));
  // the fraction of the plot height that a probability fills (log: decades above 10^lo)
  const fraction = axis
    ? (p) => (p > 0 ? Math.max(0, Math.min(1, (Math.log10(p) - axis.lo) / (axis.hi - axis.lo))) : 0)
    : (p) => p / max;
  const y = (p) => top + plotH * (1 - fraction(p));
  const slot = plotW / bars.length;
  const barW = Math.max(2, Math.min(36, slot * 0.72));

  const grid = axis
    ? axis.ticks.map((k) => s("g", { class: "mc-grid" },
      s("line", { x1: left, x2: left + plotW, y1: y(10 ** k), y2: y(10 ** k) }),
      s("text", { x: left + plotW + 6, y: y(10 ** k) + 4 }, powerOfTen(k))))
    : [0, max / 2, max].map((tick) => s("g", { class: "mc-grid" },
      s("line", { x1: left, x2: left + plotW, y1: y(tick), y2: y(tick) }),
      s("text", { x: left + plotW + 6, y: y(tick) + 4 }, formatNumber(tick, 2))));
  if (axis) {
    // the axis says that it is logarithmic (not only the toggle above it)
    grid.push(s("text", { class: "mc-scale", x: left + plotW + 6, y: top + plotH + 4 }, "log"));
  }

  const marks = bars.map((bar, index) => {
    const cx = left + slot * index + slot / 2;
    const barHeight = Math.max(bar.p > 0 ? 1 : 0, plotH * fraction(bar.p));
    const crowdedByExtra = extras.length > 0 && index === rows.length - 1 && slot < 26;
    const showLabel = bar.kind !== "value" || (index % labelEvery === 0 && !crowdedByExtra);
    const kindClass = bar.kind === "value" && (bound === "upper" || bound === "unknown") ? `mc-${bound}` : `mc-${bar.kind}`;
    const hatched = bar.kind === "residual" || (bar.kind === "value" && bound === "upper");
    // data-* keep the bar's own state, so that the analysis card can tween from one chart to the next (motion.js)
    return s("g", { class: `mc-bar ${kindClass}`, "data-key": bar.kind === "value" ? `v${bar.label}` : bar.kind },
      s("title", {}, bar.title),
      s("rect", { x: cx - barW / 2, y: top + plotH - barHeight, width: barW, height: barHeight, rx: Math.min(3, barW / 4),
        fill: hatched ? `url(#${id})` : null, "data-p": bar.p, "data-cx": cx, "data-w": barW }),
      // log scale: probability 0 has no bar height at all; an outline on the axis marks it as empty
      axis && bar.p === 0 ? s("rect", { class: "mc-zero", x: cx - barW / 2, y: top + plotH - 3, width: barW, height: 3 }) : null,
      showLabel ? s("text", { x: cx, y: height - 6, class: "mc-label" }, bar.label) : null);
  });

  const summary = `Bar chart of the distribution of ${variable}: ${rows.length} values` +
    (extras.length ? `, plus ${extras.map((e) => e.kind === "tail" ? "the remaining tail" : "the mass not yet accounted for").join(" and ")}` : "") +
    (axis ? `, logarithmic scale from ${powerOfTen(axis.lo)} to ${powerOfTen(axis.hi)}` : "") + ".";
  return s("svg", { class: "marginal-chart", viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": summary,
    "data-variable": variable, "data-max": max, "data-scale": axis ? "log" : "linear", "data-width": width, "data-top": top, "data-plot-h": plotH },
    s("defs", {}, hatchPattern(id)), ...grid, ...marks,
    s("line", { class: "mc-axis", x1: left, x2: left + plotW, y1: top + plotH, y2: top + plotH }));
}

/** The same data as a table with exact values (and upper bounds for lower-bound results). */
export function marginalTable(table, { variable = "x", bound = "exact" } = {}) {
  const withUpper = bound === "lower" && (table.rows || []).some((row) => row.upper);
  const head = h("tr", {}, h("th", { scope: "col" }, variable), h("th", { scope: "col" }, "Probability"),
    h("th", { scope: "col" }, "Exact"), withUpper ? h("th", { scope: "col" }, "At most") : null);
  const body = (table.rows || []).map((row) => h("tr", {},
    h("td", { class: "num" }, String(row.value)),
    h("td", { class: "num" }, formatValue(row.prob, row.p, 6)),
    h("td", { class: "mono exact-cell", title: row.prob || "" }, row.prob || ""),
    withUpper ? h("td", { class: "num" }, formatValue(row.upper, row.upper_p, 6)) : null));
  const rows = [...body];
  if (table.truncated && table.tail_mass) {
    rows.push(h("tr", { class: "tail-row" }, h("td", {}, "rest"), h("td", { class: "num" }, formatValue(table.tail_mass, table.tail_mass_float, 6)),
      h("td", { class: "mono exact-cell", title: table.tail_mass }, table.tail_mass), withUpper ? h("td", {}) : null));
  }
  return h("div", { class: "table-wrap" }, h("table", { class: "rv-table" }, h("thead", {}, head), h("tbody", {}, rows)));
}

/**
 * The mass line under a figure of the result view (#288): one labelled block for whatever the
 * figure shows, a marginal or a window of a joint distribution. A grey bar from 0 to 1 (its own
 * axis, so that it never reads as part of a colour scale, #286) and a list:
 *   total     the total mass of the output distribution
 *   shown     the mass of the values the figure shows (`shownLabel`)
 *   beyond    the mass of the other values (`beyondLabel`)
 *   residual  for a lower bound, the hatched "?" mass not yet accounted for (`residualLabel`)
 * Each part is {exact, p} or null; parts that are null are left out.
 */
export function massSummary({ total = null, shown = null, beyond = null, residual = null,
  shownLabel = "Shown", beyondLabel = "Beyond the shown values",
  residualLabel = "? Not yet accounted for — may still end with any value, or never terminate" } = {}) {
  const width = 300;
  const hatch = `hatch-${++patternCounter}`;
  const value = (part) => (part && Number.isFinite(part.p) ? part.p : 0);
  const pieces = shown || beyond
    ? [["shown", value(shown)], ["outside", Math.max(0, value(beyond))], ["residual", value(residual)]]
    : [["shown", value(total)], ["residual", value(residual)]];
  const sum = pieces.reduce((acc, [, p]) => acc + p, 0);
  const full = Math.max(1, sum);
  const rects = [];
  let x = 0;
  for (const [kind, p] of pieces) {
    if (!(p > 0)) continue;
    const w = Math.max(1.5, (p / full) * width);
    rects.push(s("rect", { class: `rv-mass-${kind}`, x, y: 0, width: w, height: 12, fill: kind === "residual" ? `url(#${hatch})` : null }));
    x += w;
  }
  const item = (key, label, part) => h("li", {},
    key ? h("span", { class: `rv-mass-key rv-mass-${key}`, "aria-hidden": "true" }) : h("span", { "aria-hidden": "true" }),
    h("span", {}, label),
    h("span", { class: "rv-mass-value", title: part.exact || "" }, formatValue(part.exact, part.p, 6)));
  const items = [];
  if (total) items.push(item(shown || beyond ? null : "shown", "Total mass", total));
  if (shown) items.push(item("shown", shownLabel, shown));
  if (beyond) items.push(item("outside", beyondLabel, beyond));
  if (residual && value(residual) > 0) {
    items.push(item("residual", residualLabel, residual));
  }
  const id = `rv-mass-${patternCounter}`;
  return h("section", { class: "rv-mass", "aria-labelledby": id },
    h("h4", { class: "rv-mass-title", id }, "Mass"),
    h("div", { class: "rv-mass-scale" },
      s("svg", { class: "rv-mass-bar", viewBox: `0 0 ${width} 12`, preserveAspectRatio: "none", "aria-hidden": "true" },
        s("defs", {}, hatchPattern(hatch)), s("rect", { class: "rv-mass-track", width, height: 12, rx: 3 }), ...rects),
      h("div", { class: "rv-mass-ends", "aria-hidden": "true" }, h("span", {}, "0"), h("span", {}, formatNumber(full, 4)))),
    h("ul", { class: "rv-mass-items" }, ...items));
}

/**
 * A chart over the positions of the analysis timeline (timeline.js): the size of the closed form as
 * a line, or the time per step as bars. `values[i]` is the value at position i (null: none; the line
 * breaks, no bar). The loop segments are shaded and labelled, the current position is marked with
 * a rule, an outlined point or bar and its value. A click picks the nearest position (`onPick`); the
 * keyboard moves through the timeline instead. Plain SVG with an accessible summary (`label`).
 */
export function positionChart({ values, segments = [], current = 0, kind = "line", scale = "linear",
  width: pixelWidth = 560, format = (v) => formatNumber(v, 3), label = "", segmentLabel = null, onPick = null } = {}) {
  const n = values.length;
  const width = Math.max(260, Math.round(pixelWidth));
  const height = 176;
  const top = 18;
  const bottom = 22;
  const left = 4;
  const right = 52;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const slot = plotW / Math.max(1, n);
  const cx = (i) => left + slot * (i + 0.5);
  const present = values.filter((v) => Number.isFinite(v));
  const axis = scale === "log" ? logAxis(present) : null;
  const max = axis ? 10 ** axis.hi : niceMax(Math.max(0, ...present));
  const fraction = axis
    ? (v) => (v > 0 ? Math.max(0, Math.min(1, (Math.log10(v) - axis.lo) / Math.max(1, axis.hi - axis.lo))) : 0)
    : (v) => (max > 0 ? v / max : 0);
  const y = (v) => top + plotH * (1 - fraction(v));

  const bands = segments.map((segment) => {
    const x0 = left + slot * segment.from;
    const w = slot * (segment.to - segment.from + 1);
    const text = segmentLabel ? segmentLabel(segment) : `loop ${segment.loop}`;
    return s("g", { class: "pc-band" }, s("rect", { x: x0, y: top - 14, width: Math.max(2, w), height: plotH + 14 }),
      w > 36 ? s("text", { x: x0 + 4, y: top - 4 }, text) : null);
  });
  const grid = axis
    ? axis.ticks.map((k) => s("g", { class: "mc-grid" },
      s("line", { x1: left, x2: left + plotW, y1: y(10 ** k), y2: y(10 ** k) }),
      s("text", { x: left + plotW + 6, y: y(10 ** k) + 4 }, powerOfTen(k))))
    : [0, max / 2, max].map((tick) => s("g", { class: "mc-grid" },
      s("line", { x1: left, x2: left + plotW, y1: y(tick), y2: y(tick) }),
      s("text", { x: left + plotW + 6, y: y(tick) + 4 }, format(tick))));
  if (axis) grid.push(s("text", { class: "mc-scale", x: left + plotW + 6, y: top + plotH + 4 }, "log"));

  const marks = [];
  if (kind === "line") {
    let run = [];
    const flush = () => {
      if (run.length > 1) marks.push(s("polyline", { class: "pc-line", points: run.join(" ") }));
      if (run.length === 1) marks.push(s("circle", { class: "pc-dot", cx: run[0].split(",")[0], cy: run[0].split(",")[1], r: 2 }));
      run = [];
    };
    values.forEach((v, i) => {
      if (Number.isFinite(v)) run.push(`${cx(i).toFixed(1)},${y(v).toFixed(1)}`);
      else flush();
    });
    flush();
  } else {
    const barW = Math.max(1, Math.min(28, slot * 0.7));
    values.forEach((v, i) => {
      if (!Number.isFinite(v) || i === current) return;
      const h0 = Math.max(v > 0 ? 1 : 0, plotH * fraction(v));
      marks.push(s("rect", { class: "pc-bar", x: cx(i) - barW / 2, y: top + plotH - h0, width: barW, height: h0 }));
    });
  }
  // the current position: a rule, the outlined mark and its value (or "no value")
  const v = values[current];
  const known = Number.isFinite(v);
  const xc = cx(current);
  const currentMarks = [s("line", { class: "pc-rule", x1: xc, x2: xc, y1: top - 14, y2: top + plotH })];
  if (known && kind === "line") currentMarks.push(s("circle", { class: "pc-current", cx: xc, cy: y(v), r: 4.5 }));
  if (known && kind === "bars") {
    const barW = Math.max(3, Math.min(28, slot * 0.7));
    const h0 = Math.max(v > 0 ? 2 : 0, plotH * fraction(v));
    currentMarks.push(s("rect", { class: "pc-current", x: xc - barW / 2, y: top + plotH - h0, width: barW, height: h0 }));
  }
  const text = known ? format(v) : "no value";
  const anchorEnd = xc > left + plotW * 0.7;
  const ty = known ? Math.max(top + 10, y(v) - 8) : top + plotH / 2;
  currentMarks.push(s("text", { class: "pc-value", x: xc + (anchorEnd ? -7 : 7), y: ty, "text-anchor": anchorEnd ? "end" : "start" }, text));

  const svg = s("svg", { class: `position-chart pc-${kind}`, viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
    "aria-label": label }, ...bands, ...grid, ...marks,
  s("line", { class: "mc-axis", x1: left, x2: left + plotW, y1: top + plotH, y2: top + plotH }), ...currentMarks);
  if (onPick) {
    svg.classList.add("pc-pickable");
    svg.addEventListener("click", (event) => {
      const rect = svg.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / Math.max(1, rect.width)) * width;
      onPick(Math.max(0, Math.min(n - 1, Math.floor((px - left) / slot))));
    });
  }
  return svg;
}
