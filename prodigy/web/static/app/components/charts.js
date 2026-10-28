// Charts of the result view: the interval bar of a query and the bar chart of a marginal.
// Plain SVG; colours come from CSS custom properties, so light/dark/high contrast follow.
import { h, s } from "./dom.js";
import { formatNumber, formatValue } from "./format.js";

let patternCounter = 0;

function hatchPattern(id) {
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

function niceMax(value) {
  if (!(value > 0)) return 1;
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * base) return step * base;
  }
  return 10 * base;
}

/**
 * Bar chart of a univariate coefficient table (see prodigy/analysis/result.py:
 * univariate_table). Extra bars: the mass beyond the shown values ("tail") and, for lower
 * bounds, the mass not yet accounted for (hatched).
 */
export function marginalChart(table, { residual = null, residualExact = null, variable = "x", bound = "exact", width: pixelWidth = 560 } = {}) {
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
  const max = niceMax(Math.max(...bars.map((b) => b.p)));
  const slot = plotW / bars.length;
  const barW = Math.max(2, Math.min(36, slot * 0.72));
  const id = `hatch-${++patternCounter}`;
  const labelEvery = Math.max(1, Math.ceil(bars.length / Math.max(1, Math.floor(plotW / 26))));
  const y = (p) => top + plotH * (1 - p / max);

  const grid = [0, max / 2, max].map((tick) => s("g", { class: "mc-grid" },
    s("line", { x1: left, x2: left + plotW, y1: y(tick), y2: y(tick) }),
    s("text", { x: left + plotW + 6, y: y(tick) + 4 }, formatNumber(tick, 2))));

  const marks = bars.map((bar, index) => {
    const cx = left + slot * index + slot / 2;
    const barHeight = Math.max(bar.p > 0 ? 1 : 0, plotH * (bar.p / max));
    const crowdedByExtra = extras.length > 0 && index === rows.length - 1 && slot < 26;
    const showLabel = bar.kind !== "value" || (index % labelEvery === 0 && !crowdedByExtra);
    const kindClass = bar.kind === "value" && (bound === "upper" || bound === "unknown") ? `mc-${bound}` : `mc-${bar.kind}`;
    const hatched = bar.kind === "residual" || (bar.kind === "value" && bound === "upper");
    return s("g", { class: `mc-bar ${kindClass}` },
      s("title", {}, bar.title),
      s("rect", { x: cx - barW / 2, y: top + plotH - barHeight, width: barW, height: barHeight, rx: Math.min(3, barW / 4),
        fill: hatched ? `url(#${id})` : null }),
      showLabel ? s("text", { x: cx, y: height - 6, class: "mc-label" }, bar.label) : null);
  });

  const summary = `Bar chart of the distribution of ${variable}: ${rows.length} values` +
    (extras.length ? `, plus ${extras.map((e) => e.kind === "tail" ? "the remaining tail" : "the mass not yet accounted for").join(" and ")}` : "") + ".";
  return s("svg", { class: "marginal-chart", viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": summary },
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
