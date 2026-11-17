// The output distributions of the two programs of a comparison (#154), plotted like those of an analysis: the same
// marginal charts with their intervals, the same joint view, the same switches. Both programs side by side (one
// above the other in a narrow pane), on one scale, with the values whose probabilities differ marked "≠". Uses the
// components of the analysis card (charts.js, joint-view.js, parts.js); like them it needs only the result JSON.
import { h } from "./dom.js";
import { marginalChart } from "./charts.js";
import { jointMass, marginalMass } from "./analysis-card.js";
import { chartLegend, formatValue } from "./format.js";
import { JOINT_VIEWS, jointView } from "./joint-view.js";
import { guaranteeBadge, loadScale, press, saveScale, segmented } from "./parts.js";
import { cellText, compareMarginals, comparedVariables } from "./equivalence-model.js";

const PAIR = "pair:";
const MARGINAL_VIEWS = [["chart", "Chart"], ["table", "Table"]];
const SIDES = [["first", "Program A"], ["second", "Program B"]];

const part = (exact, p) => (exact === null || exact === undefined) && (p === null || p === undefined) ? null : { exact, p };

/** What one program contributes: its result, guarantee, marginals and mass. */
function sideOf(key, label, result) {
  const ok = result?.status === "ok" && Boolean(result.distribution);
  const bound = result?.bounds?.kind || "unknown";
  return {
    key, label, result, ok, bound,
    marginals: ok ? result.distribution.marginals || {} : {},
    total: ok ? part(result.distribution.mass, result.distribution.mass_float) : null,
    residual: ok && bound === "lower" ? part(result.bounds?.residual_mass, result.bounds?.residual_mass_float) : null,
  };
}

/** Why a program has no chart of `name` (null if it has one). */
function missing(side, name) {
  if (!side.ok) return side.result?.status === "ok" ? "no output distribution" : "the analysis did not finish (see Result of below)";
  const table = side.marginals[name];
  if (!table) return `program ${side.label.at(-1)} has no variable ${name}`;
  if (table.error) return `not available: ${table.error}`;
  return null;
}

/** The marginal of `variable` in both programs as one table: value, A, B and whether they differ. */
function comparisonTable(first, second, variable, rows) {
  const cell = (row) => row ? h("td", { class: "num", title: row.prob ?? "" }, cellText(row.prob, formatValue(row.prob, row.p, 6)))
    : h("td", { class: "num rv-muted" }, "0");
  const body = rows.map((r) => h("tr", { "data-differs": r.differs ? "true" : null },
    h("td", { class: "num" }, String(r.value)), cell(r.first), cell(r.second),
    h("td", { class: "eq-diff" }, r.differs === true ? [h("span", { class: "glyph", "aria-hidden": "true" }, "≠"), " differs"]
      : r.differs === false ? h("span", { class: "rv-muted" }, "same") : "")));
  return h("div", { class: "table-wrap eq-table-wrap", tabindex: "0", role: "region", "aria-label": `Distribution of ${variable}` },
    h("table", { class: "rv-table eq-table", "aria-label": `Distribution of ${variable} in programs A and B` },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "k"), h("th", { scope: "col" }, "A"),
        h("th", { scope: "col" }, "B"), h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Comparison")))),
      h("tbody", {}, ...body)));
}

/**
 * The plots of both programs. `view` keeps the choices across re-renders ({subject, marginalMode, jointMode});
 * options: `loadJoint(payload, signal, onPartial)` for grids (the payload names the program: "first" or
 * "second"), `maxJointAxis`. Returns the element.
 */
export function comparePlots(result, other, view = {}, options = {}) {
  const sides = [sideOf("first", "Program A", result), sideOf("second", "Program B", other || {})];
  const variables = comparedVariables(result, other);
  const withPair = variables.length >= 2 && Boolean(options.loadJoint) && sides.every((s) => s.ok);
  if (!(view.subject === PAIR && withPair) && !variables.includes(view.subject)) view.subject = variables[0];
  if (view.marginalMode !== "table") view.marginalMode = "chart";
  if (!JOINT_VIEWS[view.jointMode]) view.jointMode = "heatmap";
  const isPair = () => view.subject === PAIR;
  const mode = () => (isPair() ? view.jointMode : view.marginalMode);

  const columns = sides.map((side) => {
    const status = h("p", { class: "rv-note eq-plot-status" });
    const body = h("div", { class: "eq-plot-body" });
    const mass = h("div", { class: "rv-mass-holder" });
    const element = h("section", { class: "eq-plot", "data-program": side.key, "aria-label": `${side.label}: output distribution` },
      h("div", { class: "eq-plot-head" }, h("h4", { class: "rv-block-title" }, side.label),
        side.ok ? guaranteeBadge(side.bound) : guaranteeBadge("error", "Failed")),
      status, body, mass);
    return { side, element, status, body, mass };
  });
  const grid = h("div", { class: "eq-plots" }, ...columns.map((c) => c.element));
  const wide = h("div", { class: "eq-plots-wide" });
  const legend = h("p", { class: "rv-note eq-plots-legend" });
  const viewHolder = h("span", { class: "rv-control" });
  let drawnWidth = 0;
  let joints = [];

  const scaleToggle = segmented("Scale", [["linear", "Linear"], ["log", "Log"]], (key) => { saveScale(key); render(); },
    (key) => (key === "log" ? "Logarithmic scale: small probabilities stay visible" : "Linear scale"));
  const pairLabel = variables.length === 2 ? `${variables[0]} and ${variables[1]}` : "Two variables";
  const subjects = segmented("Variable", [...variables.map((v) => [v, v]), ...(withPair ? [[PAIR, pairLabel]] : [])],
    (key) => { view.subject = key; render(); },
    (key) => (key === PAIR ? "Joint distribution of two variables in both programs" : `Distribution of ${key} in both programs`));

  /** The joint views: program A's with its controls; program B's follows A's pair and window. */
  const renderPair = () => {
    const load = (which) => (payload, signal, onPartial) => options.loadJoint({ ...payload, program: which }, signal, onPartial);
    const follower = (grid) => {
      const [, b] = columns;
      if (!grid) { b.body.replaceChildren(h("p", { class: "rv-muted" }, "Shown once program A's grid is there.")); return; }
      const bView = { x: grid.x, y: grid.y, xWindow: [grid.x_values[0], grid.x_values.at(-1)],
        yWindow: [grid.y_values[0], grid.y_values.at(-1)], windowChosen: true, mode: view.jointMode };
      const joint = jointView({ variables, load: load("second"), view: bView, maxAxis: options.maxJointAxis, scale: loadScale,
        onGrid: (g) => b.mass.replaceChildren(jointMass(g, b.side.total, b.side.residual)) });
      joints[1] = { joint, view: bView };
      b.body.replaceChildren(h("div", { class: "eq-joint-follower" }, joint.element));
    };
    const aView = view.joint || (view.joint = {});
    aView.mode = view.jointMode;
    const [a, b] = columns;
    const joint = jointView({ variables, load: load("first"), view: aView, maxAxis: options.maxJointAxis, scale: loadScale,
      onGrid: (g) => {
        a.mass.replaceChildren(jointMass(g, a.side.total, a.side.residual));
        follower(g);
      } });
    joints = [{ joint, view: aView }];
    a.body.replaceChildren(joint.element);
    a.status.textContent = "";
    b.status.textContent = "Same pair and window as program A.";
  };

  const render = () => {
    press(subjects, view.subject);
    press(scaleToggle, loadScale());
    const views = isPair() ? Object.entries(JOINT_VIEWS) : MARGINAL_VIEWS;
    if (viewHolder.dataset.kind !== (isPair() ? "pair" : "single")) {
      viewHolder.dataset.kind = isPair() ? "pair" : "single";
      viewHolder.replaceChildren(segmented(isPair() ? "Show the joint distributions as" : "Show the distributions as", views, (key) => {
        if (isPair()) {
          view.jointMode = key;
          for (const { view: v } of joints) v.mode = key;
        } else {
          view.marginalMode = key;
        }
        render();
      }));
    }
    press(viewHolder.firstChild, mode());
    scaleToggle.hidden = mode() === "table";
    if (isPair()) {
      wide.hidden = true;
      grid.hidden = false;
      legend.textContent = "The colours of each heatmap are scaled to its own largest probability.";
      if (!joints.length) renderPair();
      else for (const { joint } of joints) joint.draw();
      return;
    }
    joints = [];
    const name = view.subject;
    const rows = compareMarginals(result, other, name);
    const differing = new Set(rows.filter((r) => r.differs === true).map((r) => r.value));
    const bothExact = sides.every((s) => s.ok && s.bound === "exact");
    const notes = [];
    if (differing.size) {
      notes.push(`≠ marks the ${differing.size === 1 ? "value" : `${differing.size} values`} of ${name} whose probabilities differ`
        + (bothExact ? "." : ": their intervals do not overlap."));
    } else if (bothExact && rows.length) {
      notes.push(`The shown probabilities of ${name} agree.`);
    }
    if (!bothExact && !differing.size && sides.every((s) => s.ok)) {
      notes.push("With bounds, a value is marked only where the intervals of the two programs do not overlap.");
    }
    if (view.marginalMode === "table") {
      grid.hidden = true;
      wide.hidden = false;
      wide.replaceChildren(rows.length ? comparisonTable(result, other, name, rows) : h("p", { class: "rv-muted" }, "No values."));
      legend.textContent = notes.join(" ");
      return;
    }
    wide.hidden = true;
    grid.hidden = false;
    drawnWidth = columns[0].body.clientWidth || 0;
    const width = drawnWidth || Math.max(260, (grid.clientWidth || 560) / 2 - 12);
    // one scale for both charts: each gets the other's probabilities (and upper ends) as peers
    const probabilities = (side) => side.ok && side.marginals[name] && !side.marginals[name].error
      ? (side.marginals[name].rows || []).flatMap((r) => [r.p, r.upper_p]).filter((p) => Number.isFinite(p)) : [];
    columns.forEach((column, index) => {
      const { side } = column;
      const why = missing(side, name);
      column.status.textContent = "";
      if (why) {
        column.body.replaceChildren(h("p", { class: "rv-muted" }, `No chart: ${why}.`));
        column.mass.replaceChildren();
        return;
      }
      const table = side.marginals[name];
      column.body.replaceChildren(marginalChart(table, { residual: side.residual?.p, residualExact: side.residual?.exact, variable: name,
        bound: side.bound, width, scale: loadScale(), peers: probabilities(sides[1 - index]), differ: differing }));
      column.status.textContent = chartLegend(side.bound);
      column.mass.replaceChildren(marginalMass(table, name, side.total, side.residual));
    });
    legend.textContent = notes.join(" ");
  };

  if (typeof ResizeObserver !== "undefined") {
    // draw the charts at their real size so that labels keep their font size (the joint views have their own)
    new ResizeObserver(() => {
      const width = columns[0].body.clientWidth;
      if (!isPair() && view.marginalMode === "chart" && width && Math.abs(width - drawnWidth) > 16) render();
    }).observe(grid);
  }
  if (!variables.length) {
    return h("p", { class: "rv-muted" }, sides.some((s) => s.ok) ? "The programs have no variables to plot."
      : "No output distribution to plot: neither analysis finished; see the results below.");
  }
  render();
  return h("figure", { class: "rv-figure eq-figure" },
    h("div", { class: "rv-controls" }, variables.length > 1 || withPair ? subjects : null, viewHolder, scaleToggle),
    grid, wide, legend);
}
