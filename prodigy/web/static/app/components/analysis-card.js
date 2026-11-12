// The analysis card of the result view (issue #311): one card with switchable lenses
// (Distribution · Size · Time · Checks) and one timeline that all lenses share. Every lens has the
// same layout: the header says where the timeline stands, then the lens's controls, its figure, one
// fact line and its details as disclosures. The default position is the final result, so that a
// result without recorded steps looks like the distribution card it replaces; the answer card above
// always shows the final result, whatever the position.
//
// Like result-view.js this needs only the result JSON, the CSS custom properties of
// result-view.css and analysis-card.css and optionally KaTeX; the editor and the server are reached
// through callbacks (onPosition, loadJoint, onRecord, ...).
import { h } from "./dom.js";
import { marginalChart, marginalTable, massSummary, positionChart } from "./charts.js";
import { BOUND_TEXT, formatNumber, formatSeconds, formatValue, qualify } from "./format.js";
import { formulaBox } from "./formula-box.js";
import { JOINT_VIEWS, jointView } from "./joint-view.js";
import { chartState, reducedMotion, tweenChart } from "./motion.js";
import { copyButton, distributionFigure, expressionBox, guaranteeBadge, loadScale, press, saveScale, segmented } from "./parts.js";
import { breakpointTarget, marginalGap, positionData, positionLabel, positionOfStep, sizeSeries, timeSeries,
  timelineControl, timelinePositions } from "./timeline.js";

export const LENSES = [["distribution", "Distribution"], ["size", "Size"], ["time", "Time"], ["checks", "Checks"]];

const TRANSITION_MS = 320;
const part = (exact, p) => (exact === null || exact === undefined) && (p === null || p === undefined) ? null : { exact, p };
// variable names are identifiers, so this key never clashes with one
const PAIR = "pair:";
const MARGINAL_VIEWS = [["chart", "Chart"], ["table", "Table"]];

const BRANCH_TEXT = {
  then: "the then-branch: the part of the distribution where the guard holds",
  else: "the else-branch: the part of the distribution where the guard fails",
  left: "the left branch of the choice: the distribution weighted by its probability",
  right: "the right branch of the choice: the distribution weighted by its probability",
  block: "a query block",
};

/** Text for the guarantee of one distribution of the trace. */
export function snapshotGuarantee(snap) {
  switch (snap?.bound) {
    case "exact": return "Exact at this point.";
    case "lower": return `Lower bound: at most ${formatValue(snap.residual_mass, snap.residual_mass_float)} of the mass is not yet accounted for; it may still end up anywhere, or never terminate.`;
    case "upper": return "Upper bound (Park's lemma): exact only if the loops so far terminate almost surely, which is not checked.";
    default: return "No guarantee: approximations of different kinds were combined, or a side condition could not be checked.";
  }
}

function where(step) {
  if (!step) return "";
  if (step.kind === "normalize") return "end of the program";
  return step.line ? `line ${step.line}` : "generated";
}

function facts(...items) {
  return h("dl", { class: "ac-facts" }, ...items.filter(Boolean).map(([label, value, title]) =>
    h("div", {}, h("dt", {}, label), h("dd", { title: title || "" }, value))));
}

function disclosure(summary, ...body) {
  const content = body.flat().filter(Boolean);
  if (!content.length) return null;
  return h("details", { class: "disclosure" }, h("summary", {}, summary), h("div", { class: "disclosure-body" }, ...content));
}

// ---- distribution ---------------------------------------------------------------------------------

/** The mass line of a marginal: total mass, the values shown, the values beyond them, the "?" mass. */
function marginalMass(table, name, total, residual) {
  if (!table || table.error || !table.truncated || !table.rows?.length) return massSummary({ total, residual });
  const first = table.rows[0].value;
  const last = table.rows[table.rows.length - 1].value;
  const shownFloat = table.shown_mass_float ?? (table.mass_float ?? 0) - (table.tail_mass_float ?? 0);
  return massSummary({ total, residual,
    shown: part(table.shown_mass, shownFloat), shownLabel: `Shown: ${name} = ${first}–${last}`,
    beyond: part(table.tail_mass, table.tail_mass_float), beyondLabel: `Beyond: ${name} > ${last}` });
}

/** The mass line of a window of the joint view (null: no grid yet). */
function jointMass(grid, total, residual) {
  if (!grid) return massSummary({ total, residual });
  const range = (values) => `${values[0]}–${values[values.length - 1]}`;
  return massSummary({ total,
    residual: grid.residual_mass ? part(grid.residual_mass, grid.residual_mass_float) : residual,
    shown: part(grid.shown_mass, grid.shown_mass_float),
    shownLabel: `In the window: ${grid.x} = ${range(grid.x_values)}, ${grid.y} = ${range(grid.y_values)}`,
    beyond: part(grid.outside_mass, grid.outside_mass_float), beyondLabel: "Outside the window" });
}

/** At an iterate: the mass that has left the loop and the mass still inside it (hatched, "?"). */
function loopMass(snap, iterate) {
  const inside = part(iterate.inside_mass, iterate.inside_mass_float);
  const line = massSummary({ shown: part(snap.mass, snap.mass_float), shownLabel: "Has left the loop",
    residual: inside, residualLabel: "? Still inside the loop — not yet accounted for" });
  const earlier = (snap.residual_mass_float ?? 0) - (iterate.inside_mass_float ?? 0);
  if (earlier > 1e-12) {
    line.append(h("p", { class: "rv-note" }, `Besides, ${formatNumber(earlier, 4)} is not accounted for from earlier loops.`));
  }
  return line;
}

/**
 * What the Distribution lens draws at the current position: the result's distribution at the end,
 * else the snapshot of the trace. Marginals that were not recorded are listed with their reason.
 */
function distributionSource(ctx) {
  const { result, position, data } = ctx;
  if (position.kind === "final") {
    const dist = result.distribution;
    if (!dist) return null;
    const bound = result.bounds?.kind || "exact";
    return {
      final: true, bound, marginals: dist.marginals || {}, variables: Object.keys(dist.marginals || {}), snap: null,
      total: part(dist.mass, dist.mass_float),
      residual: bound === "lower" ? part(result.bounds?.residual_mass, result.bounds?.residual_mass_float) : null,
      joint: {}, jointBlocked: null,
    };
  }
  const snap = data.snapshot;
  if (!snap) return null;
  const bound = snap.bound || "unknown";
  const variables = [...new Set([...Object.keys(snap.marginals || {}), ...Object.keys(snap.marginals_not_recorded || {})])].sort();
  return {
    final: false, bound, marginals: snap.marginals || {}, variables, snap,
    total: part(snap.mass, snap.mass_float),
    residual: bound === "lower" ? part(snap.residual_mass, snap.residual_mass_float) : null,
    joint: position.kind === "iterate" ? { step: position.step, iterate: position.iterate } : { step: position.step },
    jointBlocked: snap.truncated ? "the closed form at this position is too large; the trace keeps only its beginning" : null,
  };
}

function gapFigure(name, reason) {
  return h("div", { class: "ac-gap", role: "img", "aria-label": `${name}: not recorded at this position (${reason}). Unknown, not 0.` },
    h("span", { class: "ac-gap-title" }, `${name}: not recorded`),
    h("span", { class: "ac-gap-reason" }, `${reason} — unknown at this position, not 0`));
}

function distributionLens(ctx) {
  const { result, view, options, position, data } = ctx;
  const source = distributionSource(ctx);
  const watches = watchesBlock(ctx);
  if (!source) {
    const text = position.kind === "final"
      ? (result.status === "ok" ? "The result has no output distribution." : "No output distribution: the analysis did not finish.")
      : "No distribution at this position: the instruction failed.";
    return { figure: h("p", { class: "rv-muted" }, text), details: [watches] };
  }
  const { variables, bound, total, residual } = source;
  const withPair = variables.length >= 2 && Boolean(options.loadJoint);
  if (!(view.subject === PAIR && withPair) && !variables.includes(view.subject)) view.subject = variables[0];
  if (view.marginalMode !== "table") view.marginalMode = "chart";
  const isPair = () => view.subject === PAIR;
  const mode = () => (isPair() ? view.mode || "heatmap" : view.marginalMode);
  const figure = h("div", { class: "rv-figure-body ac-figure" });
  const mass = h("div", { class: "rv-mass-holder" });
  const legend = h("p", { class: "rv-note", hidden: true });
  const viewHolder = h("span", { class: "rv-control" });
  let drawnWidth = 0;

  const render = () => {
    press(subjects, view.subject);
    press(scaleToggle, loadScale());
    const views = isPair() ? Object.entries(JOINT_VIEWS) : MARGINAL_VIEWS;
    if (viewHolder.dataset.kind !== (isPair() ? "pair" : "single")) {
      viewHolder.dataset.kind = isPair() ? "pair" : "single";
      viewHolder.replaceChildren(segmented(isPair() ? "Show the joint distribution as" : "Show the distribution as", views, (key) => {
        if (isPair()) view.mode = key; else view.marginalMode = key;
        render();
      }));
    }
    press(viewHolder.firstChild, mode());
    scaleToggle.hidden = mode() === "table";
    legend.hidden = true;
    if (isPair()) {
      if (source.jointBlocked) {
        figure.replaceChildren(h("p", { class: "rv-muted" }, `Joint distribution not available: ${source.jointBlocked}.`));
        mass.replaceChildren(massSummary({ total, residual }));
        return;
      }
      const joint = ctx.joint(variables, (grid) => mass.replaceChildren(jointMass(grid, total, residual)));
      if (figure.firstChild !== joint.element) figure.replaceChildren(joint.element);
      ctx.showJoint(joint);
      return;
    }
    const name = view.subject;
    const gap = source.final ? null : marginalGap(source.snap, name);
    const table = source.marginals[name];
    drawnWidth = figure.clientWidth || 560;
    if (gap && !table?.error) {
      figure.replaceChildren(gapFigure(name, gap));
    } else if (table.error) {
      figure.replaceChildren(h("p", { class: "rv-muted" }, `Not available: ${table.error}`));
    } else {
      figure.replaceChildren(mode() === "chart"
        ? marginalChart(table, { residual: residual?.p, residualExact: residual?.exact, variable: name, bound, width: drawnWidth, scale: loadScale() })
        : marginalTable(table, { variable: name, bound }));
      if (bound === "upper" && mode() === "chart") {
        legend.textContent = "Hatched bars are upper bounds.";
        legend.hidden = false;
      }
    }
    mass.replaceChildren(data.iterate ? loopMass(source.snap, data.iterate) : marginalMass(gap ? null : table, name, total, residual));
  };

  const pairLabel = variables.length === 2 ? `${variables[0]} and ${variables[1]}` : "Two variables";
  const subjects = segmented("Variable", [...variables.map((v) => [v, v]), ...(withPair ? [[PAIR, pairLabel]] : [])],
    (key) => { view.subject = key; render(); },
    (key) => (key === PAIR ? "Joint distribution of two variables" : `Distribution of ${key}`));
  const scaleToggle = segmented("Scale", [["linear", "Linear"], ["log", "Log"]], (key) => { saveScale(key); render(); },
    (key) => (key === "log" ? "Logarithmic scale: small probabilities stay visible" : "Linear scale"));
  if (typeof ResizeObserver !== "undefined") {
    // draw the chart at its real size so that labels keep their font size (the joint view has its own)
    new ResizeObserver(() => {
      if (!isPair() && mode() === "chart" && figure.clientWidth && Math.abs(figure.clientWidth - drawnWidth) > 16) render();
    }).observe(figure);
  }
  if (variables.length) render();
  else {
    figure.replaceChildren(h("p", { class: "rv-muted" }, source.snap?.marginals_omitted
      ? `Marginals not recorded at this position: ${source.snap.marginals_omitted}. Unknown, not 0.`
      : "The distribution has no variables."));
    mass.replaceChildren(data.iterate ? loopMass(source.snap, data.iterate) : massSummary({ total, residual }));
  }
  const controls = variables.length
    ? h("div", { class: "rv-controls" }, variables.length > 1 || withPair ? subjects : null, viewHolder, scaleToggle)
    : null;

  const lead = [];
  if (!source.final) {
    lead.push(h("div", { class: "ac-guarantee" }, guaranteeBadge(source.bound), h("span", {}, snapshotGuarantee(source.snap))));
  } else if (result.steps?.length) {
    lead.push(h("div", { class: "ac-guarantee" }, guaranteeBadge(source.bound), h("span", {}, "The output distribution, as in the answer card.")));
  }
  return {
    lead,
    controls,
    figure: h("figure", { class: "rv-figure" }, figure, legend),
    facts: mass,
    details: source.final
      ? [generatingFunction(result.distribution.latex, result.distribution.closed_form, false), distValuesBlock(result), finalFacts(result.distribution), watches]
      : [generatingFunction(source.snap.latex, source.snap.closed_form, source.snap.truncated, source.snap.size), snapshotDists(source.snap),
        snapshotFacts(source.snap), watches],
    open: (key) => (key === "watches" ? Boolean(view.watchesOpen) : false),
  };
}

function finalFacts(dist) {
  const items = [["Support", dist.finite_support ? "finite" : "infinite"]];
  if (dist.error_probability && dist.error_probability !== "0") {
    items.push(["Observation violated", formatValue(dist.error_probability, dist.error_probability_float, 6), dist.error_probability]);
  }
  if (dist.parameters?.length) items.push(["Parameters", dist.parameters.join(", ")]);
  return disclosure("Facts", facts(...items));
}

function snapshotFacts(snap) {
  const items = [["Mass", formatValue(snap.mass, snap.mass_float, 6), snap.mass],
    ["Size", `${snap.size.toLocaleString("en-US")} character${snap.size === 1 ? "" : "s"}`, "Length of the closed form; the same measure on every engine"]];
  if (snap.error_probability && snap.error_probability !== "0") {
    items.push(["Observation violated", formatValue(snap.error_probability, snap.error_probability_float, 6), snap.error_probability]);
  }
  return disclosure("Facts", facts(...items));
}

function generatingFunction(latex, text, truncated, size = null) {
  const shown = truncated ? `${text} …` : text;
  return h("details", { class: "disclosure rv-gf" }, h("summary", {}, "Generating function"),
    h("div", { class: "disclosure-body" },
      formulaBox({ latex: truncated ? null : latex, text: shown, label: "Generating function" }),
      truncated ? h("p", { class: "rv-note" }, `Only the first characters of ${size?.toLocaleString("en-US") ?? "many"} are recorded.`) : null,
      h("div", { class: "rv-actions" }, latex && !truncated ? copyButton("LaTeX", latex) : null, text ? copyButton("Text", text) : null)));
}

// the dist variables are not part of the distribution; their values are listed apart (#217: with KaTeX)
function snapshotDists(snap) {
  const dists = snap.dists || [];
  if (!dists.length) return null;
  const several = new Set(dists.map((d) => d.variable)).size < dists.length;
  return disclosure(`Values of dist variables (${dists.length})`, ...dists.map((d) => h("div", { class: "rv-dist-card" },
    h("div", { class: "rv-card-head" }, h("span", { class: "mono rv-dist-infer" }, `${d.variable} = ${d.infer ?? "no value"}`),
      several ? h("span", { class: "rv-meta", title: d.probability }, `probability ${formatValue(d.probability, Number(d.probability) || null, 6)}`) : null),
    d.closed_form ? formulaBox({ latex: d.latex, text: d.closed_form, label: `Generating function of ${d.variable}` }) : null)));
}

function distValueCard(variable, value, entry, result) {
  const head = h("div", { class: "rv-card-head" },
    h("span", { class: "mono rv-dist-infer" }, `${variable} = ${value.infer ?? "no value"}`),
    h("span", { class: "rv-actions" },
      value.closed ? guaranteeBadge("exact") : null,
      value.closed_form ? copyButton("Text", value.closed_form) : null));
  const children = [head];
  if (value.closed_form) {
    children.push(formulaBox({ latex: value.latex, text: value.closed_form, label: `Generating function of ${variable}` }));
  }
  const items = [];
  if (entry.depends_on_state) {
    const kind = result.bounds?.kind;
    const qualifier = kind === "lower" ? "at least " : kind === "upper" ? "at most " : "";
    items.push(["Probability of this value", qualifier + formatValue(value.probability, value.probability_float, 6), value.probability]);
  }
  if (value.mean !== null && value.mean !== undefined) items.push(["Expected value", formatValue(value.mean, value.mean_float), value.mean]);
  if (value.normalizing_constant) items.push(["Normalising constant Z", formatValue(value.normalizing_constant, value.normalizing_constant_float, 6), value.normalizing_constant]);
  if (value.finite_support !== null && value.finite_support !== undefined) items.push(["Support", value.finite_support ? "finite" : "infinite"]);
  if (items.length) children.push(facts(...items));
  if (value.note) children.push(h("p", { class: "rv-note" }, value.note));
  if (value.table && !value.table.error && value.table.rows?.length) {
    // the closed form is exact, whatever the guarantee of the program state
    children.push(distributionFigure(value.table, { bounds: { kind: "exact" } }, variable, "table"));
  }
  return h("div", { class: "rv-dist-card" }, ...children);
}

/** The values of the `dist` variables at the end of the program. */
export function distValuesBlock(result) {
  const entries = result.distributions || [];
  if (!entries.length) return null;
  const cards = [];
  for (const entry of entries) {
    for (const value of entry.values) cards.push(distValueCard(entry.variable, value, entry, result));
  }
  return h("details", { class: "disclosure rv-inferred", open: true }, h("summary", {}, "Inferred distributions"),
    h("div", { class: "disclosure-body" },
      h("p", { class: "rv-note" }, "Values of the dist variables at the end of the program, apart from the output distribution. Each is the normalised distribution of its infer(…), computed exactly."),
      ...cards));
}

// ---- watches: queries after every top-level step --------------------------------------------------

function watchesBlock(ctx) {
  const { result, options, position, data, view } = ctx;
  const watches = result.trace?.watches || [];
  if (!options.onAddWatch && !watches.length) return null;
  const steps = result.steps || [];
  // at the final result: the values after the last top-level step
  const step = position.kind === "final" ? [...steps].reverse().find((s) => s.depth === 0) : data.step;
  const field = h("input", { type: "text", class: "field mono", placeholder: result.syntax === "ppy" ? "Pr[x == 3] or Ex[x]" : "Pr[x = 3] or Ex[x]",
    "aria-label": "Query to evaluate after every step", autocomplete: "off", spellcheck: "false" });
  const form = h("form", { class: "field-with-action ac-watch-form" }, field,
    h("button", { type: "submit", class: "button small" }, steps.length ? "Add and analyze again" : "Add, record steps and analyze"));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = field.value.trim();
    if (text) options.onAddWatch?.(text);
  });
  const rows = watches.map((watch, i) => {
    const value = step?.watches?.[i];
    let shown;
    if (position.kind === "iterate" || (step && step.depth > 0)) shown = h("span", { class: "rv-muted" }, "evaluated after top-level steps only");
    else if (!value) shown = h("span", { class: "rv-muted" }, "—");
    else {
      const { glyph, text } = qualify(value, { withValue: true });
      shown = h("span", { class: "rv-q-bound", "data-kind": value.bound },
        glyph ? h("span", { class: "glyph", "aria-hidden": "true" }, glyph) : null, text,
        value.note ? h("span", { class: "rv-note" }, ` ${value.note}`) : null);
    }
    return h("li", { class: "ac-watch" }, h("code", {}, watch.text), shown,
      options.onRemoveWatch ? h("button", { type: "button", class: "link-button", "aria-label": `Remove ${watch.text} and analyze again`,
        onclick: () => options.onRemoveWatch(watch.text) }, "Remove") : null);
  });
  const block = h("details", { class: "disclosure ac-watches", open: Boolean(view.watchesOpen) || null },
    h("summary", {}, watches.length ? `Queries after every step (${watches.length})` : "Queries after every step"),
    h("div", { class: "disclosure-body" },
      rows.length ? h("ul", { class: "ac-watch-list" }, ...rows)
        : h("p", { class: "rv-note" }, "Add a query such as Pr[x = 3]; it is evaluated after every top-level step, as if it stood right after that instruction, with the same sound intervals as the program's queries."),
      options.onAddWatch ? form : null));
  block.addEventListener("toggle", () => { view.watchesOpen = block.open; });
  return block;
}

// ---- size and time -------------------------------------------------------------------------------

function stepWhere(result, model, index) {
  const position = model.positions[index];
  if (position.kind === "final") return "the final result";
  const step = result.steps[position.step];
  const text = step.summary ? ` (${step.summary.length > 32 ? `${step.summary.slice(0, 31)}…` : step.summary})` : "";
  const loop = position.kind === "iterate" ? `, iteration ${position.iteration}` : "";
  return `step ${position.step + 1}${loop}, ${where(step)}${text}`;
}

function largest(values) {
  let best = -1;
  values.forEach((v, i) => { if (Number.isFinite(v) && (best < 0 || v > values[best])) best = i; });
  return best;
}

function noSteps(ctx, what) {
  return h("div", { class: "ac-empty" },
    h("p", {}, `${what} needs the recorded steps of the analysis; this result has none.`),
    ctx.options.onRecord
      ? h("p", {}, h("button", { type: "button", class: "button", onclick: () => ctx.options.onRecord() }, "Record Steps and Analyze"))
      : null);
}

function positionsTable(ctx, sizes, times) {
  const { result, model, index, go } = ctx;
  const rows = model.positions.map((position, i) => {
    const data = positionData(result, position);
    const kind = position.kind === "final" ? result.bounds?.kind || "unknown" : data.failed ? "error" : data.snapshot?.bound || "unknown";
    const text = BOUND_TEXT[kind] || { glyph: "×", label: "Failed" };
    const name = position.kind === "final" ? "Final result"
      : position.kind === "iterate" ? `Iteration ${position.iteration}` : (data.step.summary || data.step.instruction || "");
    return h("tr", { "aria-current": i === index ? "true" : null },
      h("td", { class: "num" }, h("button", { type: "button", class: "link-button", onclick: () => go(i),
        "aria-label": `Show position ${i + 1}` }, String(i + 1))),
      h("td", { class: "num" }, data.step?.line ? String(data.step.line) : "–"),
      h("td", { class: "ac-cell-text mono", style: data.step?.depth ? `padding-left:${8 + data.step.depth * 12}px` : null }, name),
      h("td", { "data-kind": kind, class: "ac-cell-kind" }, h("span", { "aria-hidden": "true" }, text.glyph), h("span", { class: "visually-hidden" }, text.label)),
      h("td", { class: "num" }, Number.isFinite(sizes[i]) ? sizes[i].toLocaleString("en-US") : "—"),
      h("td", { class: "num" }, Number.isFinite(times[i]) ? formatSeconds(times[i]) : "—"));
  });
  return disclosure("Table of all positions", h("div", { class: "table-wrap ac-positions" }, h("table", { class: "rv-table" },
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "#"), h("th", { scope: "col" }, "Line"), h("th", { scope: "col" }, "Instruction"),
      h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Guarantee")), h("th", { scope: "col" }, "Size"), h("th", { scope: "col" }, "Time"))),
    h("tbody", {}, ...rows))));
}

function hintNote(ctx, kind) {
  const hint = (ctx.result.trace?.hints || []).find((item) => item.kind === kind);
  if (!hint) return null;
  const target = hint.step !== null && hint.step !== undefined ? positionOfStep(ctx.model, hint.step) : null;
  return h("p", { class: "rv-note" }, hint.text, " ",
    target !== null ? h("button", { type: "button", class: "link-button", onclick: () => ctx.go(target) }, "Show this step") : null);
}

function chartWidth(ctx) {
  return Math.max(300, (ctx.body.clientWidth || ctx.element.clientWidth || 600) - 4);
}

function sizeLens(ctx) {
  const { result, model, index, view } = ctx;
  if (!result.steps?.length) return { figure: noSteps(ctx, "The size of the closed form per step") };
  const sizes = sizeSeries(result, model);
  const scale = view.sizeScale === "log" ? "log" : "linear";
  const unit = (v) => `${Math.round(v).toLocaleString("en-US")} characters`;
  const top = largest(sizes);
  const finalSize = sizes[model.final];
  const chart = positionChart({ values: sizes, segments: model.segments, current: index, kind: "line", scale, width: chartWidth(ctx),
    format: (v) => formatNumber(v, 3), onPick: ctx.go,
    label: `Size of the closed form per position, ${scale === "log" ? "logarithmic" : "linear"} scale. `
      + `${top >= 0 ? `Largest ${unit(sizes[top])} at ${stepWhere(result, model, top)}. ` : ""}`
      + `At this position ${Number.isFinite(sizes[index]) ? unit(sizes[index]) : "unknown"}. The table of all positions lists every value.` });
  const controls = h("div", { class: "rv-controls" },
    segmented("Scale", [["linear", "Linear"], ["log", "Log"]], (key) => { view.sizeScale = key; ctx.redraw(); },
      (key) => (key === "log" ? "Logarithmic scale" : "Linear scale")));
  press(controls.firstChild, scale);
  return {
    controls,
    figure: h("figure", { class: "rv-figure" }, chart,
      h("figcaption", { class: "rv-note" }, "Length of the closed form in characters, the same measure on every engine.")),
    facts: facts(["At this position", Number.isFinite(sizes[index]) ? unit(sizes[index]) : "unknown"],
      top >= 0 ? ["Largest", `${unit(sizes[top])} at ${stepWhere(result, model, top)}`] : null,
      ["Final result", Number.isFinite(finalSize) ? unit(finalSize) : "unknown"]),
    details: [hintNote(ctx, "growth"), positionsTable(ctx, sizes, timeSeries(result, model))],
  };
}

function timeLens(ctx) {
  const { result, model, index, position, data } = ctx;
  if (!result.steps?.length) return { figure: noSteps(ctx, "The time per step") };
  const times = timeSeries(result, model);
  const slow = largest(times);
  const loopTime = (segment) => result.steps[segment.step]?.time;
  const chart = positionChart({ values: times, segments: model.segments, current: index, kind: "bars", width: chartWidth(ctx),
    format: (v) => formatSeconds(v), onPick: ctx.go,
    segmentLabel: (segment) => `loop ${segment.loop} · ${formatSeconds(loopTime(segment))}`,
    label: `Time per step, without nested steps. ${slow >= 0 ? `Slowest ${formatSeconds(times[slow])} at ${stepWhere(result, model, slow)}. ` : ""}`
      + "Iterates of loops are not timed one by one. The table of all positions lists every value." });
  let here;
  if (position.kind === "iterate") here = ["At this position", `iterates are not timed one by one; the loop took ${formatSeconds(data.step.time)}`];
  else if (position.kind === "final") here = ["At this position", "the final result"];
  else here = ["At this position", formatSeconds(times[index]) || "—",
    data.step.time !== data.step.self_time ? `${formatSeconds(data.step.time)} with nested steps` : ""];
  const snapshot = result.trace?.snapshot_time;
  return {
    figure: h("figure", { class: "rv-figure" }, chart,
      h("figcaption", { class: "rv-note" }, "Time of each step without its nested steps, so the bars add up. A loop is one step; its iterates have no bars of their own.")),
    facts: facts(here,
      slow >= 0 ? ["Slowest", `${formatSeconds(times[slow])} at ${stepWhere(result, model, slow)}`] : null,
      result.timing?.analysis !== undefined ? ["Total analysis", formatSeconds(result.timing.analysis)] : null,
      Number.isFinite(snapshot) ? ["Recording the steps", `${formatSeconds(snapshot)}, not counted in the steps`] : null),
    details: [hintNote(ctx, "time"), positionsTable(ctx, sizeSeries(result, model), times)],
  };
}

// ---- checks: loops and certificates ------------------------------------------------------------------

const CHECK_GLYPHS = { passed: "✓", failed: "×", unknown: "?", info: "i" };
const CHECK_LABELS = { passed: "Proven", failed: "Failed", unknown: "Not established", info: "Note" };

/** Verdicts of an invariant check: "refuted" is proven wrong, "unknown" is neither proven nor refuted. */
const VERDICTS = {
  verified: "verified",
  refuted: "refuted — the invariant is provably wrong",
  unknown: "not proven — neither proven nor refuted; the invariant may still be correct",
};

export function verdictText(verdict) {
  return VERDICTS[verdict] || verdict;
}

function checksList(checks) {
  if (!checks?.length) return null;
  return h("ul", { class: "ac-checks", "aria-label": "Checks of the certificate" }, ...checks.map((check) =>
    h("li", { class: "ac-check", "data-status": check.status },
      h("span", { class: "ac-check-glyph", "aria-hidden": "true" }, CHECK_GLYPHS[check.status] || "i"),
      h("span", { class: "ac-check-body" },
        h("span", { class: "ac-check-label" }, `${check.label} · ${CHECK_LABELS[check.status] || check.status}`),
        h("span", { class: "ac-check-text" }, check.text)))));
}

function yesNo(value, yes = "yes", no = "no") {
  if (value === true) return yes;
  if (value === false) return no;
  return "undecided";
}

/** How Prodigy's check ended for one answer of a language model: badge kind, glyph, label. */
const AI_VERDICTS = {
  proven: ["lower", "✓", "Proven"],
  not_larger: ["unknown", "=", "Proven, not larger"],
  refuted: ["error", "×", "Refuted"],
  unknown: ["unknown", "?", "Not proven"],
  rejected: ["error", "×", "Rejected"],
  no_candidate: ["error", "×", "No candidate"],
  error: ["error", "!", "Request failed"],
};

function aiVerdictBadge(attempt) {
  const verdict = attempt.verdict || (attempt.candidate ? "rejected" : "no_candidate");
  const [kind, glyph, label] = AI_VERDICTS[verdict] || AI_VERDICTS.rejected;
  return h("span", { class: "rv-badge", "data-kind": kind }, h("span", { class: "glyph", "aria-hidden": "true" }, glyph), label);
}

/** The attempts of a language model (strategy ai_invariant): candidate, Prodigy's verdict in words, full answer. */
function aiAttempts(attempts) {
  return attempts.map((a) => h("div", { class: "rv-cert-block" },
    h("dt", { class: "rv-ai-head" }, `Attempt ${a.attempt}`, " ", aiVerdictBadge(a)),
    h("dd", {},
      h("div", {}, a.candidate ? expressionBox(a.candidate, "Candidate") : (a.verdict === "error" ? null : "No candidate in the answer"),
        a.stop_reason && !["end_turn", "stop"].includes(a.stop_reason) ? ` · stopped: ${a.stop_reason}` : ""),
      a.mass && a.verdict === "proven" ? h("div", {}, `Lower bound with mass ${a.mass}; mass ${a.missing_mass} not accounted for.`) : null,
      a.reason ? h("div", {}, a.reason) : null,
      a.details ? h("details", {}, h("summary", {}, "Details of the check"), h("pre", { class: "rv-pre" }, JSON.stringify(a.details, null, 2))) : null,
      a.verdict === "error" ? null
        : h("details", {}, h("summary", {}, "Answer of the model"), h("pre", { class: "rv-pre" }, a.answer || "(no text)")))));
}

function certificateView(cert) {
  const rows = [];
  const add = (label, value) => { if (value !== null && value !== undefined && value !== "") rows.push(h("div", {}, h("dt", {}, label), h("dd", {}, value))); };
  if (cert.ai) {
    const others = cert.ai.history || cert.ai.rejected || [];
    const asked = cert.ai.asked ?? others.length + 1;
    add("Candidate from", `${cert.ai.model} (attempt ${cert.ai.attempts} of ${asked}), checked by Prodigy`);
    if (others.length) rows.push(...aiAttempts(others));
  }
  switch (cert.type) {
    case "unrolling":
      add("Iterations", cert.fixpoint ? `${cert.iterations_done} (fixed point reached)` : String(cert.iterations));
      add("Mass still in the loop", h("code", {}, cert.remaining_mass));
      break;
    case "mass_threshold":
      add("Threshold", h("code", {}, cert.threshold));
      add("Iterations", String(cert.iterations));
      add("Mass still in the loop", h("code", {}, cert.remaining_mass));
      break;
    case "evt_iteration":
      add("Iterations of Ψ", String(cert.iterations));
      add("Mass not accounted for", h("code", {}, cert.remaining_mass));
      break;
    case "invariant":
      add("Rule", "Park's lemma: Φ(I) = I for all inputs");
      add("Verdict", verdictText(cert.verdict));
      if (cert.constraints?.length) add("Constraints", h("code", {}, JSON.stringify(cert.constraints)));
      if (cert.counterexample) add("Counterexample", h("code", {}, JSON.stringify(cert.counterexample)));
      rows.push(h("div", { class: "rv-cert-block" }, h("dt", {}, "Invariant I"), h("dd", {}, expressionBox((cert.invariant || "").trim(), "Invariant I"))));
      break;
    case "evt_invariant":
    case "evt_synthesis":
      add("Verdict", cert.verdict ? verdictText(cert.verdict) : null);
      add("Equation", cert.equation ? h("code", {}, cert.equation) : null);
      add("Invariant I", cert.invariant ? expressionBox(cert.invariant, "Invariant I") : null);
      add("Φ(I) − I", cert.difference ? expressionBox(cert.difference, "Φ(I) − I") : null);
      if (cert.verdict === "refuted" || cert.verdict === "unknown") break;
      add("Non-negative coefficients", yesNo(cert.nonnegative, "yes", "no"));
      add("Finite mass", cert.finite_mass === true ? `yes, |I| = ${cert.mass}` : yesNo(cert.finite_mass, "yes", "no — may be a spurious fixed point"));
      if (cert.nonnegative === true && cert.finite_mass !== true) {
        add("Holds for the series", cert.series_identity ? `yes (${cert.series_identity})` : "not proven — no guarantee");
      }
      if (cert.candidates?.length > 1) {
        rows.push(h("div", { class: "rv-cert-block" }, h("dt", {}, "Candidates"), h("dd", {},
          h("div", { class: "table-wrap" }, h("table", { class: "rv-table" },
            h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Invariant"), h("th", {}, "≥ 0"), h("th", {}, "Finite mass"))),
            h("tbody", {}, ...cert.candidates.map((c) => h("tr", { class: c.index === cert.chosen ? "chosen" : null },
              h("td", { class: "num" }, String(c.index)), h("td", {}, expressionBox(c.invariant, `Candidate ${c.index}`)),
              h("td", {}, yesNo(c.nonnegative)), h("td", {}, yesNo(c.finite_mass))))))))));
      }
      break;
    case "evt_subinvariant":
      add("Rule", "Lower induction: ν ≤ Φ(ν) and |ν| < ∞, so [¬guard]·ν lies below the output");
      add("Sub-invariant ν", cert.subinvariant ? expressionBox(cert.subinvariant, "Sub-invariant ν") : null);
      add("|ν|", h("code", {}, cert.mass));
      add("ν ≥ 0", cert.nonnegative);
      add("ν ≤ Φ(ν)", cert.subinvariance);
      add("Mass not accounted for", h("code", {}, cert.missing_mass));
      break;
    case "evt_sandwich": {
      add("Rule", "Finite-mass sub-invariant below (lower induction), super-invariant above (Park); both checked exactly");
      const [low, high] = cert.termination_probability || [];
      add("Termination probability", h("code", {}, `[${low}, ${high === "inf" ? "∞" : high}]`));
      for (const [label, c] of [["Sub-invariant", cert.lower], ["Super-invariant", cert.upper]]) {
        if (!c) continue;
        add(label, `${c.template_size} template values, rates ${c.rates.join(", ")}, periods ${c.periods.join(", ")}`);
        add(`${label}: exit mass`, h("code", {}, `${c.kind === "upper" ? "≤" : "≥"} ${c.exit_mass_rounded ?? c.exit_mass}`));
        if (c.check) add(`${label}: check`, c.check);
        rows.push(h("div", { class: "rv-cert-block" }, h("dt", {}, `${label} H`),
          h("dd", {}, expressionBox(c.generating_function || "", `${label} H`))));
      }
      break;
    }
    case "ai_attempts":
      add("Model", cert.model);
      rows.push(...aiAttempts(cert.attempts || []));
      break;
    default:
      add("Details", expressionBox(JSON.stringify(cert), "Details"));
  }
  return h("dl", { class: "rv-cert" }, ...rows);
}

/** Badge of a failed loop: "Not proven" if its last certificate is an invariant that was neither proven nor refuted. */
function failedBadge(loop) {
  const certs = loop.certificates || [];
  const verdict = certs.length ? certs[certs.length - 1].verdict : undefined;
  if (verdict === "unknown") {
    return h("span", { class: "rv-badge", "data-kind": "unknown" }, h("span", { class: "glyph", "aria-hidden": "true" }, "?"), "Not proven");
  }
  const label = verdict === "refuted" ? "Refuted" : "Failed";
  return h("span", { class: "rv-badge", "data-kind": "error" }, h("span", { class: "glyph", "aria-hidden": "true" }, "×"), label);
}

/** A loop's verdict in words, with its glyph (never by colour alone). */
function loopVerdict(loop) {
  const kind = loop.kind || "unknown";
  const glyph = (BOUND_TEXT[kind] || BOUND_TEXT.unknown).glyph;
  const words = { exact: "exact", lower: "lower bound", upper: "upper bound" }[kind] || "no guarantee";
  const rest = kind === "lower" && loop.residual_mass && loop.residual_mass !== "0"
    ? `, at most ${formatValue(loop.residual_mass, loop.residual_mass_float ?? null)} not accounted for` : "";
  return h("span", { class: "rv-loop-verdict", "data-kind": kind },
    h("span", { class: "glyph", "aria-hidden": "true" }, glyph), ` ${words}${rest}`);
}

/** One loop of the result: verdict, strategy, messages, the checks of the trace and the certificates. */
function loopRow(loop, ctx, traceLoop = null, jump = null) {
  const { options } = ctx;
  const titles = options.strategyTitles || {};
  const line = loop.line ? `line ${loop.line}` : "";
  const name = loop.id !== null && loop.id !== undefined ? `Loop ${loop.id}` : "Loop";
  const kind = loop.failed ? "error" : loop.kind;
  const header = h("div", { class: "rv-loop-head" },
    h("div", { class: "rv-loop-title" }, h("span", { class: "rv-loop-name" }, name), " ",
      line && options.onSelectLine ? h("button", { type: "button", class: "link-button rv-loop-line", onclick: () => options.onSelectLine(loop.line) }, line)
        : h("span", { class: "rv-loop-line" }, line)),
    loop.failed ? failedBadge(loop) : (loop.time ? h("span", { class: "rv-meta" }, formatSeconds(loop.time)) : null));
  const strategy = h("div", { class: "rv-loop-strategy" }, titles[loop.strategy] || loop.strategy,
    loop.failed ? null : [" · ", loopVerdict(loop)],
    loop.invocations > 1 ? ` · entered ${loop.invocations}×` : "",
    loop.inside_checks ? ` · ${loop.inside_checks === loop.invocations ? "only" : `${loop.inside_checks}×`} inside another loop's check` : "",
    loop.failed && loop.time ? ` · ${formatSeconds(loop.time)}` : "");
  const messages = (loop.messages || []).map((m) => h("p", { class: "rv-loop-message" }, m));
  const certs = loop.certificates || [];
  const details = certs.length ? h("details", { class: "disclosure" },
    h("summary", {}, certs.length > 1 ? `Certificates (${certs.length})` : "Certificate"),
    h("div", { class: "disclosure-body" }, ...certs.slice(0, 5).map(certificateView))) : null;
  return h("li", { class: "rv-loop", "data-kind": kind }, header, strategy,
    loop.failed ? h("p", { class: "rv-loop-message rv-error-text" }, loop.failed) : null, ...messages,
    checksList(traceLoop?.checks), jump, details);
}

/** The loop of the trace at a step (strategy, params, checks, iterates), for a loop the result does not list. */
function traceLoopRow(step, ctx) {
  const loop = step.loop;
  const titles = ctx.options.strategyTitles || {};
  const params = Object.entries(loop.params || {}).filter(([, v]) => v !== null && v !== "")
    .map(([k, v]) => `${k} = ${String(v).length > 40 ? `${String(v).slice(0, 39)}…` : v}`).join(", ");
  const badge = loop.failed
    ? h("span", { class: "rv-badge", "data-kind": "error" }, h("span", { class: "glyph", "aria-hidden": "true" }, "×"), "Failed")
    : guaranteeBadge(loop.kind || "unknown");
  return h("li", { class: "rv-loop", "data-kind": loop.failed ? "error" : loop.kind },
    h("div", { class: "rv-loop-head" }, h("span", { class: "rv-loop-name" }, loop.id !== null && loop.id !== undefined ? `Loop ${loop.id}` : "Loop"), badge),
    h("div", { class: "rv-loop-strategy" }, titles[loop.strategy] || loop.title || loop.strategy, params ? ` · ${params}` : ""),
    loop.failed ? h("p", { class: "rv-loop-message rv-error-text" }, loop.failed) : null,
    ...(loop.messages || []).map((m) => h("p", { class: "rv-loop-message" }, m)),
    checksList(loop.checks));
}

function checkCounts(checks) {
  const counts = {};
  for (const check of checks) counts[check.status] = (counts[check.status] || 0) + 1;
  const words = { passed: "proven", failed: "failed", unknown: "not established", info: "notes" };
  return ["passed", "failed", "unknown", "info"].filter((k) => counts[k])
    .map((k) => `${CHECK_GLYPHS[k]} ${counts[k]} ${k === "info" && counts[k] === 1 ? "note" : words[k]}`).join(" · ") || "none recorded";
}

function checksLens(ctx) {
  const { result, model, position, data, go } = ctx;
  const loops = result.loops || [];
  const steps = result.steps || [];
  const loopStep = data.step?.loop ? data.step : null;
  // the trace's loop of a result loop: its last step (a loop may be entered more than once)
  const traceOf = (id) => [...steps].reverse().find((s) => s.loop && s.loop.id === id)?.loop || null;
  if (loopStep) {
    const loop = loopStep.loop;
    const record = loops.find((l) => l.id === loop.id);
    const iterate = data.iterate;
    const progress = iterate
      ? h("p", { class: "rv-note" }, `Iteration ${iterate.iteration} of ${loop.iterations}: ${formatValue(iterate.distribution.mass, iterate.distribution.mass_float)} has left the loop, `
        + `${formatValue(iterate.inside_mass, iterate.inside_mass_float)} is still inside.`)
      : null;
    const row = record ? loopRow(record, ctx, loop) : traceLoopRow(loopStep, ctx);
    return {
      figure: h("div", {}, h("ol", { class: "rv-loops", "aria-label": "Loop at this position" }, row), progress),
      facts: facts(["Checks of this loop", checkCounts(loop.checks || [])],
        loop.iterations ? ["Iterations", `${loop.iterations}${loop.stride > 1 ? `, every ${loop.stride}th recorded` : ""}`] : null),
    };
  }
  if (position.kind === "final") {
    if (!loops.length) {
      return { figure: h("p", { class: "rv-note" }, "The program has no loops; every instruction is transformed exactly, so there is nothing to check.") };
    }
    const segmentOf = (id) => model.segments.find((segment) => segment.loop === id);
    const rows = loops.map((loop) => {
      const segment = segmentOf(loop.id);
      const jump = segment ? h("button", { type: "button", class: "link-button", onclick: () => go(segment.to) },
        `Show loop ${loop.id} on the timeline`) : null;
      return loopRow(loop, ctx, traceOf(loop.id), jump);
    });
    const all = loops.flatMap((loop) => traceOf(loop.id)?.checks || []);
    return {
      figure: h("ol", { class: "rv-loops", "aria-label": "Loops" }, ...rows),
      facts: facts(["Loops", String(loops.length)], all.length ? ["Checks", checkCounts(all)] : null),
    };
  }
  const buttons = model.segments.map((segment) => h("button", { type: "button", class: "button small", onclick: () => go(segment.to) },
    `Loop ${segment.loop} · ${where(steps[segment.step])}`));
  return {
    figure: h("div", { class: "ac-empty" }, h("p", {}, "No loop at this position: the instruction is transformed exactly, without a certificate."),
      buttons.length ? h("div", { class: "rv-actions" }, ...buttons) : null),
  };
}

// ---- the card --------------------------------------------------------------------------------------

const LENS_RENDER = { distribution: distributionLens, size: sizeLens, time: timeLens, checks: checksLens };

/** Keys that the card handles anywhere inside it, except where they mean something else. */
function ownsArrows(target) {
  return !target.closest?.("input, select, textarea, .jv-plot, .jv-canvas, .formula-scroll, .table-wrap, .tl-track, [contenteditable]");
}

/**
 * The analysis card for `result`. options: { analysisState: {} (kept across re-renders: lens,
 * position, variable, view, joint window, ...), onPosition({step, label} | null), loadJoint(payload,
 * signal), maxJointAxis, onRecord(), onAddWatch(text), onRemoveWatch(text), onSelectLine(line),
 * strategyTitles, breakpoints: [line] }.
 * Returns {element, go(index), key(event) -> handled, model} or null if there is nothing to show.
 */
export function analysisCard(result, options = {}) {
  const steps = result.steps || [];
  const hasSomething = result.status === "ok" || steps.length || (result.loops || []).length;
  if (!hasSomething) return null;
  const view = options.analysisState || {};
  const model = timelinePositions(result);
  if (!LENS_RENDER[view.lens]) view.lens = "distribution";
  if (!Number.isInteger(view.position) || view.position < 0 || view.position > model.final) view.position = model.final;
  const breakpoints = new Set(options.breakpoints || []);
  const stepLines = model.positions.map((p) => (p.kind === "step" ? steps[p.step]?.line : null));

  const titleId = `ac-title-${Math.random().toString(36).slice(2)}`;
  const whereText = h("span", { class: "ac-where" });
  const live = h("span", { class: "visually-hidden", "aria-live": "polite" });
  const lenses = segmented("Lens", LENSES, (key) => { view.lens = key; renderBody({ focusLens: true, reset: true }); },
    (key) => ({ distribution: "The distribution at this position", size: "The size of the closed form per step",
      time: "The time per step", checks: "The checks of the loops' certificates" }[key]));
  lenses.classList.add("ac-lenses");
  const instruction = h("p", { class: "ac-instruction" });
  const lead = h("div", { class: "ac-lead" });
  const controls = h("div", { class: "ac-controls" });
  const body = h("div", { class: "ac-body" });
  const factsHolder = h("div", { class: "ac-facts-holder" });
  const details = h("div", { class: "ac-details" });
  let joint = null;
  let jointPosition = null;
  let motions = [];
  // While the timeline moves, the parts above it keep the largest height they had for this lens,
  // so that its buttons stay under the pointer (#206); a new lens starts afresh.
  const reserved = new Map();
  const reserve = (reset) => {
    for (const part of [instruction, lead, controls, body, factsHolder]) {
      if (reset) { part.style.minHeight = ""; reserved.set(part, 0); }
      if (!steps.length || !part.isConnected) continue;
      const height = Math.max(reserved.get(part) || 0, part.offsetHeight);
      reserved.set(part, height);
      part.style.minHeight = height ? `${height}px` : "";
    }
  };

  const element = h("section", { class: "rv-card ac", "aria-labelledby": titleId },
    h("div", { class: "ac-head" },
      h("h3", { class: "rv-card-title", id: titleId }, "Analysis ", whereText),
      lenses),
    instruction, lead, controls, body, factsHolder);

  const ctx = {
    result, options, view, model, body, element,
    get index() { return view.position; },
    get position() { return model.positions[view.position]; },
    get data() { return positionData(result, model.positions[view.position]); },
    go: (i) => go(i),
    redraw: () => renderBody(),
    // the joint view is kept across positions: its window and colour scale stay (#289)
    joint: (variables, onGrid) => {
      if (!joint) {
        joint = jointView({ variables, view, maxAxis: options.maxJointAxis, scale: loadScale, onGrid: (grid) => joint?.onGrid(grid),
          load: (payload, signal) => options.loadJoint({ ...payload, ...(distributionSource(ctx)?.joint || {}) }, signal) });
        jointPosition = view.position;
      }
      joint.onGrid = onGrid;
      return joint;
    },
    showJoint: (instance) => {
      if (jointPosition !== view.position) {
        jointPosition = view.position;
        instance.refresh();
      } else {
        instance.draw();
      }
    },
  };

  const label = (i) => positionLabel(result, model.positions[i]);
  const timeline = steps.length
    ? timelineControl({ model, index: view.position, go: (i) => go(i), label, breakpoints, stepLines })
    : null;
  if (timeline) {
    element.append(h("div", { class: "ac-timeline" }, timeline.element));
  } else if (options.onRecord && result.status === "ok") {
    element.append(h("div", { class: "ac-record" },
      h("p", { class: "rv-note" }, "Steps were not recorded. Record them to move through the analysis: the distribution after each instruction, the iterates of loops, and where size and time go."),
      h("button", { type: "button", class: "button small", onclick: () => options.onRecord() }, "Record Steps")));
  }
  element.append(details, live);

  function notify() {
    const position = model.positions[view.position];
    const data = positionData(result, position);
    options.onPosition?.(data.step ? { ...data.step, label: position.kind === "iterate" ? `Iteration ${position.iteration}` : `Step ${data.step.index + 1}` } : null);
  }

  function renderHead() {
    const position = model.positions[view.position];
    const data = positionData(result, position);
    const text = position.kind === "final" ? (steps.length ? "· final result" : "") : `· ${label(view.position)}`;
    whereText.textContent = text;
    const step = data.step;
    instruction.replaceChildren();
    instruction.hidden = !step && !steps.length;
    if (!step && steps.length) {
      instruction.append(h("span", {}, `After all ${steps.length} ${steps.length === 1 ? "step" : "steps"}`),
        h("span", { class: "rv-note" }, result.distribution ? "the output distribution of the program" : "the analysis did not finish"));
    }
    if (step) {
      instruction.append(...[
        step.line && options.onSelectLine
          ? h("button", { type: "button", class: "link-button", onclick: () => options.onSelectLine(step.line) }, where(step))
          : h("span", {}, where(step)),
        h("code", { class: "ac-code", title: step.instruction }, step.summary || step.instruction),
        step.depth > 0 && BRANCH_TEXT[step.branch] ? h("span", { class: "rv-note" }, `Inside ${BRANCH_TEXT[step.branch]} (not normalised).`) : null,
        step.kind === "normalize" ? h("span", { class: "rv-note" }, "The distribution is conditioned on the observations.") : null,
        step.error ? h("span", { class: "rv-error-text", role: "alert" }, `Failed: ${step.error}`) : null].filter(Boolean));
    }
  }

  function renderBody({ focusLens = false, animate = false, reset = false } = {}) {
    for (const motion of motions) motion?.cancel();
    motions = [];
    const before = animate && !reducedMotion() && view.lens === "distribution" ? chartState(body.querySelector(".marginal-chart")) : null;
    // keep the keyboard focus on the same control across the re-rendering
    const active = document.activeElement;
    const focusKey = active && controls.contains(active) ? active.closest?.("[data-key]")?.dataset.key : null;
    press(lenses, view.lens);
    renderHead();
    const parts = LENS_RENDER[view.lens](ctx);
    lead.replaceChildren(...(parts.lead || []));
    controls.replaceChildren(...[parts.controls].filter(Boolean));
    body.replaceChildren(...[parts.figure].filter(Boolean));
    factsHolder.replaceChildren(...[parts.facts].filter(Boolean));
    details.replaceChildren(...(parts.details || []).filter(Boolean));
    if (focusLens) lenses.querySelector('[aria-pressed="true"]')?.focus({ preventScroll: true });
    else if (focusKey) element.querySelector(`.ac-controls [data-key="${CSS.escape(focusKey)}"]`)?.focus({ preventScroll: true });
    if (before) motions.push(tweenChart(() => body.querySelector(".marginal-chart"), before, TRANSITION_MS));
    reserve(reset);
  }

  function go(index) {
    const target = Math.max(0, Math.min(model.final, index));
    if (target === view.position) return;
    view.position = target;
    timeline?.update(target);
    renderBody({ animate: true });
    if (!timeline?.playing()) live.textContent = label(target);
    notify();
  }

  function key(event) {
    if (!steps.length) return false;
    let target = null;
    if (event.key === "F10") target = view.position + (event.shiftKey ? -1 : 1);
    else if (event.key === "F5") target = breakpointTarget(result, model, view.position, breakpoints, event.shiftKey ? -1 : 1);
    else if (element.contains(event.target) && ownsArrows(event.target) && !event.metaKey && !event.ctrlKey && !event.altKey) {
      if (event.key === "ArrowRight") target = view.position + 1;
      else if (event.key === "ArrowLeft") target = view.position - 1;
      else if (event.key === "Home") target = 0;
      else if (event.key === "End") target = model.final;
    }
    if (target === null) return false;
    event.preventDefault();
    timeline?.stop();
    go(target);
    return true;
  }

  renderBody();
  notify();
  return { element, go, key, model, stop: () => timeline?.stop() };
}
