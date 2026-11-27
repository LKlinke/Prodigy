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
// The lenses Size and Time are in cost-lenses.js, Checks in checks-lens.js; the layout parts all lenses
// share in lens-parts.js.
import { chartFocus, marginalChart, marginalTable, massSummary } from "./charts.js";
import { checksLens } from "./checks-lens.js";
import { sizeLens, timeLens } from "./cost-lenses.js";
import { h } from "./dom.js";
import { chartLegend, formatNumber, formatValue, qualify } from "./format.js";
import { formulaBox } from "./formula-box.js";
import { JOINT_VIEWS, jointView } from "./joint-view.js";
import { disclosure, facts, where } from "./lens-parts.js";
import { chartState, reducedMotion, tweenChart } from "./motion.js";
import { copyButton, distributionFigure, guaranteeBadge, press, segmented } from "./parts.js";
import { breakpointTarget, marginalGap, positionData, positionLabel, timelineControl, timelinePositions } from "./timeline.js";
import { onViewOptionsChange, viewOptions, viewOptionsControl } from "./view-options.js";

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

// ---- distribution ---------------------------------------------------------------------------------

/** The mass line of a marginal: total mass, the values shown, the values beyond them, the "?" mass. */
export function marginalMass(table, name, total, residual) {
  if (!table || table.error || !table.truncated || !table.rows?.length) return massSummary({ total, residual });
  const first = table.rows[0].value;
  const last = table.rows[table.rows.length - 1].value;
  const shownFloat = table.shown_mass_float ?? (table.mass_float ?? 0) - (table.tail_mass_float ?? 0);
  return massSummary({ total, residual,
    shown: part(table.shown_mass, shownFloat), shownLabel: `Shown: ${name} = ${first}–${last}`,
    beyond: part(table.tail_mass, table.tail_mass_float), beyondLabel: `Beyond: ${name} > ${last}` });
}

/** The mass line of a window of the joint view (null: no grid yet). */
export function jointMass(grid, total, residual) {
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
    const views = isPair() ? Object.entries(JOINT_VIEWS) : MARGINAL_VIEWS;
    if (viewHolder.dataset.kind !== (isPair() ? "pair" : "single")) {
      viewHolder.dataset.kind = isPair() ? "pair" : "single";
      viewHolder.replaceChildren(segmented(isPair() ? "Show the joint distribution as" : "Show the distribution as", views, (key) => {
        if (isPair()) view.mode = key; else view.marginalMode = key;
        render();
      }));
    }
    press(viewHolder.firstChild, mode());
    optionsControl.setAvailable(mode() !== "table");
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
      // the selected bar of each variable stays selected from position to position (#462)
      const selection = ((view.selectedBars ||= {})[name] ||= {});
      const refocus = chartFocus(figure);
      figure.replaceChildren(mode() === "chart"
        ? marginalChart(table, { residual: residual?.p, residualExact: residual?.exact, variable: name, bound, width: drawnWidth, view: viewOptions(), selection })
        : marginalTable(table, { variable: name, bound }));
      refocus();
      // the guarantee in words, not only implied by the intervals (#405)
      legend.textContent = chartLegend(bound);
      legend.hidden = false;
    }
    mass.replaceChildren(data.iterate ? loopMass(source.snap, data.iterate) : marginalMass(gap ? null : table, name, total, residual));
  };

  const pairLabel = variables.length === 2 ? `${variables[0]} and ${variables[1]}` : "Two variables";
  const subjects = segmented("Variable", [...variables.map((v) => [v, v]), ...(withPair ? [[PAIR, pairLabel]] : [])],
    (key) => { view.subject = key; render(); },
    (key) => (key === PAIR ? "Joint distribution of two variables" : `Distribution of ${key}`));
  // the view options of the plots (#452): scale, colours, intervals; a change redraws the figure
  const optionsControl = viewOptionsControl({ label: "View options of the plots: scale, colours and intervals" });
  onViewOptionsChange(figure, () => { if (variables.length) render(); });
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
    ? h("div", { class: "rv-controls" }, variables.length > 1 || withPair ? subjects : null, viewHolder, optionsControl.button)
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
    // the inspector of the view options sits beside the plot (or above it in a narrow card), never over it
    figure: h("figure", { class: "rv-figure" }, optionsControl.frame(h("div", { class: "ac-plot" }, figure, legend))),
    facts: mass,
    details: source.final
      ? [finalGeneratingFunction(result.distribution), distValuesBlock(result), finalFacts(result.distribution), watches]
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

// The output distribution in the form that is shown (#404): with "Simplify the shown result", the result says
// which form the bounded search chose; a simplified form can be switched back to the form as computed.
function finalGeneratingFunction(dist) {
  const display = dist.display;
  if (!display?.simplified || !dist.computed_form) {
    const block = generatingFunction(dist.latex, dist.closed_form, false);
    if (display?.note) block.querySelector(".disclosure-body").prepend(h("p", { class: "rv-note" }, display.note));
    return block;
  }
  const forms = {
    shown: { latex: dist.latex, text: dist.closed_form },
    computed: { latex: dist.computed_latex, text: dist.computed_form },
  };
  const holder = h("div", {});
  const actions = h("div", { class: "rv-actions" });
  const sizes = display.computed_ops != null && display.shown_ops != null
    ? ` ${display.shown_ops} instead of ${display.computed_ops} operations.` : "";
  const note = h("p", { class: "rv-note" });
  const choose = (key) => {
    const form = forms[key];
    holder.replaceChildren(formulaBox({ latex: form.latex, text: form.text, label: "Generating function" }));
    actions.replaceChildren(form.latex ? copyButton("LaTeX", form.latex) : null, form.text ? copyButton("Text", form.text) : null);
    note.textContent = key === "shown" ? `${display.note}${sizes}` : "As computed by the analysis; the same generating function.";
    press(toggle, key);
  };
  const toggle = segmented("Form of the generating function", [["shown", "Simplified"], ["computed", "As computed"]], choose);
  choose("shown");
  return h("details", { class: "disclosure rv-gf" }, h("summary", {}, "Generating function"),
    h("div", { class: "disclosure-body" }, h("div", {}, toggle), note, holder, actions));
}

// the dist variables are not part of the distribution; their values are listed apart (#217: with KaTeX)
function snapshotDists(snap) {
  const dists = snap.dists || [];
  if (!dists.length) return null;
  const several = new Set(dists.map((d) => d.variable)).size < dists.length;
  return disclosure(`Values of dist variables (${dists.length})`, ...dists.map((d) => h("div", { class: "rv-dist-card" },
    h("div", { class: "rv-card-head" }, h("span", { class: "mono rv-dist-infer" }, `${d.variable} = ${d.infer ?? "no value"}`),
      several ? h("span", { class: "rv-meta", title: d.probability }, `probability ${formatValue(d.probability, Number(d.probability) || null, 6)}`) : null,
      d.latex ? h("span", { class: "rv-actions" }, copyButton("LaTeX", d.latex)) : null),
    d.closed_form ? formulaBox({ latex: d.latex, text: d.closed_form, label: `Generating function of ${d.variable}` }) : null)));
}

function distValueCard(variable, value, entry, result) {
  const head = h("div", { class: "rv-card-head" },
    h("span", { class: "mono rv-dist-infer" }, `${variable} = ${value.infer ?? "no value"}`),
    h("span", { class: "rv-actions" },
      value.closed ? guaranteeBadge("exact") : null,
      value.latex ? copyButton("LaTeX", value.latex) : null,
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
 * strategyTitles, breakpoints: [line], fromCharts: {variable: chartState} of the charts streamed while
 * the analysis ran (#405), where the final chart starts its transition }.
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
        joint = jointView({ variables, view, maxAxis: options.maxJointAxis, options: viewOptions, onGrid: (grid) => joint?.onGrid(grid),
          load: (payload, signal, onPartial) => options.loadJoint({ ...payload, ...(distributionSource(ctx)?.joint || {}) }, signal, onPartial) });
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

  function renderBody({ focusLens = false, animate = false, reset = false, from = null } = {}) {
    for (const motion of motions) motion?.cancel();
    motions = [];
    const before = from || (animate && !reducedMotion() && view.lens === "distribution" ? chartState(body.querySelector(".marginal-chart")) : null);
    // keep the keyboard focus on the same control across the re-rendering
    const active = document.activeElement;
    const focusKey = active && controls.contains(active) ? active.closest?.("[data-key]")?.dataset.key : null;
    const refocusChart = chartFocus(body);
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
    else refocusChart();
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

  // right after an analysis, the chart continues from the chart streamed while it ran (#405)
  const streamed = view.lens === "distribution" && view.position === model.final ? options.fromCharts?.[view.subject ?? Object.keys(result.distribution?.marginals || {})[0]] : null;
  renderBody({ from: streamed || null });
  notify();
  return { element, go, key, model, stop: () => timeline?.stop() };
}
