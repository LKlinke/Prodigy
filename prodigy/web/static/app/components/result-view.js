// Result view: renders a Prodigy analysis result (schema prodigy/analysis-result@1).
//
// Self-contained on purpose: it needs only the result JSON, the CSS custom properties of
// result-view.css and optionally KaTeX, so that the VS Code integration can reuse it in a
// webview. Callbacks connect it to an editor (jump to a loop or line).
import { copyText, h, icon } from "./dom.js";
import { intervalBar, marginalChart, marginalTable, massSummary } from "./charts.js";
import { BOUND_TEXT, exactIsInformative, formatNumber, formatSeconds, formatValue, qualify, shortExact } from "./format.js";
import { formulaBox } from "./formula-box.js";
import { JOINT_VIEWS, jointView } from "./joint-view.js";

const ENGINE_NAMES = { sympy: "SymPy", ginac: "GiNaC", symengine: "SymEngine" };

export function guaranteeBadge(kind, label = null) {
  const text = BOUND_TEXT[kind] || BOUND_TEXT.unknown;
  return h("span", { class: "rv-badge", "data-kind": kind },
    h("span", { class: "glyph", "aria-hidden": "true" }, text.glyph), label ?? text.label);
}

/** A long expression as text in a bounded box (see formula-box.js). */
function expressionBox(text, label) {
  return formulaBox({ kind: "text", text: String(text ?? ""), label });
}

function exactSpan(exact, number) {
  if (!exactIsInformative(exact, number)) return null;
  return h("span", { class: "rv-exact mono", title: exact }, shortExact(exact));
}

function copyButton(label, text) {
  const button = h("button", { type: "button", class: "button small", title: label },
    icon("copy"), h("span", {}, label));
  button.addEventListener("click", async () => {
    const ok = await copyText(text);
    button.querySelector("span").textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => { button.querySelector("span").textContent = label; }, 1400);
  });
  return button;
}

// ---- answer card: the guarantee once, then the queries ------------------------------------

function residualText(result) {
  const r = result.bounds?.residual_mass_float;
  return r !== null && r !== undefined ? formatValue(result.bounds.residual_mass, r) : result.bounds?.residual_mass;
}

/** The one sentence that explains the overall guarantee. */
function guaranteeText(result) {
  const kind = result.bounds?.kind || "unknown";
  const loops = result.loops || [];
  if (kind === "exact") {
    return loops.length
      ? "Every loop was handled by a certified method or ran to completion."
      : "The program has no loops; its semantics is computed exactly.";
  }
  if (kind === "lower") {
    return `At most ${residualText(result)} of the probability mass is not yet accounted for; it may still terminate with any value or not at all. The intervals below include it. No termination assumption is made.`;
  }
  if (kind === "upper") {
    const spurious = loops.some((l) => (l.certificates || []).some((c) => c.finite_mass === false));
    return spurious
      ? "Values are upper bounds by Park's lemma. The invariant has infinite mass, so it may be a spurious fixed point and the true values can be smaller; see How it was computed."
      : "Values are upper bounds by Park's lemma. They are exact if the loops terminate almost surely, which is not checked.";
  }
  return "Approximations of different kinds were combined, or a side condition could not be checked; values come without a guarantee.";
}

function guaranteeHead(result) {
  const kind = result.bounds?.kind || "unknown";
  const timing = result.timing || {};
  const input = result.input || {};
  const meta = [
    timing.analysis !== undefined ? formatSeconds(timing.analysis) : null,
    ENGINE_NAMES[input.engine] || input.engine,
    input.normalize === false ? "not conditioned" : null,
  ].filter(Boolean).join(" · ");
  return h("div", { class: "rv-guarantee" },
    h("div", { class: "rv-guarantee-head" }, guaranteeBadge(kind), h("span", { class: "rv-meta" }, meta)),
    h("p", { class: "rv-guarantee-text" }, guaranteeText(result)));
}

/** Variables whose expected value a query already answers (`E[x]`, `Ex[x]`). */
function coveredVariables(queries) {
  return new Set(queries.filter((q) => q.kind === "expectation").map((q) => String(q.expr ?? "").trim()));
}

/** Expected values of the variables that no query covers, as one quiet line. */
function otherExpectations(result, queries) {
  const covered = coveredVariables(queries);
  const entries = Object.entries(result.distribution?.expectations || {}).filter(([name]) => !covered.has(name));
  if (!entries.length) return null;
  const kind = result.bounds?.kind;
  const qualifier = kind === "lower" ? "≥ " : kind === "upper" ? "≤ " : "= ";
  const items = entries.map(([name, e]) => h("li", { class: "rv-expectation" },
    h("span", { class: "mono" }, `E[${name}]`), " ",
    e.value === null || e.value === undefined
      ? h("span", { class: "rv-muted", title: e.error || "" }, "not available")
      : h("span", { class: "rv-expectation-value", title: e.value }, qualifier + expectationText(e))));
  return h("div", { class: "rv-expectations" },
    h("span", { class: "rv-expectations-label" }, queries.length ? "Other expected values" : "Expected values"),
    h("ul", {}, ...items));
}

/** "1/2" for a short exact value, else the decimal (the exact form is in the tooltip). */
function expectationText(e) {
  const number = formatValue(e.value, e.value_float);
  if (!exactIsInformative(e.value, e.value_float)) return number;
  return e.value.length <= 16 ? `${e.value} ≈ ${number}` : number;
}

function answerCard(result, options) {
  const queries = result.queries || [];
  const ok = result.status === "ok";
  const overall = ok ? result.bounds?.kind || "unknown" : null;
  const children = [];
  if (ok) children.push(guaranteeHead(result));
  if (queries.length) {
    children.push(h("ol", { class: "rv-queries", "aria-label": "Queries" }, ...queries.map((q) => queryRow(q, result, options, overall))));
  }
  if (ok) children.push(otherExpectations(result, queries));
  if (children.filter(Boolean).length === 0) return null;
  return h("section", { class: "rv-card rv-answer", "data-kind": overall || "error", "aria-label": "Answer" }, ...children);
}

// ---- errors --------------------------------------------------------------------------

const STATUS_TITLES = {
  needs_decision: "A loop needs a strategy",
  error: "The analysis failed",
  timeout: "Time limit reached",
  cancelled: "Analysis cancelled",
  crashed: "The analysis process stopped",
};

/** Verdicts of an invariant check: "refuted" is proven wrong, "unknown" is neither proven nor refuted. */
const VERDICTS = {
  verified: "verified",
  refuted: "refuted — the invariant is provably wrong",
  unknown: "not proven — neither proven nor refuted; the invariant may still be correct",
};

function verdictText(verdict) {
  return VERDICTS[verdict] || verdict;
}

const VERDICT_TITLES = { refuted: "Invariant refuted", unknown: "Invariant not proven" };

function errorView(result, options) {
  const error = result.error || {};
  const status = result.status === "error" && error.type === "crash" ? "crashed" : result.status;
  const verdict = status === "error" ? error.verdict : undefined;
  const title = VERDICT_TITLES[verdict] || STATUS_TITLES[status] || "The analysis failed";
  const children = [h("h3", { class: "rv-error-title" }, title),
    h("p", { class: "rv-error-message" }, error.message || "")];
  if (status === "needs_decision" && error.loop && options.onSelectLoop) {
    children.push(h("p", {}, h("button", { type: "button", class: "button", onclick: () => options.onSelectLoop(error.loop.id) },
      `Choose a strategy for loop ${error.loop.id}`)));
  }
  if (status === "timeout") {
    children.push(h("p", { class: "rv-muted" }, "Try fewer iterations, the GiNaC engine, or a strategy that certifies a closed form."));
  }
  const details = error.details;
  if (verdict) children.push(h("p", {}, `Verdict: ${verdictText(verdict)}.`));
  if (details?.counterexample) {
    children.push(h("p", {}, "Counterexample (a state on which the invariant and one unrolling of the loop differ): ",
      h("code", {}, Object.entries(details.counterexample).map(([k, v]) => `${k} = ${v}`).join(", ") || "—")));
  }
  if (details?.difference) {
    const label = verdict === "refuted" ? "Φ(I) − I does not vanish: " : "Φ(I) − I could not be shown to vanish: ";
    children.push(h("p", {}, label), expressionBox(details.difference, "Φ(I) − I"));
  }
  if (error.traceback) {
    children.push(h("details", { class: "disclosure rv-trace" }, h("summary", {}, "Technical details"),
      h("pre", { class: "rv-pre" }, error.traceback)));
  }
  const kind = status === "cancelled" ? "neutral" : status === "timeout" || verdict === "unknown" ? "warning" : "error";
  return h("div", { class: "rv-error", "data-tone": kind, role: status === "cancelled" ? null : "alert" }, ...children);
}

// ---- queries ---------------------------------------------------------------------------

function queryLabel(query) {
  const op = { probability: "Pr", expectation: "E", marginal: "Pr", optimization: "Opt", print: "Print", plot: "Plot" }[query.kind] || query.kind;
  if (query.kind === "print") return [h("span", { class: "rv-q-op" }, "!Print")];
  if (query.kind === "plot") return [h("span", { class: "rv-q-op" }, "!Plot"), `[${query.expr}]`];
  return [h("span", { class: "rv-q-op" }, op), `[${query.expr ?? ""}]`];
}

function boundText(query) {
  return qualify(query).text;
}

function boundSpan(query) {
  const { glyph, text } = qualify(query);
  const label = (BOUND_TEXT[query.bound] || BOUND_TEXT.unknown).label;
  return h("span", { class: "rv-q-bound", "data-kind": query.bound, title: label },
    glyph ? h("span", { class: "glyph", "aria-hidden": "true" }, glyph) : null, text);
}

/**
 * What a value row says about its bound. The overall guarantee is stated once, in the head of the
 * answer card; a query repeats its kind only where it differs. An interval of a lower bound is a
 * value, not a kind, so it is always shown.
 */
function boundNote(query, overall) {
  if (query.bound !== overall) return boundSpan(query);
  if (query.bound === "lower" && query.upper !== null && query.upper !== undefined) {
    return h("span", { class: "rv-q-interval" }, qualify(query).text);
  }
  return null;
}

/** The value of a query: a short exact form large with its decimal small, else the decimal large. */
function valueParts(exact, number) {
  const decimal = formatValue(exact, number);
  if (number === null || number === undefined) return [h("span", { class: "rv-q-value mono" }, exact ?? "—")];
  if (exactIsInformative(exact, number) && exact.length <= 12) {
    return [h("span", { class: "rv-q-value", title: exact }, exact), h("span", { class: "rv-q-approx" }, `≈ ${decimal}`)];
  }
  return [h("span", { class: "rv-q-value" }, decimal), exactSpan(exact, number)];
}

function queryRow(query, result, options, overall = null) {
  const lineButton = query.line && options.onSelectLine
    ? h("button", { type: "button", class: "link-button rv-q-line", onclick: () => options.onSelectLine(query.line) }, `line ${query.line}`)
    : (query.line ? h("span", { class: "rv-q-line" }, `line ${query.line}`) : null);
  const head = h("div", { class: "rv-q-head" }, h("span", { class: "rv-q-expr mono" }, ...queryLabel(query)), lineButton);

  if (query.kind === "probability" || query.kind === "expectation") {
    const value = h("div", { class: "rv-q-value-row" }, ...valueParts(query.value, query.value_float), boundNote(query, overall));
    const children = [head, value];
    if (query.kind === "probability" && query.value_float !== null && query.bound !== "exact") {
      children.push(intervalBar({ value: query.value_float, lower: query.lower_float, upper: query.upper_float, bound: query.bound },
        `Probability ${boundText(query)}`));
    }
    if (query.note) children.push(h("p", { class: "rv-note" }, query.note));
    return h("li", { class: "rv-query", "data-kind": query.bound }, ...children);
  }

  if (query.kind === "marginal" || query.kind === "plot") {
    const table = query.data?.marginal;
    const children = [head];
    if (table) {
      children.push(distributionFigure(table, result, query.expr));
    } else if (query.data?.joint) {
      children.push(jointTable(query.data.joint));
    } else if (query.value) {
      children.push(expressionBox(query.value, `Value of ${query.expr ?? "the query"}`));
    }
    if (query.note) children.push(h("p", { class: "rv-note" }, query.note));
    return h("li", { class: "rv-query rv-query-wide", "data-kind": query.bound }, ...children);
  }

  if (query.kind === "print") {
    return h("li", { class: "rv-query rv-query-wide" }, head, expressionBox(query.value || "", "Printed distribution"));
  }
  return h("li", { class: "rv-query rv-query-wide" }, head,
    h("p", {}, query.value ? `Optimum at ${query.data?.parameter} ∈ {${query.value}}` : "No solution found."),
    query.note ? h("p", { class: "rv-note" }, query.note) : null);
}

function jointTable(joint) {
  if (!joint.rows?.length) return h("p", { class: "rv-muted" }, "Too many states to list.");
  const head = h("tr", {}, ...joint.variables.map((v) => h("th", { scope: "col" }, v)), h("th", { scope: "col" }, "Probability"));
  const body = joint.rows.map((row) => h("tr", {}, ...joint.variables.map((v) => h("td", { class: "num" }, String(row.state[v] ?? 0))),
    h("td", { class: "num", title: row.prob }, formatValue(row.prob, row.p, 6))));
  return h("div", { class: "table-wrap" }, h("table", { class: "rv-table" }, h("thead", {}, head), h("tbody", {}, body)));
}

// ---- distribution ------------------------------------------------------------------------

// The scale of the bar charts (linear | log), remembered per browser. Same key as store.js would
// use; kept here because this component must not depend on the GUI's modules (VS Code webview).
const SCALE_KEY = "prodigy.gui.chartScale";

function loadScale() {
  try {
    return window.localStorage.getItem(SCALE_KEY) === "log" ? "log" : "linear";
  } catch {
    return "linear";
  }
}

function saveScale(scale) {
  try {
    window.localStorage.setItem(SCALE_KEY, scale);
  } catch {
    // ignore: storage is a convenience only
  }
}

export function distributionFigure(table, result, variable, mode = "chart") {
  const residual = result.bounds?.kind === "lower" ? result.bounds?.residual_mass_float : null;
  const residualExact = result.bounds?.residual_mass;
  const bound = result.bounds?.kind || "exact";
  const holder = h("div", { class: "rv-figure-body" });
  const toggle = h("div", { class: "segmented small", role: "group", "aria-label": `Show the distribution of ${variable} as` });
  const views = { chart: "Chart", table: "Table" };
  const scaleToggle = h("div", { class: "segmented small", role: "group", "aria-label": `Scale of the probability axis of ${variable}` });
  let current = mode;
  let scale = loadScale();
  let drawnWidth = 0;
  const show = (which) => {
    current = which;
    drawnWidth = holder.clientWidth || 560;
    holder.replaceChildren(which === "chart"
      ? marginalChart(table, { residual, residualExact, variable, bound, width: drawnWidth, scale })
      : marginalTable(table, { variable, bound }));
    for (const button of toggle.children) button.setAttribute("aria-pressed", String(button.dataset.view === which));
    for (const button of scaleToggle.children) button.setAttribute("aria-pressed", String(button.dataset.scale === scale));
    scaleToggle.hidden = which !== "chart";
  };
  for (const [key, label] of Object.entries({ linear: "Linear", log: "Log" })) {
    scaleToggle.append(h("button", { type: "button", "data-scale": key, "aria-pressed": "false",
      title: key === "log" ? "Logarithmic probability axis: small probabilities stay visible" : "Linear probability axis",
      onclick: () => { scale = key; saveScale(key); show("chart"); } }, label));
  }
  if (typeof ResizeObserver !== "undefined") {
    // draw the chart at its real size so that labels keep their font size
    new ResizeObserver(() => {
      if (current === "chart" && holder.clientWidth && Math.abs(holder.clientWidth - drawnWidth) > 16) show("chart");
    }).observe(holder);
  }
  for (const [key, label] of Object.entries(views)) {
    toggle.append(h("button", { type: "button", "data-view": key, "aria-pressed": "false", onclick: () => show(key) }, label));
  }
  show(mode);
  const caption = [];
  if (table.truncated && table.rows?.length) {
    caption.push(`Values 0–${table.rows[table.rows.length - 1].value}` +
      (table.tail_mass_float ? `; ${formatValue(table.tail_mass, table.tail_mass_float)} of the mass lies beyond.` : "."));
  }
  if (residual) caption.push(`Hatched: ${formatValue(residualExact, residual)} not yet accounted for.`);
  if (bound === "upper") caption.push("Hatched bars are upper bounds.");
  if (bound === "unknown") caption.push("Approximate values without a guarantee.");
  return h("figure", { class: "rv-figure" },
    h("div", { class: "rv-figure-head" }, scaleToggle, toggle),
    holder,
    caption.length ? h("figcaption", { class: "rv-note" }, caption.join(" ")) : null);
}

let cardCounter = 0;

/** A card of the result view: a titled section on the inspector's background. */
function card(className, title, ...content) {
  const id = `rv-card-${++cardCounter}`;
  return h("section", { class: `rv-card ${className}`, "aria-labelledby": id },
    h("h3", { class: "rv-card-title", id }, title), ...content);
}

/** A segmented control of mutually exclusive buttons; `onChoose(key)` runs on a click. */
function segmented(label, choices, onChoose, title = () => null) {
  const group = h("div", { class: "segmented small", role: "group", "aria-label": label });
  for (const [key, text] of choices) {
    group.append(h("button", { type: "button", "data-key": key, "aria-pressed": "false", title: title(key),
      onclick: () => onChoose(key) }, text));
  }
  return group;
}

function press(group, key) {
  for (const button of group.children) button.setAttribute("aria-pressed", String(button.dataset.key === key));
}

const part = (exact, p) => (exact === null || exact === undefined) && (p === null || p === undefined) ? null : { exact, p };

// variable names are identifiers, so this key never clashes with one
const PAIR = "pair:";
const MARGINAL_VIEWS = [["chart", "Chart"], ["table", "Table"]];

/** The mass line of a marginal: total mass, the values shown, the values beyond them, the "?" mass. */
function marginalMass(table, name, total, residual) {
  if (table.error || !table.truncated || !table.rows?.length) return massSummary({ total, residual });
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

function distributionFacts(dist) {
  const facts = h("dl", { class: "rv-facts" });
  const addFact = (label, value, title) => facts.append(h("div", {}, h("dt", {}, label), h("dd", { title: title || "" }, value)));
  addFact("Support", dist.finite_support ? "finite" : "infinite");
  if (dist.error_probability && dist.error_probability !== "0") {
    addFact("Observation violated", formatValue(dist.error_probability, dist.error_probability_float, 6), dist.error_probability);
  }
  if (dist.parameters?.length) addFact("Parameters", dist.parameters.join(", "));
  return facts;
}

function generatingFunction(dist) {
  return h("details", { class: "disclosure rv-gf" }, h("summary", {}, "Generating function"),
    h("div", { class: "disclosure-body" },
      formulaBox({ latex: dist.latex, text: dist.closed_form, label: "Generating function" }),
      h("div", { class: "rv-actions" }, dist.latex ? copyButton("LaTeX", dist.latex) : null, copyButton("Text", dist.closed_form))));
}

/**
 * The distribution card: one figure with one control row. The variable choice lists every
 * variable and, with two or more and a `loadJoint` callback, a pair, which switches the same
 * figure to the joint view. The view (chart/table, or heatmap/3D/table) and the scale follow.
 * Under the figure one mass line for whatever it shows.
 */
function distributionCard(result, options) {
  const dist = result.distribution;
  const inferred = distValuesBlock(result);
  if (!dist) return inferred ? card("rv-distribution", "Output distribution", inferred) : null;
  const bound = result.bounds?.kind || "exact";
  const total = part(dist.mass, dist.mass_float);
  const residual = bound === "lower" ? part(result.bounds?.residual_mass, result.bounds?.residual_mass_float) : null;
  const variables = Object.keys(dist.marginals || {});
  const withPair = variables.length >= 2 && Boolean(options.loadJoint);
  const view = options.jointState || {};
  if (!(view.subject === PAIR && withPair) && !variables.includes(view.subject)) view.subject = variables[0];
  if (view.marginalMode !== "table") view.marginalMode = "chart";

  const figure = h("div", { class: "rv-figure-body" });
  const mass = h("div", { class: "rv-mass-holder" });
  const legend = h("p", { class: "rv-note", hidden: true });
  const viewHolder = h("span", { class: "rv-control" });
  let joint = null;
  let drawnWidth = 0;

  const isPair = () => view.subject === PAIR;
  const mode = () => (isPair() ? view.mode || "heatmap" : view.marginalMode);
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
      if (!joint) {
        joint = jointView({ variables, load: options.loadJoint, view, maxAxis: options.maxJointAxis, scale: loadScale,
          onGrid: (grid) => mass.replaceChildren(jointMass(grid, total, residual)) });
      }
      if (figure.firstChild !== joint.element) figure.replaceChildren(joint.element);
      joint.draw();
      return;
    }
    const name = view.subject;
    const table = dist.marginals[name];
    drawnWidth = figure.clientWidth || 560;
    if (table.error) {
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
    mass.replaceChildren(marginalMass(table, name, total, residual));
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
  const controls = variables.length
    ? h("div", { class: "rv-controls" }, variables.length > 1 || withPair ? subjects : null, viewHolder, scaleToggle)
    : null;
  if (variables.length) render();
  else mass.replaceChildren(massSummary({ total, residual }));
  return card("rv-distribution", "Output distribution",
    distributionFacts(dist),
    controls,
    variables.length ? h("figure", { class: "rv-figure" }, figure, legend) : null,
    mass,
    generatingFunction(dist),
    inferred);
}

// ---- dist values (nested inference) ------------------------------------------------------

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
  const facts = h("dl", { class: "rv-facts" });
  const addFact = (label, text, title) => facts.append(h("div", {}, h("dt", {}, label), h("dd", { title: title || "" }, text)));
  if (entry.depends_on_state) {
    const kind = result.bounds?.kind;
    const qualifier = kind === "lower" ? "at least " : kind === "upper" ? "at most " : "";
    addFact("Probability of this value", qualifier + formatValue(value.probability, value.probability_float, 6), value.probability);
  }
  if (value.mean !== null && value.mean !== undefined) addFact("Expected value", formatValue(value.mean, value.mean_float), value.mean);
  if (value.normalizing_constant) addFact("Normalising constant Z", formatValue(value.normalizing_constant, value.normalizing_constant_float, 6), value.normalizing_constant);
  if (value.finite_support !== null && value.finite_support !== undefined) addFact("Support", value.finite_support ? "finite" : "infinite");
  if (facts.children.length) children.push(facts);
  if (value.note) children.push(h("p", { class: "rv-note" }, value.note));
  if (value.table && !value.table.error && value.table.rows?.length) {
    // the closed form is exact, whatever the guarantee of the program state
    children.push(distributionFigure(value.table, { bounds: { kind: "exact" } }, variable, "table"));
  }
  return h("div", { class: "rv-dist-card" }, ...children);
}

/** The values of the `dist` variables, a block of the distribution card. */
function distValuesBlock(result) {
  const entries = result.distributions || [];
  if (!entries.length) return null;
  const cards = [];
  for (const entry of entries) {
    for (const value of entry.values) cards.push(distValueCard(entry.variable, value, entry, result));
  }
  return h("section", { class: "rv-block rv-inferred", "aria-label": "Inferred distributions" },
    h("h4", { class: "rv-block-title" }, "Inferred distributions"),
    h("p", { class: "rv-note" }, "Values of the dist variables at the end of the program, apart from the output distribution. Each is the normalised distribution of its infer(…), computed exactly."),
    ...cards);
}

// ---- loops -------------------------------------------------------------------------------

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

function loopRow(loop, options) {
  const titles = options.strategyTitles || {};
  const where = loop.line ? `line ${loop.line}` : "";
  const name = loop.id !== null ? `Loop ${loop.id}` : "Loop";
  const kind = loop.failed ? "error" : loop.kind;
  const header = h("div", { class: "rv-loop-head" },
    h("div", { class: "rv-loop-title" }, h("span", { class: "rv-loop-name" }, name), " ",
      where && options.onSelectLine ? h("button", { type: "button", class: "link-button rv-loop-line", onclick: () => options.onSelectLine(loop.line) }, where)
        : h("span", { class: "rv-loop-line" }, where)),
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
    loop.failed ? h("p", { class: "rv-loop-message rv-error-text" }, loop.failed) : null, ...messages, details);
}

/** How it was computed: one row per loop, and the way into the Steps tab. */
function computedCard(result, options) {
  const loops = result.loops || [];
  const steps = result.steps || [];
  if (!loops.length && !steps.length) return null;
  const children = [];
  if (loops.length) children.push(h("ol", { class: "rv-loops", "aria-label": "Loops" }, ...loops.map((loop) => loopRow(loop, options))));
  else children.push(h("p", { class: "rv-note" }, "The program has no loops; every instruction is transformed exactly."));
  if (steps.length) {
    const hints = (result.trace?.hints || []).filter((hint) => hint.kind !== "limits");
    const label = `Step Through ${steps.length} ${steps.length === 1 ? "Step" : "Steps"}`;
    children.push(h("div", { class: "rv-steps-row" },
      options.onOpenSteps
        ? h("button", { type: "button", class: "button", onclick: () => options.onOpenSteps() }, label)
        : h("span", { class: "rv-muted" }, `${steps.length} steps recorded.`)),
    ...hints.map((hint) => h("p", { class: "rv-note" }, hint.text)));
  }
  return card("rv-computed", "How it was computed", ...children);
}

// ---- details: warnings, console output, JSON ---------------------------------------------------

/** A visible notice for warnings; their text is in the details below. */
function warningNotice(result, details) {
  const count = result.warnings?.length || 0;
  if (!count) return null;
  const open = () => {
    details.open = true;
    details.querySelector("summary")?.focus();
    details.scrollIntoView?.({ block: "nearest" });
  };
  return h("p", { class: "notice warning rv-warning-notice" }, icon("warning"),
    h("span", {}, count === 1 ? "The analysis reported a warning. " : `The analysis reported ${count} warnings. `,
      h("button", { type: "button", class: "link-button", onclick: open }, count === 1 ? "Show it" : "Show them")));
}

function detailsLine(result) {
  const warnings = result.warnings || [];
  const parts = [];
  if (warnings.length) parts.push(warnings.length === 1 ? "1 warning" : `${warnings.length} warnings`);
  if (result.log?.trim()) parts.push("console output");
  parts.push("JSON");
  const body = [];
  if (warnings.length) {
    body.push(h("ul", { class: "rv-warnings", "aria-label": "Warnings" },
      ...warnings.map((w) => h("li", { class: "notice warning" }, icon("warning"), h("span", {}, w)))));
  }
  if (result.log?.trim()) {
    body.push(h("div", { class: "rv-log-block" }, h("h4", { class: "rv-block-title" }, "Console output"),
      h("pre", { class: "rv-pre rv-log", tabindex: "0", "aria-label": "Console output" }, cleanLog(result.log))));
  }
  const json = JSON.stringify(result, null, 2);
  body.push(h("div", { class: "rv-actions" }, h("a", { class: "button small", download: "prodigy-result.json",
    href: URL.createObjectURL(new Blob([json], { type: "application/json" })) }, icon("download"), "Result as JSON")));
  return h("details", { class: "disclosure rv-details" },
    h("summary", {}, "Details", h("span", { class: "rv-details-hint" }, parts.join(" · "))),
    h("div", { class: "disclosure-body" }, ...body));
}

/** Applies carriage returns the way a terminal would. */
export function cleanLog(text) {
  return text.split("\n").map((line) => {
    const parts = line.split("\r");
    return parts[parts.length - 1] || parts.filter(Boolean).pop() || "";
  }).join("\n").trim();
}

// ---- public API ----------------------------------------------------------------------------

/**
 * Renders `result` into `container`.
 * options: { onSelectLoop(id), onSelectLine(line), onOpenSteps(), strategyTitles: {name: title}, stale: bool, elapsed,
 *            loadJoint(payload, signal) -> Promise<grid>, jointState: {}, maxJointAxis }
 */
export function renderResult(container, result, options = {}) {
  const children = [];
  if (options.stale) {
    children.push(h("p", { class: "notice" }, icon("info"),
      h("span", {}, "The program or setup changed since this result. Analyze again to update it.")));
  }
  const details = detailsLine(result);
  children.push(warningNotice(result, details));
  if (result.status !== "ok") children.push(errorView(result, options));
  children.push(answerCard(result, options));
  if (result.status === "ok") children.push(distributionCard(result, options));
  children.push(computedCard(result, options), details);
  container.replaceChildren(h("div", { class: "result-view", "data-stale": options.stale ? "true" : "false" },
    ...children.filter(Boolean)));
}

/** Progress view while an analysis runs. state: {elapsed, timeLimit, progress: [...], log, loops} */
export function renderRunning(container, state, options = {}) {
  const titles = options.strategyTitles || {};
  const bars = (state.progress || []).map((p) => {
    const fraction = p.total ? Math.min(1, p.done / p.total) : null;
    const label = `Loop ${p.loop ?? "?"} · ${titles[p.strategy] || p.strategy}` + (p.message ? ` · ${p.message}` : "");
    return h("div", { class: "rv-progress" },
      h("div", { class: "rv-progress-label" }, h("span", { title: label }, label),
        h("span", { class: "num" }, fraction !== null ? `${Math.round(fraction * 100)} %` : "")),
      fraction !== null
        ? h("progress", { max: "1", value: String(fraction), "aria-label": label })
        : h("progress", { "aria-label": label }));
  });
  const log = cleanLog(state.log || "").split("\n").slice(-12).join("\n");
  // a fixed height, scrolled to the newest line: the log changes on every poll and its lines wrap
  // differently, which must not move the view
  // (also while still empty, so that its first line does not move the view either)
  const logBox = h("pre", { class: "rv-pre rv-log rv-running-log", "aria-label": "Console output", tabindex: log ? "0" : null }, log);
  container.replaceChildren(h("div", { class: "result-view" },
    h("div", { class: "rv-running" },
      h("div", { class: "rv-running-head" }, h("span", { class: "spinner", "aria-hidden": "true" }),
        h("span", {}, `Analyzing · ${formatSeconds(state.elapsed || 0)}`),
        state.timeLimit ? h("span", { class: "rv-muted" }, `limit ${formatSeconds(state.timeLimit)}`) : null),
      ...bars,
      logBox)));
  logBox.scrollTop = logBox.scrollHeight;
}

export { formatValue, formatNumber };
