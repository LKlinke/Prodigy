// Result view: renders a Prodigy analysis result (schema prodigy/analysis-result@1).
//
// Self-contained on purpose: it needs only the result JSON, the CSS custom properties of
// result-view.css and analysis-card.css and optionally KaTeX, so that the VS Code integration can
// reuse it in a webview. Callbacks connect it to an editor (jump to a loop or line, highlight the
// step of the analysis card's timeline).
import { h, icon } from "./dom.js";
import { intervalBar } from "./charts.js";
import { BOUND_TEXT, exactIsInformative, formatNumber, formatSeconds, formatValue, qualify, shortExact } from "./format.js";
import { analysisCard, verdictText } from "./analysis-card.js";
import { distributionFigure, expressionBox, guaranteeBadge } from "./parts.js";

export { guaranteeBadge, distributionFigure };

const ENGINE_NAMES = { sympy: "SymPy", ginac: "GiNaC", symengine: "SymEngine" };

function exactSpan(exact, number) {
  if (!exactIsInformative(exact, number)) return null;
  return h("span", { class: "rv-exact mono", title: exact }, shortExact(exact));
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
 * Renders `result` into `container` and returns {analysis} (the analysis card's controller, or
 * null; see analysis-card.js).
 * options: { onSelectLoop(id), onSelectLine(line), strategyTitles: {name: title}, stale: bool, elapsed,
 *            loadJoint(payload, signal) -> Promise<grid>, analysisState: {}, maxJointAxis,
 *            onPosition(step | null), onRecord(), onAddWatch(text), onRemoveWatch(text), breakpoints }
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
  const analysis = analysisCard(result, options);
  children.push(analysis?.element, details);
  container.replaceChildren(h("div", { class: "result-view", "data-stale": options.stale ? "true" : "false" },
    ...children.filter(Boolean)));
  return { analysis };
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
