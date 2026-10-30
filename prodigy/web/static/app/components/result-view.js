// Result view: renders a Prodigy analysis result (schema prodigy/analysis-result@1).
//
// Self-contained on purpose: it needs only the result JSON, the CSS custom properties of
// result-view.css and optionally KaTeX, so that the VS Code integration can reuse it in a
// webview. Callbacks connect it to an editor (jump to a loop or line).
import { copyText, h, icon } from "./dom.js";
import { intervalBar, marginalChart, marginalTable } from "./charts.js";
import { BOUND_TEXT, exactIsInformative, formatNumber, formatSeconds, formatValue, qualify, shortExact } from "./format.js";
import { renderMath } from "./math.js";

const ENGINE_NAMES = { sympy: "SymPy", ginac: "GiNaC", symengine: "SymEngine" };

function guaranteeBadge(kind, label = null) {
  const text = BOUND_TEXT[kind] || BOUND_TEXT.unknown;
  return h("span", { class: "rv-badge", "data-kind": kind },
    h("span", { class: "glyph", "aria-hidden": "true" }, text.glyph), label ?? text.label);
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

function section(title, ...content) {
  return h("section", { class: "rv-section" }, h("h3", { class: "rv-section-title" }, title), ...content);
}

// ---- summary -------------------------------------------------------------------------

function residualText(result) {
  const r = result.bounds?.residual_mass_float;
  return r !== null && r !== undefined ? formatValue(result.bounds.residual_mass, r) : result.bounds?.residual_mass;
}

function summary(result, options) {
  const kind = result.bounds?.kind || "unknown";
  const loops = result.loops || [];
  let text;
  if (kind === "exact") {
    text = loops.length
      ? "Every loop was handled by a certified method or ran to completion."
      : "The program has no loops; its semantics is computed exactly.";
  } else if (kind === "lower") {
    text = `At most ${residualText(result)} of the probability mass is not yet accounted for; it may still terminate with any value or not at all. The intervals below include it. No termination assumption is made.`;
  } else if (kind === "upper") {
    const spurious = loops.some((l) => (l.certificates || []).some((c) => c.finite_mass === false));
    text = spurious
      ? "Values are upper bounds by Park's lemma. The invariant has infinite mass, so it may be a spurious fixed point and the true values can be smaller; see Loops."
      : "Values are upper bounds by Park's lemma. They are exact if the loops terminate almost surely, which is not checked.";
  } else {
    text = "Approximations of different kinds were combined, or a side condition could not be checked; values come without a guarantee.";
  }
  const timing = result.timing || {};
  const input = result.input || {};
  const meta = [
    timing.analysis !== undefined ? formatSeconds(timing.analysis) : null,
    ENGINE_NAMES[input.engine] || input.engine,
    input.normalize === false ? "not conditioned" : null,
  ].filter(Boolean).join(" · ");
  return h("div", { class: "rv-summary", "data-kind": kind },
    h("div", { class: "rv-summary-head" }, guaranteeBadge(kind), h("span", { class: "rv-meta" }, meta)),
    h("p", { class: "rv-summary-text" }, text));
}

// ---- errors --------------------------------------------------------------------------

const STATUS_TITLES = {
  needs_decision: "A loop needs a strategy",
  error: "The analysis failed",
  timeout: "Time limit reached",
  cancelled: "Analysis cancelled",
  crashed: "The analysis process stopped",
};

function errorView(result, options) {
  const error = result.error || {};
  const status = result.status === "error" && error.type === "crash" ? "crashed" : result.status;
  const children = [h("h3", { class: "rv-error-title" }, STATUS_TITLES[status] || "The analysis failed"),
    h("p", { class: "rv-error-message" }, error.message || "")];
  if (status === "needs_decision" && error.loop && options.onSelectLoop) {
    children.push(h("p", {}, h("button", { type: "button", class: "button", onclick: () => options.onSelectLoop(error.loop.id) },
      `Choose a strategy for loop ${error.loop.id}`)));
  }
  if (status === "timeout") {
    children.push(h("p", { class: "rv-muted" }, "Try fewer iterations, the GiNaC engine, or a strategy that certifies a closed form."));
  }
  const details = error.details;
  if (details?.counterexample) {
    children.push(h("p", {}, "Counterexample (a state on which the invariant and one unrolling of the loop differ): ",
      h("code", {}, Object.entries(details.counterexample).map(([k, v]) => `${k} = ${v}`).join(", ") || "—")));
  }
  if (details?.difference) {
    children.push(h("p", {}, "Φ(I) − I does not vanish: ", h("code", { class: "rv-wrap" }, details.difference)));
  }
  if (error.traceback) {
    children.push(h("details", { class: "disclosure rv-trace" }, h("summary", {}, "Technical details"),
      h("pre", { class: "rv-pre" }, error.traceback)));
  }
  const kind = status === "cancelled" ? "neutral" : status === "timeout" ? "warning" : "error";
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

function queryRow(query, result, options) {
  const lineButton = query.line && options.onSelectLine
    ? h("button", { type: "button", class: "link-button rv-q-line", onclick: () => options.onSelectLine(query.line) }, `line ${query.line}`)
    : (query.line ? h("span", { class: "rv-q-line" }, `line ${query.line}`) : null);
  const head = h("div", { class: "rv-q-head" }, h("span", { class: "rv-q-expr mono" }, ...queryLabel(query)), lineButton);

  if (query.kind === "probability" || query.kind === "expectation") {
    const value = h("div", { class: "rv-q-value-row" },
      h("span", { class: "rv-q-value" }, formatValue(query.value, query.value_float)),
      exactSpan(query.value, query.value_float),
      boundSpan(query));
    const children = [head, value];
    if (query.kind === "probability" && query.value_float !== null) {
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
      children.push(h("code", { class: "rv-wrap" }, query.value));
    }
    if (query.note) children.push(h("p", { class: "rv-note" }, query.note));
    return h("li", { class: "rv-query rv-query-wide", "data-kind": query.bound }, ...children);
  }

  if (query.kind === "print") {
    return h("li", { class: "rv-query rv-query-wide" }, head, h("code", { class: "rv-wrap" }, query.value || ""));
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

function distributionFigure(table, result, variable, mode = "chart") {
  const residual = result.bounds?.kind === "lower" ? result.bounds?.residual_mass_float : null;
  const residualExact = result.bounds?.residual_mass;
  const bound = result.bounds?.kind || "exact";
  const holder = h("div", { class: "rv-figure-body" });
  const toggle = h("div", { class: "segmented small", role: "group", "aria-label": `Show the distribution of ${variable} as` });
  const views = { chart: "Chart", table: "Table" };
  let current = mode;
  let drawnWidth = 0;
  const show = (which) => {
    current = which;
    drawnWidth = holder.clientWidth || 560;
    holder.replaceChildren(which === "chart"
      ? marginalChart(table, { residual, residualExact, variable, bound, width: drawnWidth })
      : marginalTable(table, { variable, bound }));
    for (const button of toggle.children) button.setAttribute("aria-pressed", String(button.dataset.view === which));
  };
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
    h("div", { class: "rv-figure-head" }, toggle),
    holder,
    caption.length ? h("figcaption", { class: "rv-note" }, caption.join(" ")) : null);
}

function distributionSection(result) {
  const dist = result.distribution;
  if (!dist) return null;
  const math = h("div", { class: "rv-closed-form", tabindex: "0", "aria-label": `Generating function ${dist.closed_form}` });
  renderMath(math, dist.latex, dist.closed_form);
  const facts = h("dl", { class: "rv-facts" });
  const addFact = (label, value, title) => facts.append(h("div", {}, h("dt", {}, label), h("dd", { title: title || "" }, value)));
  addFact("Total mass", formatValue(dist.mass, dist.mass_float, 6), dist.mass);
  if (dist.error_probability && dist.error_probability !== "0") {
    addFact("Observation violated", formatValue(dist.error_probability, dist.error_probability_float, 6), dist.error_probability);
  }
  addFact("Support", dist.finite_support ? "finite" : "infinite");
  if (dist.parameters?.length) addFact("Parameters", dist.parameters.join(", "));

  const expectations = Object.entries(dist.expectations || {});
  let expectationRow = null;
  if (expectations.length) {
    const kind = result.bounds?.kind;
    const qualifier = kind === "lower" ? "at least " : kind === "upper" ? "at most " : "";
    const rows = expectations.map(([name, e]) => h("tr", {},
      h("td", { class: "mono" }, name),
      h("td", { class: "num" }, e.value === null ? "—" : qualifier + formatValue(e.value, e.value_float)),
      h("td", { class: "mono exact-cell", title: e.value || e.error || "" }, e.value ?? "not available")));
    expectationRow = h("div", { class: "table-wrap" }, h("table", { class: "rv-table rv-expectations" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Variable"), h("th", { scope: "col" }, "Expected value"),
        h("th", { scope: "col" }, "Exact"))),
      h("tbody", {}, ...rows)));
  }

  const variables = Object.keys(dist.marginals || {});
  let marginals = null;
  if (variables.length) {
    const holder = h("div", {});
    const picker = h("div", { class: "segmented small", role: "group", "aria-label": "Variable" });
    const choose = (name) => {
      const table = dist.marginals[name];
      holder.replaceChildren(table.error ? h("p", { class: "rv-muted" }, `Not available: ${table.error}`) : distributionFigure(table, result, name));
      for (const button of picker.children) button.setAttribute("aria-pressed", String(button.dataset.variable === name));
    };
    for (const name of variables) {
      picker.append(h("button", { type: "button", "data-variable": name, "aria-pressed": "false", onclick: () => choose(name) }, name));
    }
    choose(variables[0]);
    marginals = h("div", { class: "rv-marginals" },
      h("div", { class: "rv-subhead" }, h("h4", {}, "Marginal distributions"), variables.length > 1 ? picker : null), holder);
  }

  const latexCopy = dist.latex ? copyButton("LaTeX", dist.latex) : null;
  return section("Output distribution",
    h("div", { class: "rv-card" },
      h("div", { class: "rv-card-head" }, h("span", { class: "rv-muted" }, "Generating function"),
        h("span", { class: "rv-actions" }, latexCopy, copyButton("Text", dist.closed_form))),
      math, facts),
    expectationRow ? h("div", { class: "rv-card" }, expectationRow) : null,
    marginals ? h("div", { class: "rv-card" }, marginals) : null);
}

// ---- loops -------------------------------------------------------------------------------

function yesNo(value, yes = "yes", no = "no") {
  if (value === true) return yes;
  if (value === false) return no;
  return "undecided";
}

/** The attempts of a language model (strategy ai_invariant): candidate, verdict, full answer. */
function aiAttempts(attempts) {
  return attempts.map((a) => h("div", { class: "rv-cert-block" },
    h("dt", {}, `Attempt ${a.attempt}`),
    h("dd", {},
      h("div", {}, a.candidate ? h("code", { class: "rv-wrap" }, a.candidate) : "No candidate in the answer",
        a.stop_reason && !["end_turn", "stop"].includes(a.stop_reason) ? ` · stopped: ${a.stop_reason}` : ""),
      a.reason ? h("div", {}, a.reason) : null,
      h("details", {}, h("summary", {}, "Answer of the model"), h("pre", { class: "rv-pre" }, a.answer || "(no text)")))));
}

function certificateView(cert) {
  const rows = [];
  const add = (label, value) => { if (value !== null && value !== undefined && value !== "") rows.push(h("div", {}, h("dt", {}, label), h("dd", {}, value))); };
  if (cert.ai) {
    const rejected = cert.ai.rejected?.length ? `, after ${cert.ai.rejected.length} rejected` : "";
    add("Candidate from", `${cert.ai.model} (attempt ${cert.ai.attempts}${rejected}), checked by Prodigy`);
    if (cert.ai.rejected?.length) rows.push(...aiAttempts(cert.ai.rejected));
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
      add("Verdict", cert.verdict);
      if (cert.constraints?.length) add("Constraints", h("code", {}, JSON.stringify(cert.constraints)));
      if (cert.counterexample) add("Counterexample", h("code", {}, JSON.stringify(cert.counterexample)));
      rows.push(h("div", { class: "rv-cert-block" }, h("dt", {}, "Invariant I"), h("dd", {}, h("pre", { class: "rv-pre" }, (cert.invariant || "").trim()))));
      break;
    case "evt_invariant":
    case "evt_synthesis":
      add("Equation", h("code", {}, cert.equation));
      add("Invariant I", h("code", { class: "rv-wrap" }, cert.invariant));
      add("Non-negative coefficients", yesNo(cert.nonnegative, "yes", "no"));
      add("Finite mass", cert.finite_mass === true ? `yes, |I| = ${cert.mass}` : yesNo(cert.finite_mass, "yes", "no — may be a spurious fixed point"));
      if (cert.candidates?.length > 1) {
        rows.push(h("div", { class: "rv-cert-block" }, h("dt", {}, "Candidates"), h("dd", {},
          h("div", { class: "table-wrap" }, h("table", { class: "rv-table" },
            h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Invariant"), h("th", {}, "≥ 0"), h("th", {}, "Finite mass"))),
            h("tbody", {}, ...cert.candidates.map((c) => h("tr", { class: c.index === cert.chosen ? "chosen" : null },
              h("td", { class: "num" }, String(c.index)), h("td", { class: "mono rv-wrap" }, c.invariant),
              h("td", {}, yesNo(c.nonnegative)), h("td", {}, yesNo(c.finite_mass))))))))));
      }
      break;
    case "evt_subinvariant":
      add("Rule", "Lower induction: ν ≤ Φ(ν) and |ν| < ∞, so [¬guard]·ν lies below the output");
      add("Sub-invariant ν", h("code", { class: "rv-wrap" }, cert.subinvariant));
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
          h("dd", {}, h("pre", { class: "rv-pre" }, c.generating_function || ""))));
      }
      break;
    }
    case "ai_attempts":
      add("Model", cert.model);
      rows.push(...aiAttempts(cert.attempts || []));
      break;
    default:
      add("Details", h("code", { class: "rv-wrap" }, JSON.stringify(cert)));
  }
  return h("dl", { class: "rv-cert" }, ...rows);
}

function loopsSection(result, options) {
  const loops = result.loops || [];
  if (!loops.length) return null;
  const titles = options.strategyTitles || {};
  return section("Loops", h("div", { class: "rv-loops" }, ...loops.map((loop) => {
    const where = loop.line ? `line ${loop.line}` : "";
    const name = loop.id !== null ? `Loop ${loop.id}` : "Loop";
    const kind = loop.failed ? "error" : loop.kind;
    const header = h("div", { class: "rv-loop-head" },
      h("div", {}, h("span", { class: "rv-loop-name" }, name), " ",
        where && options.onSelectLine ? h("button", { type: "button", class: "link-button rv-loop-line", onclick: () => options.onSelectLine(loop.line) }, where)
          : h("span", { class: "rv-loop-line" }, where),
        h("div", { class: "rv-loop-strategy" }, titles[loop.strategy] || loop.strategy,
          loop.invocations > 1 ? ` · entered ${loop.invocations}×` : "",
          loop.inside_checks ? ` · ${loop.inside_checks === loop.invocations ? "only" : `${loop.inside_checks}×`} inside another loop's check` : "",
          loop.time ? ` · ${formatSeconds(loop.time)}` : "")),
      loop.failed ? h("span", { class: "rv-badge", "data-kind": "error" }, h("span", { class: "glyph" }, "×"), "Failed") : guaranteeBadge(kind));
    const messages = (loop.messages || []).map((m) => h("p", { class: "rv-loop-message" }, m));
    const certs = loop.certificates || [];
    const details = certs.length ? h("details", { class: "disclosure" },
      h("summary", {}, certs.length > 1 ? `Certificates (${certs.length})` : "Certificate"),
      h("div", { class: "disclosure-body" }, ...certs.slice(0, 5).map(certificateView))) : null;
    return h("article", { class: "rv-loop", "data-kind": kind }, header,
      loop.failed ? h("p", { class: "rv-loop-message rv-error-text" }, loop.failed) : null, ...messages, details);
  })));
}

// ---- steps, warnings, log ------------------------------------------------------------------

function stepsSection(result) {
  if (!result.steps?.length) return null;
  return section("Steps", h("div", { class: "table-wrap" }, h("table", { class: "rv-table rv-steps" },
    h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "After"), h("th", {}, "Distribution"), h("th", {}, "Mass"))),
    h("tbody", {}, ...result.steps.map((step) => h("tr", {},
      h("td", { class: "num" }, String(step.index + 1)),
      h("td", { class: "mono rv-wrap" }, step.instruction.trim().split("\n")[0]),
      h("td", { class: "mono rv-wrap" }, step.distribution),
      h("td", { class: "num" }, formatValue(step.mass, step.mass_float))))))));
}

function extrasSection(result) {
  const children = [];
  if (result.warnings?.length) {
    children.push(...result.warnings.map((w) => h("p", { class: "notice warning" }, icon("warning"), h("span", {}, w))));
  }
  if (result.log?.trim()) {
    children.push(h("details", { class: "disclosure" }, h("summary", {}, "Console output"),
      h("div", { class: "disclosure-body" }, h("pre", { class: "rv-pre rv-log" }, cleanLog(result.log)))));
  }
  const json = JSON.stringify(result, null, 2);
  const download = h("a", { class: "button small", download: "prodigy-result.json",
    href: URL.createObjectURL(new Blob([json], { type: "application/json" })) }, icon("download"), "Result as JSON");
  children.push(h("div", { class: "rv-actions" }, download));
  return h("section", { class: "rv-section rv-extras" }, ...children);
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
 * options: { onSelectLoop(id), onSelectLine(line), strategyTitles: {name: title}, stale: bool, elapsed }
 */
export function renderResult(container, result, options = {}) {
  const children = [];
  if (options.stale) {
    children.push(h("p", { class: "notice" }, icon("info"),
      h("span", {}, "The program or setup changed since this result. Analyze again to update it.")));
  }
  if (result.status !== "ok") {
    children.push(errorView(result, options));
  } else {
    children.push(summary(result, options));
  }
  const queries = result.queries || [];
  if (queries.length) {
    children.push(section("Queries", h("ol", { class: "rv-queries" }, ...queries.map((q) => queryRow(q, result, options)))));
  }
  if (result.status === "ok") {
    children.push(distributionSection(result));
  }
  children.push(loopsSection(result, options), stepsSection(result), extrasSection(result));
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
      h("div", { class: "rv-progress-label" }, h("span", {}, label),
        h("span", { class: "num" }, fraction !== null ? `${Math.round(fraction * 100)} %` : "")),
      fraction !== null
        ? h("progress", { max: "1", value: String(fraction), "aria-label": label })
        : h("progress", { "aria-label": label }));
  });
  const log = cleanLog(state.log || "").split("\n").slice(-12).join("\n");
  container.replaceChildren(h("div", { class: "result-view" },
    h("div", { class: "rv-running" },
      h("div", { class: "rv-running-head" }, h("span", { class: "spinner", "aria-hidden": "true" }),
        h("span", {}, `Analyzing · ${formatSeconds(state.elapsed || 0)}`),
        state.timeLimit ? h("span", { class: "rv-muted" }, `limit ${formatSeconds(state.timeLimit)}`) : null),
      ...bars,
      log ? h("pre", { class: "rv-pre rv-log", "aria-label": "Console output" }, log) : null)));
}

export { formatValue, formatNumber };
