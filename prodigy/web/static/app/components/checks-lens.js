// The lens Checks of the analysis card (analysis-card.js): the loops of the result, their certificates and
// the checks that proved them (or did not), including the attempts of a language model.
import { h } from "./dom.js";
import { BOUND_TEXT, formatSeconds, formatValue } from "./format.js";
import { facts, where } from "./lens-parts.js";
import { expressionBox, guaranteeBadge } from "./parts.js";

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
      if (cert.counterexample) add(cert.witness ? "Input of the witness" : "Candidate counterexample", h("code", {}, JSON.stringify(cert.counterexample)));
      if (cert.witness) add("Witness", h("code", {}, JSON.stringify(cert.witness.state)));
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
    loop.failed ? failedBadge(loop) : (loop.time ? h("span", { class: "rv-meta", title: traceLoop ? "Including the time to record the snapshots of its iterates; the Time lens leaves that out" : null },
      traceLoop ? `${formatSeconds(loop.time)} with recording` : formatSeconds(loop.time)) : null));
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

export function checksLens(ctx) {
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
