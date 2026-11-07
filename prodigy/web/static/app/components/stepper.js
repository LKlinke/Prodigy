// Stepper: navigates the step trace of an analysis result (prodigy/analysis/trace.py).
//
// Like the result view it needs only the result JSON and CSS custom properties, so other front
// ends (the VS Code integration) can reuse it. The editor is reached through callbacks.
import { h, s } from "./dom.js";
import { hatchPattern, niceMax } from "./charts.js";
import { BOUND_TEXT, formatNumber, formatSeconds, formatValue, qualify } from "./format.js";
import { renderMath } from "./math.js";
import { distributionFigure, guaranteeBadge } from "./result-view.js";

const KIND_LABELS = {
  assign: "Assignment", while: "Loop", if: "Conditional", choice: "Probabilistic choice", observe: "Observation",
  loop: "Repetition", query_block: "Query block", skip: "Skip", abort: "Abort", query: "Query",
  normalize: "Conditioning",
};

const BRANCH_TEXT = {
  then: "the then-branch: the part of the distribution where the guard holds",
  else: "the else-branch: the part of the distribution where the guard fails",
  left: "the left branch of the choice: the distribution weighted by its probability",
  right: "the right branch of the choice: the distribution weighted by its probability",
  block: "a query block",
};

const CHECK_GLYPHS = { passed: "✓", failed: "×", unknown: "?", info: "i" };
const CHECK_LABELS = { passed: "Proven", failed: "Failed", unknown: "Not established", info: "Note" };

const reducedMotion = () => Boolean(globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

function where(step) {
  if (step.kind === "normalize") return "end of the program";
  if (!step.line) return "generated";
  return `line ${step.line}`;
}

/** Text for the guarantee of one distribution of the trace. */
export function guaranteeText(snap) {
  switch (snap?.bound) {
    case "exact": return "Exact at this point.";
    case "lower": return `Lower bound: at most ${formatValue(snap.residual_mass, snap.residual_mass_float)} of the mass is not yet accounted for; it may still end up anywhere, or never terminate.`;
    case "upper": return "Upper bound (Park's lemma): exact only if the loops so far terminate almost surely, which is not checked.";
    default: return "No guarantee: approximations of different kinds were combined, or a side condition could not be checked.";
  }
}

function boundsOf(snap) {
  return { bounds: { kind: snap.bound || "unknown", residual_mass: snap.residual_mass, residual_mass_float: snap.residual_mass_float } };
}

// ---- one distribution --------------------------------------------------------------------

function distributionView(snap, label, state) {
  if (!snap) return h("p", { class: "rv-muted" }, "No distribution: the instruction failed.");
  const math = h("div", { class: "rv-closed-form st-closed-form", tabindex: "0", "aria-label": `${label}: generating function ${snap.closed_form}` });
  if (snap.truncated) {
    math.textContent = `${snap.closed_form} …`;
    math.classList.add("math-fallback");
  } else {
    renderMath(math, snap.latex, snap.closed_form);
  }
  const facts = h("dl", { class: "rv-facts" });
  const fact = (name, value, title) => facts.append(h("div", {}, h("dt", {}, name), h("dd", { title: title || "" }, value)));
  fact("Mass", formatValue(snap.mass, snap.mass_float, 6), snap.mass);
  if (snap.error_probability && snap.error_probability !== "0") {
    fact("Observation violated", formatValue(snap.error_probability, snap.error_probability_float, 6), snap.error_probability);
  }
  fact("Size", `${snap.size.toLocaleString("en-US")} character${snap.size === 1 ? "" : "s"}`, "Length of the closed form; the same measure on every engine");
  const variables = Object.keys(snap.marginals || {});
  let marginals = null;
  if (variables.length) {
    const holder = h("div", {});
    const picker = h("div", { class: "segmented small", role: "group", "aria-label": "Variable" });
    const choose = (name) => {
      state.variable = name;
      const table = snap.marginals[name];
      holder.replaceChildren(table.error ? h("p", { class: "rv-muted" }, `Not available: ${table.error}`)
        : distributionFigure(table, boundsOf(snap), name));
      for (const button of picker.children) button.setAttribute("aria-pressed", String(button.dataset.variable === name));
    };
    for (const name of variables) {
      picker.append(h("button", { type: "button", "data-variable": name, "aria-pressed": "false", onclick: () => choose(name) }, name));
    }
    choose(variables.includes(state.variable) ? state.variable : variables[0]);
    marginals = h("div", { class: "rv-marginals" },
      h("div", { class: "rv-subhead" }, h("h4", {}, "Marginal distributions"), variables.length > 1 ? picker : null), holder);
  } else if (snap.marginals_omitted) {
    marginals = h("p", { class: "rv-note" }, `Marginals not recorded: ${snap.marginals_omitted}.`);
  }
  return h("div", { class: "st-distribution" },
    h("div", { class: "st-guarantee" }, guaranteeBadge(snap.bound || "unknown"), h("span", {}, guaranteeText(snap))),
    math, facts, marginals);
}

// ---- loop: checks and animated iterates ------------------------------------------------------

function checksList(checks) {
  if (!checks?.length) return null;
  return h("ul", { class: "st-checks", "aria-label": "Checks of the certificate" }, ...checks.map((check) =>
    h("li", { class: "st-check", "data-status": check.status },
      h("span", { class: "st-check-glyph", "aria-hidden": "true" }, CHECK_GLYPHS[check.status] || "i"),
      h("span", { class: "st-check-body" },
        h("span", { class: "st-check-label" }, `${check.label} · ${CHECK_LABELS[check.status] || check.status}`),
        h("span", { class: "st-check-text" }, check.text)))));
}

/** The bars of every iterate on one fixed scale, so that the growth of the terminated part and the
 * shrinking residual ("?") are visible. Returns {svg, show(index)}. */
function iterateChart(iterates, variable, width) {
  const tables = iterates.map((it) => it.distribution.marginals?.[variable]);
  const MAX_BARS = 12;
  const values = [...new Set(tables.flatMap((t) => (t?.rows || []).map((r) => r.value)))].sort((a, b) => a - b).slice(0, MAX_BARS);
  const shown = new Set(values);
  // the mass beyond the shown values: the table's tail plus rows that do not get a bar of their own
  const tailOf = (table) => {
    if (!table) return 0;
    const rest = (table.rows || []).filter((r) => !shown.has(r.value)).reduce((sum, r) => sum + (r.p || 0), 0);
    return rest + (table.truncated ? table.tail_mass_float || 0 : 0);
  };
  const hasTail = tables.some((t) => tailOf(t) > 1e-12);
  const hasResidual = iterates.some((it) => it.distribution.residual_mass_float > 0);
  const bars = [...values.map((v) => ({ kind: "value", key: v, label: String(v) }))];
  if (hasTail) bars.push({ kind: "tail", label: `≥${values.length ? values[values.length - 1] + 1 : 0}` });
  if (hasResidual) bars.push({ kind: "residual", label: "?" });
  const heights = iterates.map((it, index) => {
    const table = tables[index];
    const byValue = new Map((table?.rows || []).map((r) => [r.value, r.p ?? 0]));
    return bars.map((bar) => {
      if (bar.kind === "value") return byValue.get(bar.key) ?? 0;
      if (bar.kind === "tail") return tailOf(table);
      return it.distribution.residual_mass_float || 0;
    });
  });
  const max = niceMax(Math.max(1e-9, ...heights.flat()));
  const W = Math.max(260, Math.round(width));
  const H = 168;
  const top = 14;
  const bottom = 22;
  const right = 40;
  const plotW = W - 4 - right;
  const plotH = H - top - bottom;
  const slot = plotW / Math.max(1, bars.length);
  const barW = Math.max(2, Math.min(36, slot * 0.72));
  const id = `it-hatch-${Math.random().toString(36).slice(2)}`;
  const y = (p) => top + plotH * (1 - p / max);
  const rects = [];
  const titles = [];
  const marks = bars.map((bar, index) => {
    const cx = 4 + slot * index + slot / 2;
    const rect = s("rect", { x: cx - barW / 2, y: top, width: barW, height: plotH, rx: Math.min(3, barW / 4),
      fill: bar.kind === "residual" ? `url(#${id})` : null, class: "it-rect" });
    const title = s("title", {});
    rects.push(rect);
    titles.push(title);
    const labelEvery = Math.max(1, Math.ceil(bars.length / Math.max(1, Math.floor(plotW / 26))));
    const label = bar.kind !== "value" || index % labelEvery === 0;
    return s("g", { class: `mc-bar mc-${bar.kind}` }, title, rect,
      label ? s("text", { x: cx, y: H - 6, class: "mc-label" }, bar.label) : null);
  });
  const grid = [0, max / 2, max].map((tick) => s("g", { class: "mc-grid" },
    s("line", { x1: 4, x2: 4 + plotW, y1: y(tick), y2: y(tick) }),
    s("text", { x: 4 + plotW + 6, y: y(tick) + 4 }, formatNumber(tick, 2))));
  const svg = s("svg", { class: "marginal-chart iterate-chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" },
    s("defs", {}, hatchPattern(id)), ...grid, ...marks,
    s("line", { class: "mc-axis", x1: 4, x2: 4 + plotW, y1: top + plotH, y2: top + plotH }));
  const show = (index) => {
    heights[index].forEach((p, i) => {
      rects[i].style.transform = `scaleY(${Math.max(0, Math.min(1, p / max))})`;
      const bar = bars[i];
      const what = bar.kind === "residual" ? "Not yet accounted for (still inside the loop)" : bar.kind === "tail" ? `${variable} ${bar.label}` : `${variable} = ${bar.label}`;
      titles[i].textContent = `${what}: ${formatNumber(p, 4)}`;
    });
    const it = iterates[index];
    svg.setAttribute("aria-label", `Iterate ${it.iteration}: distribution of ${variable} on the part that has left the loop`
      + (hasResidual ? `, and ${formatNumber(it.distribution.residual_mass_float || 0, 3)} not yet accounted for` : "") + ".");
  };
  return { svg, show };
}

function iteratePlayer(loop, state, width) {
  const iterates = loop.iterates || [];
  if (!iterates.length) return null;
  const variables = [...new Set(iterates.flatMap((it) => Object.keys(it.distribution.marginals || {})))];
  const holder = h("div", { class: "st-player-chart" });
  const status = h("p", { class: "st-player-status num", "aria-live": "polite" });
  const massBar = h("div", { class: "st-massbar", role: "img" }, h("span", { class: "st-massbar-out" }), h("span", { class: "st-massbar-in" }));
  const slider = h("input", { type: "range", class: "st-slider", min: "0", max: String(iterates.length - 1), step: "1",
    "aria-label": "Iterate" });
  const play = h("button", { type: "button", class: "button small st-play", "aria-pressed": "false" });
  let chart = null;
  let timer = null;
  let index = Math.min(state.iterate ?? iterates.length - 1, iterates.length - 1);
  const pace = () => (reducedMotion() ? 1000 : 550);
  const setPlayLabel = (playing) => {
    play.replaceChildren(h("span", { "aria-hidden": "true" }, playing ? "❚❚" : "▶"), playing ? " Pause" : " Play");
    play.setAttribute("aria-pressed", String(playing));
    play.title = playing ? "Pause the animation (Space)" : "Play the iterates from the first one (Space)";
    status.setAttribute("aria-live", playing ? "off" : "polite");
  };
  const draw = () => {
    const it = iterates[index];
    const snap = it.distribution;
    slider.value = String(index);
    const total = (snap.mass_float ?? 0) + (it.inside_mass_float ?? 0);
    const out = total > 0 ? (snap.mass_float ?? 0) / total : 0;
    massBar.firstChild.style.width = `${(out * 100).toFixed(2)}%`;
    massBar.lastChild.style.width = `${((1 - out) * 100).toFixed(2)}%`;
    const inside = formatValue(it.inside_mass, it.inside_mass_float);
    const left = formatValue(snap.mass, snap.mass_float);
    massBar.setAttribute("aria-label", `Left the loop: ${left}; still inside: ${inside}`);
    slider.setAttribute("aria-valuetext", `Iteration ${it.iteration}: left the loop ${left}, still inside ${inside}`);
    status.replaceChildren(
      h("span", { class: "st-player-iteration" }, `Iteration ${it.iteration} of ${loop.iterations}`),
      h("span", {}, h("span", { class: "st-key st-key-out", "aria-hidden": "true" }), `left the loop ${left}`),
      h("span", {}, h("span", { class: "st-key st-key-in", "aria-hidden": "true" }), `still inside ${inside}`),
      h("span", { class: "rv-badge", "data-kind": snap.bound }, h("span", { class: "glyph", "aria-hidden": "true" },
        (BOUND_TEXT[snap.bound] || BOUND_TEXT.unknown).glyph), (BOUND_TEXT[snap.bound] || BOUND_TEXT.unknown).label));
    chart?.show(index);
    state.iterate = index;
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    setPlayLabel(false);
  };
  const start = () => {
    if (index >= iterates.length - 1) index = 0;
    draw();
    setPlayLabel(true);
    timer = setInterval(() => {
      if (!holder.isConnected) { stop(); return; }
      if (index >= iterates.length - 1) { stop(); return; }
      index += 1;
      draw();
    }, pace());
  };
  play.addEventListener("click", () => (timer ? stop() : start()));
  slider.addEventListener("input", () => { stop(); index = Number(slider.value); draw(); });
  const picker = h("div", { class: "segmented small", role: "group", "aria-label": "Variable of the chart" });
  const choose = (name) => {
    state.iterateVariable = name;
    chart = iterateChart(iterates, name, width);
    holder.replaceChildren(chart.svg);
    for (const button of picker.children) button.setAttribute("aria-pressed", String(button.dataset.variable === name));
    draw();
  };
  for (const name of variables) {
    picker.append(h("button", { type: "button", "data-variable": name, "aria-pressed": "false", onclick: () => choose(name) }, name));
  }
  setPlayLabel(false);
  const element = h("div", { class: "st-player" },
    h("div", { class: "rv-subhead" }, h("h4", {}, "Kleene iterates"), variables.length > 1 ? picker : null),
    h("p", { class: "rv-note" },
      `The part that has left the loop after each iteration grows towards the loop's output; the hatched bar “?” is the mass not yet accounted for, which shrinks.`
      + (loop.stride > 1 ? ` Every ${loop.stride}th iteration is shown, and the last one.` : "")),
    holder,
    massBar,
    h("div", { class: "st-player-controls" }, play, slider),
    status);
  element.addEventListener("keydown", (event) => {
    if (event.key === " " && event.target !== play) { event.preventDefault(); play.click(); }
  });
  if (variables.length) {
    choose(variables.includes(state.iterateVariable) ? state.iterateVariable : variables[0]);
  } else {
    holder.replaceChildren(h("p", { class: "rv-muted" }, "No marginals recorded for the iterates."));
    draw();
  }
  return { element, stop };
}

function loopView(step, options, state, width) {
  const loop = step.loop;
  if (!loop) return null;
  const titles = options.strategyTitles || {};
  const badge = loop.failed
    ? h("span", { class: "rv-badge", "data-kind": loop.checks?.some((c) => c.status === "unknown") && !loop.checks?.some((c) => c.status === "failed") ? "unknown" : "error" },
      h("span", { class: "glyph", "aria-hidden": "true" }, "×"), "Failed")
    : guaranteeBadge(loop.kind || "unknown");
  const player = iteratePlayer(loop, state, width);
  state.stopPlayer = player?.stop;
  const params = Object.entries(loop.params || {}).filter(([, v]) => v !== null && v !== "")
    .map(([k, v]) => `${k} = ${String(v).length > 40 ? `${String(v).slice(0, 39)}…` : v}`).join(", ");
  return h("section", { class: "rv-section" },
    h("h3", { class: "rv-section-title" }, "Loop"),
    h("div", { class: "rv-card st-loop" },
      h("div", { class: "rv-loop-head" },
        h("div", {}, h("span", { class: "rv-loop-name" }, loop.id !== null && loop.id !== undefined ? `Loop ${loop.id}` : "Loop"),
          h("div", { class: "rv-loop-strategy" }, titles[loop.strategy] || loop.title || loop.strategy, params ? ` · ${params}` : "")),
        badge),
      loop.failed ? h("p", { class: "rv-loop-message rv-error-text" }, loop.failed) : null,
      ...(loop.messages || []).map((m) => h("p", { class: "rv-loop-message" }, m)),
      checksList(loop.checks),
      player?.element || (loop.strategy && !loop.iterates?.length && !loop.failed
        ? h("p", { class: "rv-note" }, "This strategy does not iterate; the trace shows the distribution before and after the loop and the checks of its certificate.")
        : null)));
}

// ---- watches and hints -------------------------------------------------------------------------

function watchesView(step, result, options) {
  const watches = result.trace?.watches || [];
  const values = step.watches || [];
  const field = h("input", { type: "text", class: "field mono", placeholder: result.syntax === "ppy" ? "Pr[x == 3] or Ex[x]" : "Pr[x = 3] or Ex[x]",
    "aria-label": "Query to evaluate after every step", autocomplete: "off", spellcheck: "false" });
  const add = h("button", { type: "submit", class: "button small" }, "Add and analyze again");
  const form = h("form", { class: "field-with-action st-watch-form" }, field, add);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = field.value.trim();
    if (text) options.onAddWatch?.(text);
  });
  const rows = watches.map((watch, i) => {
    const value = values[i];
    let shown;
    if (step.depth > 0) shown = h("span", { class: "rv-muted" }, "evaluated after top-level steps only");
    else if (!value) shown = h("span", { class: "rv-muted" }, "—");
    else {
      const { glyph, text } = qualify(value, { withValue: true });
      shown = h("span", { class: "rv-q-bound", "data-kind": value.bound },
        glyph ? h("span", { class: "glyph", "aria-hidden": "true" }, glyph) : null, text,
        value.note ? h("span", { class: "rv-note" }, ` ${value.note}`) : null);
    }
    return h("li", { class: "st-watch" }, h("code", {}, watch.text), shown,
      h("button", { type: "button", class: "link-button", "aria-label": `Remove ${watch.text} and analyze again`,
        onclick: () => options.onRemoveWatch?.(watch.text) }, "Remove"));
  });
  return h("section", { class: "rv-section" },
    h("h3", { class: "rv-section-title" }, "Queries at this step"),
    h("div", { class: "rv-card" },
      rows.length ? h("ul", { class: "st-watches" }, ...rows)
        : h("p", { class: "rv-note" }, "Add a query such as Pr[x = 3]; it is evaluated after every top-level step, as if it stood right after that instruction, with the same sound intervals as the program's queries."),
      form));
}

function hintsView(result, go) {
  const hints = result.trace?.hints || [];
  if (!hints.length) return null;
  return h("section", { class: "rv-section" }, h("h3", { class: "rv-section-title" }, "Diagnosis"),
    ...hints.map((hint) => h("p", { class: "notice" }, h("span", { class: "st-hint-glyph", "aria-hidden": "true" },
      hint.kind === "time" ? "⏱" : hint.kind === "growth" ? "↗" : "i"),
    h("span", {}, hint.text, " ", hint.step !== null && hint.step !== undefined
      ? h("button", { type: "button", class: "link-button", onclick: () => go(hint.step) }, "Show this step") : null))));
}

// ---- the list of steps -------------------------------------------------------------------------

function stepList(steps, current, breakpoints, go) {
  const list = h("ol", { class: "st-list", role: "listbox", "aria-label": "Steps", tabindex: "0", "data-focus-key": "list",
    "aria-activedescendant": `st-step-${current}` });
  for (const step of steps) {
    const kind = step.error ? "error" : step.bound || "unknown";
    const text = (BOUND_TEXT[kind] || { glyph: "×", label: "Failed" });
    const hasBreak = step.line && breakpoints.has(step.line);
    list.append(h("li", { id: `st-step-${step.index}`, role: "option", class: "st-item", "aria-selected": String(step.index === current),
      "data-depth": String(step.depth), onclick: () => go(step.index) },
    h("span", { class: "st-item-break", "aria-hidden": "true" }, hasBreak ? "●" : ""),
    h("span", { class: "st-item-line num" }, step.line ? String(step.line) : "–"),
    h("span", { class: "st-item-text mono", style: `padding-left:${step.depth * 14}px` },
      step.branch && step.branch !== "block" ? h("span", { class: "st-item-branch" }, `${step.branch} `) : null, step.summary || step.instruction),
    h("span", { class: "st-item-kind", "data-kind": kind, title: step.error ? "Failed" : text.label },
      h("span", { "aria-hidden": "true" }, step.error ? "×" : text.glyph), h("span", { class: "visually-hidden" }, step.error ? "Failed" : text.label)),
    h("span", { class: "st-item-time num" }, formatSeconds(step.time)),
    hasBreak ? h("span", { class: "visually-hidden" }, ", breakpoint") : null));
  }
  list.addEventListener("keydown", (event) => {
    const moves = { ArrowDown: 1, ArrowUp: -1, Home: -Infinity, End: Infinity, PageDown: 10, PageUp: -10 };
    if (!(event.key in moves)) return;
    event.preventDefault();
    const target = Math.max(0, Math.min(steps.length - 1, current + moves[event.key]));
    go(target, { focusList: true });
  });
  // keep the current step visible inside the list without scrolling the panel
  requestAnimationFrame(() => {
    const item = list.querySelector('[aria-selected="true"]');
    if (!item) return;
    const top = item.offsetTop - list.offsetTop;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (top + item.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = top + item.offsetHeight - list.clientHeight;
  });
  return list;
}

// ---- public API ----------------------------------------------------------------------------------

/** Index of the next (dir 1) or previous (dir -1) step on a line with a breakpoint, else the last (first) step. */
export function breakpointTarget(steps, current, breakpoints, dir = 1) {
  for (let i = current + dir; i >= 0 && i < steps.length; i += dir) {
    if (steps[i].line && breakpoints.has(steps[i].line)) return i;
  }
  return dir > 0 ? steps.length - 1 : 0;
}

/**
 * Renders the stepper for `result` into `container`.
 * options: { index, breakpoints: [line], onIndex(i), onSelectLine(line), onAddWatch(text),
 *   onRemoveWatch(text), onRecord(), strategyTitles, stale, focusList }
 * Keeps per-view state (variable, before/after, iterate) in options.viewState.
 */
export function renderStepper(container, result, options = {}) {
  const view = options.viewState || {};
  view.stopPlayer?.();
  const steps = result?.steps || [];
  if (!steps.length) {
    const text = result
      ? "This result has no recorded steps."
      : "No analysis yet.";
    container.replaceChildren(h("div", { class: "result-view stepper" },
      h("div", { class: "st-empty" },
        h("p", {}, text, " The stepper shows the distribution after each instruction, with its line in the editor, the iterates of loops and why a certificate failed."),
        h("p", {}, h("button", { type: "button", class: "button prominent", onclick: () => options.onRecord?.() }, "Record steps and analyze")),
        h("p", { class: "rv-note" }, "Recording keeps each step's distribution (closed form, marginals, mass), which takes some extra time. Option: Setup → Options → Record each step."))));
    return;
  }
  const index = Math.max(0, Math.min(steps.length - 1, options.index ?? 0));
  const step = steps[index];
  const breakpoints = new Set(options.breakpoints || []);
  const go = (i, extra = {}) => options.onIndex?.(Math.max(0, Math.min(steps.length - 1, i)), extra);

  const nav = h("div", { class: "st-nav", role: "toolbar", "aria-label": "Step through the analysis" },
    h("button", { type: "button", class: "button small", "data-focus-key": "first", onclick: () => go(0), disabled: index === 0, title: "First step (Home)" }, "⇤ First"),
    h("button", { type: "button", class: "button small", "data-focus-key": "prev-break", onclick: () => go(breakpointTarget(steps, index, breakpoints, -1)), disabled: index === 0,
      title: "Back to the previous breakpoint (⇧F5)" }, "⟸ Breakpoint"),
    h("button", { type: "button", class: "button small", "data-focus-key": "back", onclick: () => go(index - 1), disabled: index === 0, title: "Previous step (⇧F10 or ←)" }, "← Back"),
    h("button", { type: "button", class: "button small prominent", "data-focus-key": "next", onclick: () => go(index + 1), disabled: index === steps.length - 1,
      title: "Next step (F10 or →)" }, "Next →"),
    h("button", { type: "button", class: "button small", "data-focus-key": "next-break", onclick: () => go(breakpointTarget(steps, index, breakpoints, 1)),
      disabled: index === steps.length - 1, title: "Continue to the next breakpoint (F5)" }, "Breakpoint ⟹"),
    h("button", { type: "button", class: "button small", "data-focus-key": "last", onclick: () => go(steps.length - 1), disabled: index === steps.length - 1, title: "Last step (End)" }, "Last ⇥"));

  const lines = [...new Set(steps.filter((s) => s.line).map((s) => s.line))].sort((a, b) => a - b);
  const jump = h("select", { class: "st-jump", "aria-label": "Jump to the first step on a line" },
    h("option", { value: "" }, "Jump to line…"),
    ...lines.map((line) => h("option", { value: String(line) }, `Line ${line}${breakpoints.has(line) ? " ●" : ""}`)));
  jump.addEventListener("change", () => {
    const line = Number(jump.value);
    const after = steps.findIndex((s, i) => i > index && s.line === line);
    go(after >= 0 ? after : steps.findIndex((s) => s.line === line));
  });

  const head = h("div", { class: "st-head" },
    h("div", { class: "st-position", "aria-live": "polite" },
      h("span", { class: "st-count num" }, `Step ${index + 1} of ${steps.length}`), " · ",
      step.line && options.onSelectLine
        ? h("button", { type: "button", class: "link-button", onclick: () => options.onSelectLine(step.line) }, where(step))
        : h("span", {}, where(step))),
    h("div", { class: "st-tools" }, jump,
      h("span", { class: "rv-muted st-breakpoints" }, breakpoints.size
        ? `Breakpoints: line ${[...breakpoints].sort((a, b) => a - b).join(", ")}`
        : "No breakpoints · click a line number or press F9 in the editor")));

  const instruction = h("div", { class: "rv-card st-instruction" },
    h("div", { class: "rv-card-head" }, h("span", { class: "rv-muted" }, KIND_LABELS[step.kind] || step.kind),
      h("span", { class: "rv-muted num" }, [formatSeconds(step.time),
        step.self_time !== undefined && step.self_time !== step.time ? `${formatSeconds(step.self_time)} without nested steps` : null,
        step.size_after !== null && step.size_after !== undefined ? `size ${step.size_before} → ${step.size_after}` : null].filter(Boolean).join(" · "))),
    h("pre", { class: "rv-pre st-code" }, step.instruction.trim()),
    step.depth > 0 && BRANCH_TEXT[step.branch]
      ? h("p", { class: "rv-note" }, `Inside ${BRANCH_TEXT[step.branch]} (not normalised).`) : null,
    step.kind === "normalize" ? h("p", { class: "rv-note" }, "The distribution is conditioned on the observations: divided by the probability that none was violated.") : null,
    step.error ? h("p", { class: "rv-loop-message rv-error-text", role: "alert" }, `Failed: ${step.error}`) : null);

  const side = view.side === "before" || !step.after ? "before" : "after";
  const toggle = h("div", { class: "segmented small", role: "group", "aria-label": "Distribution before or after the step" },
    ...["before", "after"].map((which) => h("button", { type: "button", "data-focus-key": which, "aria-pressed": String(which === side), disabled: which === "after" && !step.after,
      onclick: () => { view.side = which; renderStepper(container, result, options); } }, which === "before" ? "Before" : "After")));
  const width = Math.max(300, (container.clientWidth || 560) - 64);
  const distribution = h("section", { class: "rv-section" },
    h("div", { class: "rv-subhead" }, h("h3", { class: "rv-section-title" }, side === "before" ? "Distribution before the step" : "Distribution after the step"), toggle),
    h("div", { class: "rv-card" }, distributionView(side === "before" ? step.before : step.after, side, view)));

  const children = [
    options.stale ? h("p", { class: "notice" }, h("span", {}, "The program or setup changed since these steps were recorded. Analyze again to update them.")) : null,
    head, nav, instruction, distribution, loopView(step, options, view, width), watchesView(step, result, options),
    hintsView(result, go),
    h("section", { class: "rv-section" }, h("h3", { class: "rv-section-title" }, "All steps"),
      h("div", { class: "rv-card st-list-card" }, stepList(steps, index, breakpoints, go))),
  ];
  // keep the keyboard focus on the same control across the re-rendering (e.g. on Next)
  const active = document.activeElement;
  const focusKey = active && container.contains(active) ? active.closest("[data-focus-key]")?.dataset.focusKey : null;
  container.replaceChildren(h("div", { class: "result-view stepper", "data-stale": options.stale ? "true" : "false" }, ...children.filter(Boolean)));
  const again = focusKey ? container.querySelector(`[data-focus-key="${focusKey}"]`) : null;
  if (options.focusList || (focusKey && (!again || again.disabled))) container.querySelector(".st-list")?.focus({ preventScroll: true });
  else again?.focus({ preventScroll: true });
}
