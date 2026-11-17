// The Setup panel: input distribution, one strategy per loop, analysis options.
import { h, icon } from "./components/dom.js";
import { defaultReason } from "./mode-setup.js";

export const GUI_DEFAULTS = { auto: { iterations: 25 }, unroll: { iterations: 25 }, evt_iterate: { iterations: 25 } };

/** The groups of the strategy pop-up, by what the user wants (the registry's `group`, #324). */
export const STRATEGY_GROUPS = [
  ["automatic", "Automatic"],
  ["exact", "Exact result"],
  ["bounds", "Bounds"],
  ["explore", "Explore"],
  ["other", "More"],
];

const GROUP_HELP = {
  automatic: "Prodigy picks: exact if it can prove it quickly, otherwise a lower bound.",
  exact: "Prove the loop's output with an invariant: exact, or an upper bound if a side condition fails.",
  bounds: "Sound bounds without proving termination: every query becomes an interval.",
  explore: "Insight into the loop for guessing an invariant; the result is only a lower bound.",
};

/** The glyph of the chip next to a strategy (the text is the registry's `outcome`). */
const OUTCOME_GLYPHS = { lower: "≥", upper: "≤", exact: "✓" };

/** The strategies of each non-empty group, in menu order; an unknown group counts as "other". */
export function groupStrategies(strategies) {
  const known = new Set(STRATEGY_GROUPS.map(([key]) => key));
  const groupOf = (s) => (known.has(s.group) ? s.group : "other");
  return STRATEGY_GROUPS.map(([key, title]) => ({ key, title, members: strategies.filter((s) => groupOf(s) === key) }))
    .filter((g) => g.members.length);
}

function outcomeChip(strategy) {
  const text = strategy.outcome || { lower: "lower bound", upper: "upper bound", exact: "exact" }[strategy.soundness];
  if (!text) return null;
  const glyph = strategy.group === "explore" ? "◇" : OUTCOME_GLYPHS[strategy.soundness] || "";
  return h("span", { class: "chip", title: `What you get: ${text}` }, h("span", { "aria-hidden": "true" }, glyph), text);
}

export function defaultParams(strategy) {
  const params = {};
  for (const param of strategy.params) {
    if (param.default !== null && param.default !== undefined && !param.optional) params[param.name] = param.default;
  }
  return { ...params, ...(GUI_DEFAULTS[strategy.name] || {}) };
}

/** The decision of a loop nobody chose a strategy for: "auto" (a short synthesis, accepted only if exact,
 * otherwise unrolling), else unrolling. */
export function defaultDecision(strategies) {
  const chosen = strategies.find((s) => s.name === "auto") || strategies.find((s) => s.name === "unroll") || strategies[0];
  return { strategy: chosen.name, params: defaultParams(chosen) };
}

/** Parameters shown right away (they must be provided) and those behind the "Parameters" disclosure
 * (they have a default or are optional). */
export function splitParams(strategy) {
  const params = strategy.params.filter((p) => p.kind !== "file");
  const hasDefault = (p) => p.optional || (p.default !== null && p.default !== undefined) || GUI_DEFAULTS[strategy.name]?.[p.name] !== undefined;
  return { required: params.filter((p) => !hasDefault(p)), more: params.filter(hasDefault) };
}

/** A loop's identity across edits: its guard and the function it is in. */
function loopSignature(loop) {
  return JSON.stringify([loop.cond, loop.function || null]);
}

/** Moves the decisions to the loops of a new outline (issue #277). Loop ids are positions, so
 * inserting or removing a loop shifts them. A decision follows the loop with the same guard (and
 * function), in order among equals; the leftover old and new loops are paired in order when there
 * are as many of each (a guard edited in place). A decision whose loop is gone is kept in
 * `graveyard` (signature to decisions) and comes back when an unclaimed loop with that guard
 * reappears, e.g. on undo. Nothing is ever attached to a loop it did not belong to.
 * Returns whether any decision changed its key or was buried or restored. */
export function rekeyDecisions(decisions, oldOutline, newOutline, graveyard = {}) {
  if (!oldOutline || !newOutline) return false;
  const oldLoops = oldOutline.loops || [];
  const newLoops = newOutline.loops || [];
  const target = new Map(); // old id -> new id
  const claimed = new Set();
  const bySignature = new Map();
  for (const loop of newLoops) {
    const list = bySignature.get(loopSignature(loop)) || [];
    list.push(loop);
    bySignature.set(loopSignature(loop), list);
  }
  const seen = new Map();
  for (const loop of oldLoops) {
    const signature = loopSignature(loop);
    const rank = seen.get(signature) || 0;
    seen.set(signature, rank + 1);
    const match = (bySignature.get(signature) || [])[rank];
    if (match) { target.set(String(loop.id), String(match.id)); claimed.add(String(match.id)); }
  }
  const lostOld = oldLoops.filter((l) => !target.has(String(l.id)));
  const freeNew = newLoops.filter((l) => !claimed.has(String(l.id)));
  if (lostOld.length === freeNew.length) {
    lostOld.forEach((l, i) => { target.set(String(l.id), String(freeNew[i].id)); claimed.add(String(freeNew[i].id)); });
  }
  let changed = false;
  const moved = {};
  for (const [key, decision] of Object.entries(decisions)) {
    if (key === "*") continue;
    delete decisions[key];
    if (target.has(key)) {
      moved[target.get(key)] = decision;
      if (target.get(key) !== key) changed = true;
    } else {
      const loop = oldLoops.find((l) => String(l.id) === key);
      if (loop) { (graveyard[loopSignature(loop)] ||= []).push(decision); changed = true; }
    }
  }
  for (const loop of newLoops) {
    const id = String(loop.id);
    if (moved[id] === undefined && !claimed.has(id)) {
      const buried = graveyard[loopSignature(loop)];
      if (buried?.length) { moved[id] = buried.shift(); changed = true; }
    }
  }
  Object.assign(decisions, moved);
  return changed;
}

/** Drops the decisions of loops the program no longer has. Without an outline (before the first
 * check after a reload, or while the program has never parsed) the loops are unknown, so every
 * decision is kept (issue #194). */
export function pruneDecisions(decisions, outline) {
  if (!outline) return;
  const known = new Set((outline.loops || []).map((l) => String(l.id)));
  for (const key of Object.keys(decisions)) {
    if (key !== "*" && !known.has(key)) delete decisions[key];
  }
}

/** Sets the invariant programs of a conversion, or of its undo or redo: {before, after} map loop ids
 * to texts. An invariant that was edited since (or whose loop now has another strategy) is left as
 * it is. Returns whether a decision changed. */
export function swapInvariants(decisions, { before, after }) {
  let changed = false;
  for (const [key, text] of Object.entries(after || {})) {
    const decision = decisions[key];
    if (decision?.strategy !== "invariant" || decision.params?.invariant !== before?.[key]) continue;
    decision.params.invariant = text;
    changed = true;
  }
  return changed;
}

/** Problems with a parameter value (empty string if fine). */
export function validateParam(param, raw) {
  const text = raw === null || raw === undefined ? "" : String(raw).trim();
  if (text === "") return param.optional ? "" : "Required.";
  if (param.kind === "int") {
    if (!/^\d+$/.test(text)) return "Enter a whole number.";
    if (param.minimum !== null && Number(text) < param.minimum) return `At least ${param.minimum}.`;
  }
  if (param.kind === "probability") {
    const match = text.match(/^(\d+(?:\.\d*)?|\.\d+)(?:\s*\/\s*(\d+))?$/);
    if (!match) return "Enter a probability such as 0.99 or 99/100.";
    const value = match[2] ? Number(match[1]) / Number(match[2]) : Number(match[1]);
    if (!(value >= 0 && value <= 1)) return "Between 0 and 1.";
  }
  return "";
}

function observeBefore(source, line) {
  const lines = source.split("\n").slice(0, Math.max(0, (line || 1) - 1));
  return lines.some((text) => /\bobserve\s*\(/.test(text.replace(/(#|\/\/).*$/, "")));
}

/** Loops (and strategies) whose "Parameters" disclosure is open; kept across re-renders. */
const openParams = new Set();

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;

// placeholders in the syntax of the program (set by renderSetup)
const PLACEHOLDERS = {
  pgcl: { program: "nat x;\nif (x = 1) { … } else { skip }", expression: "1/(1 - c/2)" },
  ppy: { program: "x: nat;\nif (x == 1) { … } else { skip; }", expression: "1/(1 - c/2)" },
};
let placeholders = PLACEHOLDERS.pgcl;

function paramField(param, value, onInput, suggest = null) {
  const id = nextId(`param-${param.name}`);
  const errorId = `${id}-error`;
  // an invariant program is written in the syntax of the program
  const text = placeholders === PLACEHOLDERS.ppy ? param.label.replace("pGCL program", ".ppy program") : param.label;
  const label = h("label", { for: id }, text, param.optional ? h("span", { class: "optional" }, " (optional)") : null);
  let control;
  if (param.kind === "pgcl") {
    control = h("textarea", { id, class: "mono", rows: "7", spellcheck: "false", placeholder: placeholders.program });
  } else if (param.kind === "choice") {
    control = h("select", { id }, ...param.choices.map((choice) => h("option", { value: choice }, choice)));
  } else {
    const numeric = param.kind === "int";
    control = h("input", {
      id, class: `field${numeric ? "" : " mono"}`, type: numeric ? "number" : "text",
      min: numeric && param.minimum !== null ? String(param.minimum) : null, step: numeric ? "1" : null,
      inputmode: numeric ? "numeric" : null, spellcheck: "false", autocomplete: "off",
      placeholder: param.kind === "probability" ? "0.99" : param.kind === "expression" ? placeholders.expression
        : numeric && param.optional && param.default !== null && param.default !== undefined ? `${param.default} (default)` : "",
    });
  }
  control.value = value ?? "";
  control.setAttribute("aria-describedby", `${errorId} ${id}-help`);
  const error = h("p", { class: "field-error", id: errorId, "aria-live": "polite" });
  const update = () => {
    const problem = validateParam(param, control.value);
    error.textContent = problem;
    control.setAttribute("aria-invalid", problem ? "true" : "false");
    onInput(param.kind === "int" && /^\d+$/.test(control.value.trim()) ? Number(control.value) : control.value);
  };
  control.addEventListener("input", update);
  const problem = value !== undefined && value !== "" ? validateParam(param, control.value) : "";
  error.textContent = problem;
  if (problem) control.setAttribute("aria-invalid", "true");
  const action = suggest ? suggest(control) : null;
  return h("div", { class: "form-row" }, label,
    action ? h("div", { class: "field-with-action" }, control, action.button) : control,
    error, action ? action.status : null,
    param.help ? h("p", { class: "help", id: `${id}-help` }, param.help) : null);
}

// ---- "Suggest": one candidate from a language model (POST /api/suggest) ----------------------

/** Requests in flight or finished, by loop and strategy; they outlive re-renders of the panel. */
const suggestions = new Map();

/** The Suggest button and its status line for the candidate field of an EVT strategy.
 * ctx.suggest(loopId, form, {signal, onProgress}) runs the request and resolves to its result. */
function suggestAction(ctx, loop, strategy, param, control) {
  const ai = ctx.meta.ai || { available: false, reason: "This server does not offer suggestions." };
  const key = `${loop.id}:${strategy.name}`;
  const statusId = nextId("suggest-status");
  const button = h("button", { type: "button", class: "button", "aria-describedby": statusId });
  const status = h("div", { class: "suggest-status", id: statusId });
  const model = ai.model || "the language model";

  const fill = (entry) => {
    // the field may have been rendered again meanwhile: fill the decision and the current field
    const decision = ctx.setup.decisions[String(loop.id)];
    if (!decision || decision.strategy !== strategy.name) return false;
    entry.previous = decision.params[param.name] ?? "";
    decision.params[param.name] = entry.candidate;
    const field = entry.control?.isConnected ? entry.control : null;
    if (field) {
      field.value = entry.candidate;
      field.dispatchEvent(new Event("input"));
    } else {
      ctx.onChange();
    }
    return true;
  };

  const view = () => {
    const entry = suggestions.get(key);
    const running = entry?.state === "running";
    button.disabled = !ai.available;
    button.setAttribute("aria-label", running ? `Stop asking ${model}` : `Suggest a candidate with ${model}`);
    button.title = ai.available ? (running ? "Stop the request" : `Ask ${model} for a candidate`) : ai.reason;
    button.replaceChildren(running ? "Stop" : "Suggest");
    if (!ai.available) {
      status.replaceChildren(h("p", { class: "help" }, `Suggest needs a language model on the server. ${ai.reason}`));
      return;
    }
    if (!entry) {
      status.replaceChildren();
      return;
    }
    const answer = entry.answer ? h("details", { class: "suggest-answer" },
      h("summary", {}, `Answer of ${entry.model || model}`), h("pre", { class: "mono" }, entry.answer)) : null;
    if (running) {
      status.replaceChildren(h("p", { class: "suggest-progress" }, h("span", { class: "spinner", "aria-hidden": "true" }),
        h("span", { role: "status" }, `Asking ${model}…`),
        h("span", { class: "suggest-elapsed", "aria-hidden": "true" }, entry.elapsed ? `${Math.round(entry.elapsed)} s` : "")));
    } else if (entry.state === "done") {
      const undo = entry.previous ? h("button", { type: "button", class: "link-button" }, "Restore previous") : null;
      undo?.addEventListener("click", () => {
        entry.candidate = entry.previous;
        fill(entry);
        suggestions.delete(key);
        view();
        entry.control?.focus();
      });
      status.replaceChildren(h("div", { class: "notice", role: "status" }, icon("info"),
        h("div", {}, h("p", {}, `Proposed by ${entry.model}. Only a proposal: review or edit it, then analyse; Prodigy checks it. `, undo),
          answer)));
    } else {
      status.replaceChildren(h("div", { class: "notice error", role: "alert" }, icon("error"),
        h("div", {}, h("p", {}, entry.message), answer)));
    }
  };

  const start = async () => {
    const entry = { state: "running", elapsed: 0, controller: new AbortController(), control, view, source: ctx.currentSource() };
    suggestions.set(key, entry);
    view();
    let result;
    try {
      result = await ctx.suggest(loop.id, strategy.name, {
        signal: entry.controller.signal,
        onProgress: (elapsed) => {
          entry.elapsed = elapsed;
          if (suggestions.get(key) === entry) entry.view?.();
        },
      });
    } catch (error) {
      result = { status: "error", error: { message: error.message } };
    }
    if (suggestions.get(key) !== entry) return;
    const suggestion = result.suggestion || result.error?.suggestion || {};
    Object.assign(entry, { model: suggestion.model, answer: suggestion.answer, candidate: suggestion.candidate });
    if (result.status === "ok" && suggestion.candidate) {
      if (ctx.currentSource() !== entry.source) {
        Object.assign(entry, { state: "error", message: `The program changed while ${model} was asked; ask again.` });
      } else if (fill(entry)) {
        entry.state = "done";
      } else {
        suggestions.delete(key);
      }
    } else if (result.status === "cancelled") {
      suggestions.delete(key);
    } else {
      Object.assign(entry, { state: "error", message: result.error?.message || result.status || "The request failed." });
    }
    entry.view?.();
  };

  button.addEventListener("click", () => {
    const entry = suggestions.get(key);
    if (entry?.state === "running") entry.controller.abort();
    else start();
  });
  const entry = suggestions.get(key);
  if (entry) Object.assign(entry, { control, view }); // the field on screen now
  view();
  return { button, status };
}

/** The strategies the menu offers: in Analyze mode not those of `hidden` (the strategy `invariant`, a comparison
 * with a second program, belongs to Compare mode, #399). */
export function offeredStrategies(strategies, hidden = null) {
  return hidden ? strategies.filter((s) => !hidden.has(s.name)) : strategies;
}

/** A pop-up button with sections (one per group); the summary of the chosen strategy describes it. */
function strategySelect(strategies, current, summaryId, onChange) {
  const id = nextId("strategy");
  const select = h("select", { id, "aria-describedby": summaryId });
  for (const group of groupStrategies(strategies)) {
    select.append(h("optgroup", { label: group.title },
      ...group.members.map((s) => h("option", { value: s.name, title: s.summary || s.description }, s.title))));
  }
  select.value = current;
  select.addEventListener("change", () => onChange(select.value));
  return h("div", { class: "form-row" }, h("label", { for: id }, "Strategy"), select);
}

/** "How to choose": every group with its strategies and their one-sentence summaries. */
function strategyGuide(strategies, compareNote = false) {
  return h("details", { class: "disclosure strategy-guide" }, h("summary", {}, "How to choose a strategy"),
    h("div", { class: "disclosure-body" }, compareNote ? h("p", { class: "help" }, "To check a loop against a loop-free "
      + "program you write (Check invariant), use “Compare this loop” on its card: that is a comparison of two programs.") : null,
    ...groupStrategies(strategies).map((group) => h("section", { class: "guide-group" },
      h("h5", { class: "guide-title" }, group.title),
      GROUP_HELP[group.key] ? h("p", { class: "help" }, GROUP_HELP[group.key]) : null,
      h("dl", { class: "guide-list" }, ...group.members.flatMap((s) => [
        h("dt", {}, s.title), h("dd", { class: "help" }, s.summary || s.description)]))))));
}

function paramSummary(params, values) {
  const shown = params.filter((p) => values[p.name] !== undefined && values[p.name] !== null && values[p.name] !== "")
    .map((p) => `${p.label.replace(/ \(.*\)$/, "")} = ${values[p.name]}`);
  return shown.length ? shown.join(" · ") : "defaults";
}

function canSuggest(ctx, strategy, param) {
  const forms = ctx.meta.ai?.forms || [];
  return Boolean(ctx.suggest) && param.kind === "expression" && forms.includes(strategy.name);
}

function loopCard(loop, ctx) {
  const { strategies, setup, onChange, source } = ctx;
  const byName = Object.fromEntries(strategies.map((s) => [s.name, s]));
  const key = String(loop.id);
  // the loops of program B in Compare mode have ids of their own: their cards are "loop-card-b-0", …
  const domId = `${ctx.idPrefix || ""}${loop.id}`;
  if (!setup.decisions[key] || !byName[setup.decisions[key].strategy]) {
    // Compare has defaults of its own (mode-setup.js: compareDefault)
    setup.decisions[key] = ctx.defaultFor?.(loop) || defaultDecision(strategies);
  }
  const card = h("article", { class: "card", id: `loop-card-${domId}`, "aria-labelledby": `loop-title-${domId}` });
  const offered = offeredStrategies(strategies, ctx.hidden);
  const comparable = Boolean(ctx.onCompareLoop) && !loop.function;

  const render = () => {
    const decision = setup.decisions[key];
    const strategy = byName[decision.strategy];
    const head = h("div", { class: "loop-head" },
      h("h4", { id: `loop-title-${domId}`, class: "loop-name" }, `${ctx.loopLabel || "Loop"} ${loop.id} `,
        h("span", { class: "loop-line" }, loop.line ? `· line ${loop.line}` : "")),
      outcomeChip(strategy));
    const code = h("div", { class: "loop-code", title: `while (${loop.cond})` }, `while (${loop.cond})`);
    if (ctx.hidden?.has(decision.strategy)) {
      // a decision Analyze no longer offers (saved, from the library or from Compare): it stays and runs, and
      // the card says what it is instead of silently changing it (#399)
      card.replaceChildren(head, code, ...comparedLoop(loop, decision, ctx, () => {
        setup.decisions[key] = defaultDecision(offered);
        render();
        onChange();
        card.querySelector("select")?.focus();
      }));
      return;
    }
    const { required, more } = splitParams(strategy);
    const summaryId = `loop-summary-${domId}`;
    const warnings = [];
    if (strategy.requires_unconditioned && observeBefore(source, loop.line)) {
      warnings.push(h("p", { class: "notice warning" }, icon("warning"),
        h("span", {}, "This strategy does not support observe statements before the loop.")));
    }
    const byDefault = ctx.programName ? defaultReason(decision, ctx.programName) : null;
    card.replaceChildren(...[
      head,
      code,
      decision.compared ? comparedNote(decision.compared) : null,
      byDefault ? h("p", { class: "notice compare-default", "data-default": "true" }, icon("info"), h("span", {}, byDefault)) : null,
      strategySelect(offered, decision.strategy, summaryId, (name) => {
        setup.decisions[key] = { strategy: name, params: defaultParams(byName[name]) };
        render();
        onChange();
        card.querySelector("select")?.focus();
      }),
      h("p", { class: "help strategy-summary", id: summaryId }, strategy.summary || strategy.description),
      ...warnings,
      required.length ? h("div", { class: "params" }, ...required.map((p) => field(p))) : null,
      more.length ? moreParams() : null,
      strategy.summary && strategy.description ? h("details", { class: "disclosure strategy-about" },
        h("summary", {}, "About this strategy"), h("div", { class: "disclosure-body" }, h("p", { class: "help" }, strategy.description))) : null,
      comparable ? compareButton(loop, ctx, "Compare this loop", "Check this loop against a loop-free program in Compare mode") : null,
    ].filter(Boolean));

    function field(param, onValue = null) {
      return paramField(param, decision.params[param.name], (value) => {
        decision.params[param.name] = value;
        // a default edited by hand is a choice: it no longer follows the programs
        if (decision.byDefault) {
          delete decision.byDefault;
          card.querySelector(".compare-default")?.remove();
        }
        onValue?.();
        onChange();
      }, canSuggest(ctx, strategy, param) ? (control) => suggestAction(ctx, loop, strategy, param, control)
        : param.kind === "pgcl" && ctx.programB ? (control) => programBAction(ctx, control) : null);
    }

    // parameters with defaults: behind a disclosure whose summary shows their values (progressive disclosure)
    function moreParams() {
      const note = h("span", { class: "disclosure-note" });
      const update = () => { note.textContent = paramSummary(more, decision.params); };
      update();
      const details = h("details", { class: "disclosure loop-params" },
        h("summary", {}, "Parameters ", note), h("div", { class: "disclosure-body params" }, ...more.map((p) => field(p, update))));
      const openKey = `${domId}:${strategy.name}`;
      details.open = openParams.has(openKey) || more.some((p) => validateParam(p, decision.params[p.name]));
      details.addEventListener("toggle", () => {
        if (details.open) openParams.add(openKey);
        else openParams.delete(openKey);
      });
      return details;
    }
  };
  render();
  return card;
}

// ---- loops compared with a loop-free program (#399) ------------------------------------------------

/** "Use program B" next to the invariant program of a loop of program A in Compare mode (and "Use program A" for
 * a loop of program B): the loop is checked against the other program (Park's lemma). */
function programBAction(ctx, control) {
  const other = ctx.programName === "B" ? "A" : "B";
  const button = h("button", { type: "button", class: "button", title: `Check this loop against program ${other} (Park's lemma)` },
    `Use program ${other}`);
  button.addEventListener("click", () => {
    control.value = ctx.programB();
    control.dispatchEvent(new Event("input"));
    control.focus();
  });
  return { button, status: null };
}

function compareButton(loop, ctx, label, title) {
  return h("div", { class: "loop-actions" }, h("button", { type: "button", class: "button small", title,
    onclick: () => ctx.onCompareLoop(loop.id) }, icon("compare"), h("span", {}, label)));
}

/** The note on a loop whose decision came from "Continue in Analyze" of a comparison. */
function comparedNote(compared) {
  const text = compared.guarantee === "exact"
    ? "From Compare: this loop equals program B for every input; the strategy below proved it there. The result shows what it proves here."
    : "From Compare: program B is this loop's result, an upper bound (Program A ⪯ Program B).";
  return h("p", { class: "notice compared-note" }, icon("compare"), h("span", {}, text));
}

/** The body of a loop card in Analyze mode whose loop continues with a loop-free program B (the strategy
 * `invariant`): what that means, program B, and the ways on: Compare, or another strategy. */
function comparedLoop(loop, decision, ctx, chooseOther) {
  const program = decision.params?.invariant || "";
  const text = decision.compared
    ? "From Compare: program B is this loop's result, an upper bound (Program A ⪯ Program B). Analyze checks it again (Park's lemma)."
    : "This loop continues with a loop-free program B (Check invariant): an upper bound by Park's lemma, exact if the "
      + "loop terminates almost surely. Checking a loop against a program is a comparison; edit it in Compare.";
  const programView = h("details", { class: "disclosure compared-program" }, h("summary", {}, "Program B"),
    h("div", { class: "disclosure-body" }, h("pre", { class: "mono compared-source" }, program.trim() || "(empty)")));
  const actions = h("div", { class: "loop-actions" },
    ctx.onCompareLoop && !loop.function ? h("button", { type: "button", class: "button small",
      title: "Open this loop as program A and program B in Compare mode", onclick: () => ctx.onCompareLoop(loop.id) },
    icon("compare"), h("span", {}, "Open in Compare")) : null,
    h("button", { type: "button", class: "button small", onclick: chooseOther }, "Choose a strategy"));
  return [h("p", { class: "notice compared-note" }, icon("info"), h("span", {}, text)), programView, actions];
}

function toggleRow(label, help, checked, onToggle) {
  const id = nextId("toggle");
  const input = h("input", { type: "checkbox", id, role: "switch" });
  input.checked = Boolean(checked);
  input.addEventListener("change", () => onToggle(input.checked));
  return h("div", { class: "form-row" },
    h("label", { class: "toggle", for: id }, h("span", {}, label), input, h("span", { class: "switch", "aria-hidden": "true" })),
    help ? h("p", { class: "help" }, help) : null);
}

function numberRow(label, help, value, { min, max }, onValue) {
  const id = nextId("number");
  const input = h("input", { type: "number", id, class: "field", min: String(min), max: String(max), step: "1", inputmode: "numeric" });
  input.value = value;
  input.style.maxWidth = "120px";
  input.addEventListener("change", () => {
    const n = Math.max(min, Math.min(max, Math.round(Number(input.value) || min)));
    input.value = n;
    onValue(n);
  });
  return h("div", { class: "form-row" }, h("label", { for: id }, label), input, help ? h("p", { class: "help" }, help) : null);
}

function selectRow(label, help, value, choices, onValue) {
  const id = nextId("select");
  const select = h("select", { id }, ...choices.map(([v, text]) => h("option", { value: v }, text)));
  select.value = value;
  select.addEventListener("change", () => onValue(select.value));
  return h("div", { class: "form-row" }, h("label", { for: id }, label), select, help ? h("p", { class: "help" }, help) : null);
}

/** Shows `message` (empty: none) as the error of the initial-distribution field `input`. The error
 * paragraph is the one after the field unless given. */
export function applyInputProblem(input, message, error = input.parentElement?.querySelector(".field-error")) {
  if (error) {
    error.textContent = message;
    // a refused distribution can persist across reloads: offer to reset it right at the field
    if (message && input.value.trim()) {
      error.append(" ", h("button", { type: "button", class: "link-button", onclick: () => {
        input.value = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.focus();
      } }, "Reset to 1"));
    }
  }
  input.setAttribute("aria-invalid", message ? "true" : "false");
}

/** Renders the setup panel. ctx: {outline, setup, strategies, meta, source, onChange, focusLoop,
 * suggest(loopId, form, {signal, onProgress}), currentSource(), hidden, onCompareLoop(loopId), second}; `hidden`
 * (a Set) names strategies the menu does not offer (Analyze mode: `invariant`, #399), and `onCompareLoop` adds
 * "Compare this loop" to each loop card; `second` ({outline, setup: {decisions},
 * source, syntax, outlineStale}) adds the loops of program B in Compare mode (their cards are "loop-card-b-<id>"). */
export function renderSetup(container, ctx) {
  const { outline, setup, meta, onChange } = ctx;
  const comparing = Boolean(ctx.second);
  placeholders = PLACEHOLDERS[ctx.syntax] || PLACEHOLDERS.pgcl;
  const inputId = nextId("input");
  const input = h("input", { id: inputId, class: "field mono input-distribution", type: "text", spellcheck: "false",
    autocomplete: "off", placeholder: "1   (every variable starts at 0)", "aria-describedby": `${inputId}-error ${inputId}-help` });
  input.value = setup.input || "";
  const inputError = h("p", { class: "field-error", id: `${inputId}-error`, "aria-live": "polite" });
  applyInputProblem(input, ctx.inputProblem || "", inputError);
  input.addEventListener("input", () => { setup.input = input.value; onChange(); ctx.onInputChange?.(); });

  const loops = outline?.loops || [];
  const loopCards = loops.map((loop) => loopCard(loop, ctx));
  pruneDecisions(setup.decisions, outline);

  const options = setup.options;
  const label = (name) => name.replace(/_/g, " ");
  const heuristicRows = [
    selectRow("Template heuristic", "Candidate shapes for the strategy “from heuristics”.", options.templateHeuristic,
      meta.template_heuristics.map((t) => [t, label(t)]), (v) => { options.templateHeuristic = v; onChange(); }),
    selectRow("Positivity heuristic", "Check that a candidate has no negative coefficients, for “from heuristics”.",
      options.positivityHeuristic, meta.positivity_heuristics.map((p) => [p, label(p)]),
      (v) => { options.positivityHeuristic = v; onChange(); }),
  ];
  for (const row of heuristicRows) row.hidden = options.synthesis !== "from_heuristics";
  const optionsBody = h("div", { class: "disclosure-body options-grid" },
    toggleRow("Condition on observations", "Divide by the probability that all observe statements hold.", options.normalize,
      (v) => { options.normalize = v; onChange(); }),
    toggleRow("Simplify expressions", "Slower, but gives shorter closed forms.", options.simplification,
      (v) => { options.simplification = v; onChange(); }),
    toggleRow("Record each step", "Record the trace for the timeline of the analysis card: the distribution at each instruction, loop iterates and certificate checks.", options.recordSteps,
      (v) => { options.recordSteps = v; onChange(); }),
    numberRow("Coefficients per marginal", "How many values the charts and tables show.", options.coefficients,
      { min: 0, max: meta.limits.max_coefficients }, (v) => { options.coefficients = v; onChange(); }),
    numberRow("Time limit (seconds)", `At most ${meta.limits.time_limit} s on this server.`, options.timeLimit,
      { min: 1, max: meta.limits.time_limit }, (v) => { options.timeLimit = v; onChange(); }),
    selectRow("EVT synthesis strategy", "Templates and positivity checks used by “Synthesise EVT invariant”.",
      options.synthesis, meta.synthesis_strategies.map((s) => [s, s.replace(/_/g, " ")]), (v) => {
        options.synthesis = v;
        for (const row of heuristicRows) row.hidden = v !== "from_heuristics";
        onChange();
      }),
    ...heuristicRows,
    selectRow("Solver", "For equation systems during invariant checks.", options.solver,
      meta.solvers.map((s) => [s, s === "z3" ? "Z3" : "SymPy"]), (v) => { options.solver = v; onChange(); }),
  );
  const optionsDisclosure = h("details", { class: "disclosure" }, h("summary", {}, "Analysis options"), optionsBody);
  optionsDisclosure.open = Boolean(ctx.optionsOpen);
  optionsDisclosure.addEventListener("toggle", () => ctx.onOptionsToggle?.(optionsDisclosure.open));

  const loopsSection = loopSection(ctx, { outline, loopCards, title: ctx.second ? "Loops of program A" : "Loops",
    titleId: "loops-title", outlineStale: ctx.outlineStale });
  // Compare mode: program B has loops and decisions of its own
  const second = ctx.second ? (() => {
    const bCtx = { ...ctx, ...ctx.second, suggest: null, hidden: null, onCompareLoop: null, programB: ctx.programA || null,
      programName: "B", idPrefix: "b-", loopLabel: "Loop" };
    const bLoops = ctx.second.outline?.loops || [];
    const cards = bLoops.map((loop) => loopCard(loop, bCtx));
    pruneDecisions(ctx.second.setup.decisions, ctx.second.outline);
    return loopSection(bCtx, { outline: ctx.second.outline, loopCards: cards, title: "Loops of program B",
      titleId: "loops-b-title", outlineStale: ctx.second.outlineStale, guide: false });
  })() : null;

  if (comparing) {
    renderCompareSetup(container, ctx, { loopsSection, second });
    return;
  }
  const inputHelp = h("p", { class: "help", id: `${inputId}-help` }, ctx.syntax === "ppy"
    ? "A generating function over the program variables, e.g. x**3 for x = 3, or (x + x**2)/2."
    : "A generating function over the program variables, e.g. x^3 for x = 3, or (x + x^2)/2.");
  container.replaceChildren(
    h("section", { class: "section", "aria-labelledby": "input-title" },
      h("div", { class: "section-header" }, h("h3", { class: "section-title", id: "input-title" }, "Input")),
      h("div", { class: "form-row" }, h("label", { for: inputId }, "Initial distribution"), input, inputError, inputHelp)),
    loopsSection,
    h("section", { class: "section" }, optionsDisclosure),
  );
}

/** The Setup of Compare: only what the check for every input needs. A loop section for each program that has
 * loops (their strategies in the check), and the time limit (the engine is chosen in the toolbar). No input: the
 * check covers every input at once. */
function renderCompareSetup(container, ctx, { loopsSection, second }) {
  const { setup, meta, onChange } = ctx;
  const hasLoops = (outline, stale) => Boolean(stale) || (outline?.loops || []).length > 0;
  const sections = [
    hasLoops(ctx.outline, ctx.outlineStale) ? loopsSection : null,
    hasLoops(ctx.second.outline, ctx.second.outlineStale) ? second : null,
  ].filter(Boolean);
  const intro = h("p", { class: "help compare-intro" }, "Check proves whether programs A and B have the same output "
    + "distribution for every input (one second-order generating function stands for all inputs), so there is no "
    + "input to choose."
    + (sections.length ? " Each loop needs a strategy that the check can prove."
      : ctx.outline && ctx.second.outline ? " Neither program has loops, so there is nothing to choose." : ""));
  const check = h("section", { class: "section", "aria-labelledby": "check-options-title" },
    h("div", { class: "section-header" }, h("h3", { class: "section-title", id: "check-options-title" }, "Check")),
    numberRow("Time limit (seconds)", `At most ${meta.limits.time_limit} s on this server.`, setup.options.timeLimit,
      { min: 1, max: meta.limits.time_limit }, (v) => { setup.options.timeLimit = v; onChange(); }));
  container.replaceChildren(intro, ...sections, check);
}

function loopSection(ctx, { outline, loopCards, title, titleId, outlineStale, guide = true }) {
  const loops = outline?.loops || [];
  return h("section", { class: "section", "aria-labelledby": titleId },
    h("div", { class: "section-header" }, h("h3", { class: "section-title", id: titleId }, title),
      h("span", { class: "section-note" }, loops.length ? `${loops.length}` : "")),
    outlineStale ? h("p", { class: "notice warning" }, icon("warning"),
      h("span", {}, "The program does not parse right now; the loops are those of the last version that did.")) : null,
    loops.length && guide ? strategyGuide(offeredStrategies(ctx.strategies, ctx.hidden), Boolean(ctx.hidden)) : null,
    loops.length ? loopCards : h("p", { class: "help" },
      outline ? "This program has no while loops; its semantics is computed exactly." : "Loops appear here once the program parses."));
}

/** Scrolls to the card of a loop and focuses its strategy pop-up, or with `field` its first text field. */
export function focusLoopCard(container, loopId, { field = false } = {}) {
  const card = container.querySelector(`#loop-card-${loopId}`);
  if (!card) return;
  card.scrollIntoView({ block: "center", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  card.classList.add("focused");
  setTimeout(() => card.classList.remove("focused"), 1600);
  const target = field ? card.querySelector(".params input.field, .params textarea") : null;
  (target || card.querySelector("select, .loop-actions button"))?.focus({ preventScroll: true });
}

/** Problems that prevent running (shown next to the Analyze button): {loop, message, param} (`param` names the
 * parameter whose value is missing or wrong). `syntax` is the syntax of the program (for the labels). */
export function setupProblems(outline, setup, strategies, syntax = "pgcl") {
  const byName = Object.fromEntries(strategies.map((s) => [s.name, s]));
  const problems = [];
  for (const loop of outline?.loops || []) {
    const decision = setup.decisions[String(loop.id)];
    const strategy = decision && byName[decision.strategy];
    if (!strategy) {
      problems.push({ loop: loop.id, message: `Choose a strategy for loop ${loop.id}.` });
      continue;
    }
    for (const param of strategy.params.filter((p) => p.kind !== "file")) {
      const problem = validateParam(param, decision.params[param.name]);
      const label = syntax === "ppy" ? param.label.replace("pGCL program", ".ppy program") : param.label;
      if (problem) problems.push({ loop: loop.id, param: param.name, message: `Loop ${loop.id}: ${label}: ${problem}` });
    }
  }
  return problems;
}
