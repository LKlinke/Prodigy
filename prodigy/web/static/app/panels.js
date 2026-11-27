// The inspector (Setup and Result tabs), the setup panel, the engine pop-up and the annotations in the editors.
import { $, editor, otherEditor, saveSetup, state } from "./app-state.js";
import { scheduleInputCheck } from "./checking.js";
import { compareLoop } from "./compare-mode.js";
import { compareDefault } from "./mode-setup.js";
import { refreshStale, showAnalysisStep } from "./result-panel.js";
import { requestSuggestion } from "./run.js";
import { defaultDecision, focusLoopCard, renderSetup } from "./setup.js";
import { save } from "./store.js";
import { h } from "./components/dom.js";
import { BOUND_TEXT, qualify } from "./components/format.js";

/** Strategies Analyze mode does not offer: checking a loop against a program is a comparison (#399). */
const COMPARE_ONLY = new Set(["invariant"]);

const TABS = ["setup", "result"];

export function selectTab(name, { focus = false } = {}) {
  for (const tab of TABS) {
    const button = $(`tab-${tab}`);
    const selected = tab === name;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(`panel-${tab}`).hidden = !selected;
    if (selected && focus) button.focus();
  }
  state.tab = name;
  if (name !== "result") state.analysis.card?.stop();
  showAnalysisStep();
}

export function initTabs() {
  for (const tab of TABS) {
    $(`tab-${tab}`).addEventListener("click", () => selectTab(tab));
    $(`tab-${tab}`).addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        const next = TABS[(TABS.indexOf(tab) + (event.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
        selectTab(next, { focus: true });
      }
    });
  }
}

export function renderSetupPanel() {
  renderSetup($("panel-setup"), {
    outline: state.outline,
    setup: state.setup,
    strategies: state.strategies,
    meta: state.meta,
    source: editor.getSource(),
    syntax: editor.getSyntax(),
    outlineStale: Boolean(state.outlineStale && state.outline),
    optionsOpen: state.optionsOpen,
    onOptionsToggle: (open) => { state.optionsOpen = open; },
    inputProblem: state.inputProblem || "",
    onInputChange: scheduleInputCheck,
    onChange: () => {
      saveSetup();
      refreshStale();
      updateAnnotations();
    },
    suggest: requestSuggestion,
    currentSource: () => editor.getSource(),
    ...(state.mode === "analyze" ? { hidden: COMPARE_ONLY, onCompareLoop: compareLoop } : {}),
    ...(state.mode === "compare" ? {
      programName: "A",
      programB: () => otherEditor.getSource(),
      programA: () => editor.getSource(),
      defaultFor: (loop) => compareDefault(loop, state.outline, { outline: state.compare.outline, source: otherEditor.getSource() }),
      second: {
        outline: state.compare.outline,
        outlineStale: Boolean(state.compare.outlineStale && state.compare.outline),
        setup: { decisions: state.compare.decisions },
        source: otherEditor.getSource(),
        syntax: otherEditor.getSyntax(),
        defaultFor: (loop) => compareDefault(loop, state.compare.outline, { outline: state.outline, source: editor.getSource() }),
        onChange: () => {
          save("compareDecisionsB", state.compare.decisions);
          refreshStale();
          updateAnnotations();
        },
      },
    } : {}),
  });
  if (state.mode === "compare") save("compareDecisionsB", state.compare.decisions);
  saveSetup();
}

const ENGINE_HELP = "The engine that computes the generating functions. GiNaC is usually much faster; "
  + "SymPy is the reference implementation.";

/** Fills the engine pop-up from the server's engines; unavailable ones stay listed but disabled. */
export function renderEngine() {
  const select = $("engine-select");
  const engines = state.meta.engines;
  select.replaceChildren(...engines.map((engine) => h("option", {
    value: engine.name, disabled: engine.available ? null : "", title: engine.reason || null,
  }, engine.available ? engine.label : `${engine.label} (not installed)`)));
  select.value = state.setup.options.engine;
  const missing = engines.filter((e) => !e.available && e.reason).map((e) => e.reason);
  select.title = [ENGINE_HELP, ...missing].join(" ");
}

export function engineChanged() {
  state.setup.options.engine = $("engine-select").value;
  saveSetup();
  scheduleInputCheck();
  refreshStale();
  updateAnnotations();
}

export function showLoop(loopId, { field = false } = {}) {
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId, { field });
}

/** "Use as EVT invariant candidate" of an exploration: the loop switches to Check EVT invariant with the
 * candidate, and the Setup shows its field to review or edit before analysing again (#324). */
export function useEvtCandidate(loopId, candidate) {
  state.setup.decisions[String(loopId)] = { strategy: "evt_invariant", params: { invariant: candidate } };
  saveSetup();
  refreshStale();
  renderSetupPanel();
  updateAnnotations();
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId, { field: true });
}

/** The pill above a loop; `other` names the other program of a comparison and gives its text. */
function shortStrategy(decision, other = null) {
  const p = decision.params || {};
  if (decision.strategy === "invariant" && other) {
    if (!String(p.invariant ?? "").trim()) return "Check invariant · write it in Setup";
    if (p.invariant === other.source) return `Check invariant: program ${other.name}`;
    return "Check invariant";
  }
  switch (decision.strategy) {
    case "unroll": return `Unroll, N = ${p.iterations ?? "?"}`;
    case "mass_threshold": return `Until mass ${p.threshold ?? "?"}`;
    case "auto": return `Automatic, N = ${p.iterations ?? "?"}`;
    case "evt_iterate": return `Explore EVT, N = ${p.iterations ?? "?"}`;
    case "invariant": return "Invariant";
    case "evt_invariant": return "EVT invariant";
    case "evt_synthesis": return "EVT synthesis";
    default: return state.titles[decision.strategy] || decision.strategy;
  }
}

export function updateAnnotations() {
  const loops = (state.outline?.loops || []).map((loop) => {
    const decision = state.setup.decisions[String(loop.id)] || defaultDecision(state.strategies);
    const strategy = shortStrategy(decision, state.mode === "compare" ? { name: "B", source: otherEditor.getSource() } : null);
    const spec = { id: loop.id, line: loop.line, strategy, onClick: showLoop };
    const record = !state.resultStale ? (state.result?.loops || []).find((l) => l.id === loop.id) : null;
    if (record) {
      spec.kind = record.failed ? "error" : record.kind;
      spec.outcome = record.failed ? "failed" : (BOUND_TEXT[record.kind] || BOUND_TEXT.unknown).label.toLowerCase();
    }
    return spec;
  });
  const queries = [];
  if (state.result && !state.resultStale) {
    for (const query of state.result.queries || []) {
      if (!query.line || !["probability", "expectation", "variance", "covariance"].includes(query.kind)) continue;
      const { glyph, text } = qualify(query, { withValue: true });
      queries.push({ line: query.line, kind: query.bound, glyph: glyph || "", text,
        title: `${(BOUND_TEXT[query.bound] || BOUND_TEXT.unknown).label}: ${query.value ?? ""}` });
    }
  }
  editor.setAnnotations({ loops, queries });
  // program B in Compare mode: the strategy above each loop, and after a check what it certified
  const otherLoops = state.mode === "compare" && !state.resultStale ? state.result?.other_loops || [] : [];
  otherEditor?.setAnnotations({ loops: state.mode !== "compare" ? [] : (state.compare.outline?.loops || []).map((loop) => {
    const decision = state.compare.decisions[String(loop.id)] || defaultDecision(state.strategies);
    const spec = { id: loop.id, line: loop.line, strategy: shortStrategy(decision, { name: "A", source: editor.getSource() }),
      onClick: (id) => showLoop(`b-${id}`) };
    const record = otherLoops.find((l) => l.id === loop.id);
    if (record) {
      spec.kind = record.failed ? "error" : record.kind;
      spec.outcome = record.failed ? "failed" : (BOUND_TEXT[record.kind] || BOUND_TEXT.unknown).label.toLowerCase();
    }
    return spec;
  }), queries: [] });
}
