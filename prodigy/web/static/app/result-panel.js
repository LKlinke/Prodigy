// The Result panel: the result of Analyze (its analysis card, timeline, watches, joint grids) or the verdict
// of Compare, and whether it is still current.
import { api } from "./api.js";
import { $, editor, saveSetup, state } from "./app-state.js";
import { continueTarget, openInCompare } from "./compare-mode.js";
import { renderSetupPanel, selectTab, showLoop, useEvtCandidate } from "./panels.js";
import { buildRequest, run } from "./run.js";
import { changeSince } from "./staleness.js";
import { save } from "./store.js";
import { h } from "./components/dom.js";
import { renderEquivalence } from "./components/equivalence-view.js";
import { renderResult } from "./components/result-view.js";

/** Compares the request of the shown result with the current program and setup; changing them
 * back (e.g. with undo) makes the result current again. */
export function refreshStale() {
  const change = state.result ? changeSince(state.result.__request, buildRequest()) : null;
  if (change === state.resultChange) return;
  state.resultChange = change;
  state.resultStale = Boolean(change);
  renderResultPanel();
}

/** Another program: no steps, no breakpoints. */
export function resetAnalysis() {
  state.analysis.card?.stop();
  state.analysis.step = null;
  state.analysisView = null;
  editor.setBreakpoints([]);
  state.analysis.breakpoints = [];
  editor.showStep(null);
}

/** The editor highlights the instruction of the timeline's position while the result is shown. */
export function showAnalysisStep() {
  const step = state.analysis.step;
  if (state.tab === "result" && step && !state.resultStale && !state.job) editor?.showStep(step);
  else editor?.showStep(null);
}

/** Records steps from now on (the option in Setup) and analyzes again. */
function recordAndRun() {
  state.setup.options.recordSteps = true;
  saveSetup();
  renderSetupPanel();
  run();
}

function changeWatches(watches) {
  state.analysis.watches = watches;
  save("watches", watches);
  recordAndRun();
}

/** F10/⇧F10 and F5/⇧F5 anywhere, ← → Home End inside the analysis card. */
export function analysisKey(event) {
  const card = state.analysis.card;
  if (!card || document.querySelector("dialog[open]")) return false;
  const global = event.key === "F10" || event.key === "F5";
  if (!global && state.tab !== "result") return false;
  if (global && state.tab !== "result") {
    if (!card.model.positions.some((p) => p.kind !== "final")) return false;
    selectTab("result");
  }
  return card.key(event);
}

const BADGE_LABELS = { exact: "Exact result", lower: "Lower bound", upper: "Upper bound", unknown: "No guarantee",
  error: "Analysis failed", running: "Analysis running", equivalent: "Equivalent", not_equivalent: "Not equivalent",
  undecided: "Equivalence not decided" };

export function setBadge(kind) {
  const badge = $("result-badge");
  badge.hidden = !kind;
  if (kind) {
    badge.dataset.kind = kind;
    badge.title = BADGE_LABELS[kind] || "";
    badge.setAttribute("role", "img");
    badge.setAttribute("aria-label", BADGE_LABELS[kind] || kind);
  }
}

export function renderResultPanel() {
  const panel = $("panel-result");
  if (state.job) return;
  if (!state.result) {
    state.analysis.card?.stop();
    state.analysis.card = null;
    state.analysis.step = null;
    showAnalysisStep();
    panel.replaceChildren(h("div", { class: "result-view" }, h("p", { class: "help" },
      state.mode === "compare"
        ? "No result yet. Press Check (⌘↩) to check whether programs A and B are equivalent."
        : "No result yet. Press Analyze (⌘↩) to compute the output distribution and the answers to the queries.")));
    setBadge(null);
    return;
  }
  state.analysis.card?.stop();
  if (state.result.equivalence) {
    renderEquivalence(panel, state.result, {
      stale: state.resultStale,
      change: state.resultChange,
      onRun: run,
      continueTo: continueTarget(),
    });
    state.analysis.card = null;
    const verdict = state.result.equivalence.verdict;
    setBadge(state.resultStale ? null : verdict === "equivalent" || verdict === "not_equivalent" ? verdict : "undecided");
    return;
  }
  // the final result takes over from the streamed charts: its bars start where those were (#405)
  const fromCharts = state.liveCharts;
  state.liveCharts = null;
  const { analysis } = renderResult(panel, state.result, {
    fromCharts,
    compare: state.mode === "compare",
    stale: state.resultStale,
    change: state.resultChange,
    onRun: run,
    strategyTitles: state.titles,
    onSelectLoop: showLoop,
    onUseCandidate: useEvtCandidate,
    onOpenInCompare: openInCompare,
    explorationState: explorationMemory(state.result),
    onSelectLine: (line) => editor.revealLine(line),
    elapsed: state.result.__elapsed,
    analysisState: analysisState(state.result),
    breakpoints: state.analysis.breakpoints,
    onPosition: (step) => { state.analysis.step = step; showAnalysisStep(); },
    onRecord: recordAndRun,
    onAddWatch: (text) => {
      if (!state.analysis.watches.includes(text)) changeWatches([...state.analysis.watches, text]);
    },
    onRemoveWatch: (text) => changeWatches(state.analysis.watches.filter((w) => w !== text)),
    ...(state.result.__job ? {
      loadJoint: (payload, signal, onPartial) => loadJoint(state.result.__job, payload, signal, onPartial),
      maxJointAxis: state.meta?.limits?.max_joint_axis,
    } : {}),
  });
  state.analysis.card = analysis;
  const kind = state.result.status === "ok" ? state.result.bounds?.kind : "error";
  setBadge(state.resultStale ? null : kind);
}

/** The chosen N of each exploration card, kept while the result stays the same. */
function explorationMemory(result) {
  if (state.explorationView?.result !== result) state.explorationView = { result, loops: {} };
  return state.explorationView.loops;
}

const jointCache = new Map();

/** The choices of the analysis card (lens, position, variable or pair, window, view) for one result. */
function analysisState(result) {
  if (!state.analysisView || state.analysisView.result !== result) {
    // a new result keeps the lens, the variable (or pair) and the kinds of view; the timeline starts
    // at the final result and the window from the default again
    const previous = state.analysisView || {};
    state.analysisView = { result, lens: previous.lens, x: previous.x, y: previous.y, mode: previous.mode,
      camera: previous.camera, subject: previous.subject, marginalMode: previous.marginalMode,
      sizeScale: previous.sizeScale, watchesOpen: previous.watchesOpen };
  }
  return state.analysisView;
}

/** Runs a grid job for two variables of the result of analysis `jobId` and resolves to the grid;
 * `onPartial(stream)` gets the cells computed so far on each poll (#405). */
async function loadJoint(jobId, payload, signal, onPartial = null) {
  const key = `${jobId} ${JSON.stringify(payload)}`;
  if (jointCache.has(key)) return jointCache.get(key);
  // the grid is no longer wanted when the signal aborts: stop it at once, not at the next poll, so that the
  // grid asked for next does not find the slot taken
  let started = null;
  signal.addEventListener("abort", () => started?.then((job) => api.cancel(job.job)).catch(() => {}), { once: true });
  started = api.joint(jobId, payload);
  const response = await started;
  for (;;) {
    if (signal.aborted) throw new DOMException("cancelled", "AbortError");
    const status = await api.job(response.job);
    if (signal.aborted) continue;
    if (status.state !== "starting" && status.state !== "running") {
      if (status.result?.status !== "ok") throw new Error(status.result?.error?.message || status.state);
      const grid = status.result.joint;
      while (jointCache.size > 24) jointCache.delete(jointCache.keys().next().value);
      jointCache.set(key, grid);
      // the same grid with its window spelled out (the view asks for it after a re-render)
      const window = { x: payload.x, y: payload.y, x_window: [grid.x_values[0], grid.x_values.at(-1)],
        y_window: [grid.y_values[0], grid.y_values.at(-1)] };
      jointCache.set(`${jobId} ${JSON.stringify(window)}`, grid);
      return grid;
    }
    if (status.stream?.grid) onPartial?.(status.stream.grid);
    await new Promise((resolve) => setTimeout(resolve, status.elapsed < 2 ? 150 : 400));
  }
}
