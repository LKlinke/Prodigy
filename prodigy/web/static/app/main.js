// Prodigy web GUI: wires the editor, the setup panel, the analysis jobs and the result view.
import { api } from "./api.js";
import { canWriteFiles, createDocument } from "./document.js";
import { createEditor, SYNTAXES } from "./editor.js";
import { createLibrary } from "./library.js";
import { defaultDecision, focusLoopCard, renderSetup, setupProblems, swapInvariants } from "./setup.js";
import { changeSince } from "./staleness.js";
import { load, save } from "./store.js";
import { h, icon } from "./components/dom.js";
import { BOUND_TEXT, formatSeconds, formatValue, qualify } from "./components/format.js";
import { renderResult, renderRunning } from "./components/result-view.js";

const WELCOME = `# A fair coin is flipped until it shows heads; c counts the tails.
# Press Analyze (⌘↩). How the loop is analysed is set in Setup.
x: nat = 1;
c: nat;

while (x == 1) {
    { x = 0; } [1/2] { c = c + 1; }
}

query Pr[c == 0];
query Pr[c < 3];
query Ex[c];
`;

const $ = (id) => document.getElementById(id);

const state = {
  meta: null,
  strategies: [],
  titles: {},
  outline: null,
  diagnostics: [],
  setup: null,
  job: null,
  result: null,
  // whether the result no longer matches the program and setup, and what changed: null, "program",
  // "setup" or "both" (the result remembers the request of its analysis, see staleness.js)
  resultStale: false,
  resultChange: null,
  optionsOpen: false,
  tab: "setup",
  // the analysis card of the result: its controller, the step its timeline shows (for the editor),
  // breakpoint lines and watches (queries after every step)
  analysis: { card: null, step: null, breakpoints: [], watches: [] },
};

const TABS = ["setup", "result"];

let editor;
let documents;
let checkTimer = null;
let checkController = null;
let pollTimer = null;

// ---- tabs ------------------------------------------------------------------------------

function selectTab(name, { focus = false } = {}) {
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

function initTabs() {
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

// ---- split view ---------------------------------------------------------------------------

function initDivider() {
  const split = $("split");
  const divider = $("divider");
  const apply = (percent) => {
    const value = Math.max(25, Math.min(75, percent));
    split.style.setProperty("--split", `${value}%`);
    divider.setAttribute("aria-valuenow", String(Math.round(value)));
    return value;
  };
  let current = apply(load("split", 55));
  divider.addEventListener("pointerdown", (event) => {
    divider.setPointerCapture(event.pointerId);
    divider.classList.add("dragging");
    const rect = split.getBoundingClientRect();
    const move = (e) => { current = apply(((e.clientX - rect.left) / rect.width) * 100); };
    const up = () => {
      divider.classList.remove("dragging");
      divider.removeEventListener("pointermove", move);
      divider.removeEventListener("pointerup", up);
      save("split", current);
    };
    divider.addEventListener("pointermove", move);
    divider.addEventListener("pointerup", up);
  });
  divider.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 10 : 2;
    if (event.key === "ArrowLeft") current = apply(current - step);
    else if (event.key === "ArrowRight") current = apply(current + step);
    else return;
    event.preventDefault();
    save("split", current);
  });
}

// ---- document title -------------------------------------------------------------------------

function renderTitle({ name, category, edited, fileName }) {
  $("doc-title").textContent = name;
  $("doc-subtitle").textContent = [category, edited ? "Edited" : ""].filter(Boolean).join(" · ");
  $("doc-title-wrap").title = fileName || "";
  document.title = `${name}${edited ? " — Edited" : ""} — Prodigy`;
}

/** New or Open replaced the program: its loops and results no longer apply. */
async function programReplaced() {
  state.setup.decisions = {};
  state.setup.input = "";
  state.pendingDefault = null;
  state.outline = null;
  state.result = null;
  state.resultStale = false;
  state.resultChange = null;
  resetAnalysis();
  save("setup", state.setup);
  renderResultPanel();
  selectTab("setup");
  clearTimeout(checkTimer);
  await runCheck();
  renderSetupPanel();
  updateAnnotations();
}

// ---- checking -------------------------------------------------------------------------------

function scheduleCheck(delay = 350) {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(runCheck, delay);
}

async function runCheck() {
  const source = editor.getSource();
  const syntax = editor.getSyntax();
  checkController?.abort();
  checkController = new AbortController();
  try {
    const result = await api.check(source, syntax, checkController.signal);
    if (source !== editor.getSource() || syntax !== editor.getSyntax()) return;
    state.diagnostics = result.diagnostics;
    editor.setDiagnostics(result.diagnostics);
    renderProblems();
    const wasStale = state.outlineStale;
    state.outlineStale = !result.outline;
    if (result.outline) state.outline = result.outline;
    if (result.outline || wasStale !== state.outlineStale) renderSetupPanel();
    // the request depends on the outline's loops
    if (result.outline) refreshStale();
    updateAnnotations();
  } catch (error) {
    if (error.name === "AbortError") return;
    state.diagnostics = [];
    renderProblems(error.message);
  }
}

function renderProblems(failure = null) {
  const button = $("problems");
  const problems = state.diagnostics.filter((d) => d.severity !== "info");
  button.replaceChildren();
  button.onclick = null;
  if (failure) {
    button.dataset.severity = "error";
    button.append(icon("error"), h("span", {}, failure));
    return;
  }
  if (!problems.length) {
    button.dataset.severity = "none";
    button.disabled = true;
    button.append(h("span", {}, "No problems"));
    return;
  }
  const first = problems.find((d) => d.severity === "error") || problems[0];
  button.disabled = false;
  button.dataset.severity = first.severity;
  button.title = "Show the first problem in the program";
  button.append(icon(first.severity === "error" ? "error" : "warning"),
    h("span", {}, `${problems.length > 1 ? `${problems.length} problems · ` : ""}Line ${first.line}: ${first.message}`));
  button.onclick = () => editor.revealLine(first.line);
}

// ---- setup ------------------------------------------------------------------------------------

function defaultSetup(meta) {
  const engine = meta.engines.find((e) => e.default)?.name || "sympy";
  return {
    input: "",
    decisions: {},
    options: {
      engine, normalize: true, simplification: true, coefficients: 30,
      timeLimit: Math.min(120, meta.limits.time_limit), synthesis: "default", solver: "sympy", recordSteps: false,
      templateHeuristic: "default", positivityHeuristic: "default",
    },
  };
}

function renderSetupPanel() {
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
    onChange: () => {
      save("setup", state.setup);
      refreshStale();
      updateAnnotations();
    },
    suggest: requestSuggestion,
    currentSource: () => editor.getSource(),
  });
  save("setup", state.setup);
}

// ---- engine (toolbar pop-up button) --------------------------------------------------------------

const ENGINE_HELP = "The engine that computes the generating functions. GiNaC is usually much faster; "
  + "SymPy is the reference implementation.";

/** Fills the engine pop-up from the server's engines; unavailable ones stay listed but disabled. */
function renderEngine() {
  const select = $("engine-select");
  const engines = state.meta.engines;
  select.replaceChildren(...engines.map((engine) => h("option", {
    value: engine.name, disabled: engine.available ? null : "", title: engine.reason || null,
  }, engine.available ? engine.label : `${engine.label} (not installed)`)));
  select.value = state.setup.options.engine;
  const missing = engines.filter((e) => !e.available && e.reason).map((e) => e.reason);
  select.title = [ENGINE_HELP, ...missing].join(" ");
}

function engineChanged() {
  state.setup.options.engine = $("engine-select").value;
  save("setup", state.setup);
  refreshStale();
  updateAnnotations();
}

function showLoop(loopId) {
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId);
}

/** "Use as EVT invariant candidate" of an exploration: the loop switches to Check EVT invariant with the
 * candidate, and the Setup shows its field to review or edit before analysing again (#324). */
function useEvtCandidate(loopId, candidate) {
  state.setup.decisions[String(loopId)] = { strategy: "evt_invariant", params: { invariant: candidate } };
  save("setup", state.setup);
  refreshStale();
  renderSetupPanel();
  updateAnnotations();
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId, { field: true });
}

// ---- annotations in the editor ------------------------------------------------------------------

function shortStrategy(decision) {
  const p = decision.params || {};
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

function updateAnnotations() {
  const loops = (state.outline?.loops || []).map((loop) => {
    const decision = state.setup.decisions[String(loop.id)] || defaultDecision(state.strategies);
    const strategy = shortStrategy(decision);
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
}

// ---- results ----------------------------------------------------------------------------------------

/** Compares the request of the shown result with the current program and setup; changing them
 * back (e.g. with undo) makes the result current again. */
function refreshStale() {
  const change = state.result ? changeSince(state.result.__request, buildRequest()) : null;
  if (change === state.resultChange) return;
  state.resultChange = change;
  state.resultStale = Boolean(change);
  renderResultPanel();
}

// ---- analysis card: timeline, editor highlight, watches ------------------------------------------

/** Another program: no steps, no breakpoints. */
function resetAnalysis() {
  state.analysis.card?.stop();
  state.analysis.step = null;
  state.analysisView = null;
  editor.setBreakpoints([]);
  state.analysis.breakpoints = [];
  editor.showStep(null);
}

/** The editor highlights the instruction of the timeline's position while the result is shown. */
function showAnalysisStep() {
  const step = state.analysis.step;
  if (state.tab === "result" && step && !state.resultStale && !state.job) editor?.showStep(step);
  else editor?.showStep(null);
}

/** Records steps from now on (the option in Setup) and analyzes again. */
function recordAndRun() {
  state.setup.options.recordSteps = true;
  save("setup", state.setup);
  renderSetupPanel();
  run();
}

function changeWatches(watches) {
  state.analysis.watches = watches;
  save("watches", watches);
  recordAndRun();
}

/** F10/⇧F10 and F5/⇧F5 anywhere, ← → Home End inside the analysis card. */
function analysisKey(event) {
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
  error: "Analysis failed", running: "Analysis running" };

function setBadge(kind) {
  const badge = $("result-badge");
  badge.hidden = !kind;
  if (kind) {
    badge.dataset.kind = kind;
    badge.title = BADGE_LABELS[kind] || "";
    badge.setAttribute("role", "img");
    badge.setAttribute("aria-label", BADGE_LABELS[kind] || kind);
  }
}

function renderResultPanel() {
  const panel = $("panel-result");
  if (state.job) return;
  if (!state.result) {
    state.analysis.card?.stop();
    state.analysis.card = null;
    state.analysis.step = null;
    showAnalysisStep();
    panel.replaceChildren(h("div", { class: "result-view" }, h("p", { class: "help" },
      "No result yet. Press Analyze (⌘↩) to compute the output distribution and the answers to the queries.")));
    setBadge(null);
    return;
  }
  state.analysis.card?.stop();
  const { analysis } = renderResult(panel, state.result, {
    stale: state.resultStale,
    change: state.resultChange,
    onRun: run,
    strategyTitles: state.titles,
    onSelectLoop: showLoop,
    onUseCandidate: useEvtCandidate,
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
      loadJoint: (payload, signal) => loadJoint(state.result.__job, payload, signal),
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

// ---- joint distributions (grids computed by the server from the result it keeps) ------------------

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

/** Runs a grid job for two variables of the result of analysis `jobId` and resolves to the grid. */
async function loadJoint(jobId, payload, signal) {
  const key = `${jobId} ${JSON.stringify(payload)}`;
  if (jointCache.has(key)) return jointCache.get(key);
  const response = await api.joint(jobId, payload);
  for (;;) {
    if (signal.aborted) {
      api.cancel(response.job).catch(() => {});
      throw new DOMException("cancelled", "AbortError");
    }
    const status = await api.job(response.job);
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
    await new Promise((resolve) => setTimeout(resolve, status.elapsed < 2 ? 150 : 400));
  }
}

// ---- running --------------------------------------------------------------------------------------

function setRunning(running) {
  $("btn-run").hidden = running;
  $("btn-stop").hidden = !running;
  if (!running) $("run-status").replaceChildren();
}

function showRunMessage(message, tone = "error") {
  const status = $("run-status");
  status.replaceChildren(h("span", { class: tone === "error" ? "outcome" : "", "data-kind": tone === "error" ? "error" : null },
    icon(tone === "error" ? "error" : "info"), " ", message));
}

/** A short notice (opened, saved); information disappears after a few seconds. */
function notify(message, tone = "info") {
  showRunMessage(message, tone);
  const shown = $("run-status").firstChild;
  if (tone === "info") {
    setTimeout(() => {
      if (!state.job && $("run-status").firstChild === shown) $("run-status").replaceChildren();
    }, 5000);
  }
}

function buildRequest() {
  const options = state.setup.options;
  const decisions = {};
  for (const loop of state.outline?.loops || []) {
    const decision = state.setup.decisions[String(loop.id)];
    if (decision) decisions[String(loop.id)] = { strategy: decision.strategy, params: decision.params };
  }
  return {
    source: editor.getSource(),
    syntax: editor.getSyntax(),
    input_distribution: state.setup.input?.trim() || null,
    engine: options.engine,
    normalize: options.normalize,
    simplification: options.simplification,
    decisions,
    coefficients: options.coefficients,
    record_steps: options.recordSteps,
    ...(options.recordSteps && state.analysis.watches.length ? { watches: state.analysis.watches } : {}),
    synthesis_strategy: options.synthesis,
    ...(options.synthesis === "from_heuristics"
      ? { template_heuristic: options.templateHeuristic, positivity_heuristic: options.positivityHeuristic } : {}),
    solver: options.solver,
    time_limit: options.timeLimit,
  };
}

/** Asks the server's language model for a candidate for one loop (a job like an analysis, with
 * the server's time limit) and resolves to the job's result; aborting the signal cancels it. */
async function requestSuggestion(loopId, form, { signal, onProgress }) {
  const { time_limit: _limit, coefficients: _c, record_steps: _r, watches: _w, ...analysis } = buildRequest();
  const response = await api.suggest({ ...analysis, loop: loopId, form });
  for (;;) {
    if (signal.aborted) {
      const status = await api.cancel(response.job);
      return status.result || { status: "cancelled" };
    }
    const status = await api.job(response.job);
    onProgress(status.elapsed);
    if (status.state !== "starting" && status.state !== "running") return status.result || { status: status.state };
    await new Promise((resolve) => {
      const timer = setTimeout(resolve, status.elapsed < 2 ? 200 : 500);
      signal.addEventListener("abort", () => { clearTimeout(timer); resolve(); }, { once: true });
    });
  }
}

async function run() {
  if (state.job) return;
  clearTimeout(checkTimer);
  await runCheck();
  const errors = state.diagnostics.filter((d) => d.severity === "error");
  if (errors.length) {
    showRunMessage(`Fix line ${errors[0].line} first.`);
    editor.revealLine(errors[0].line);
    return;
  }
  const problems = setupProblems(state.outline, state.setup, state.strategies);
  if (problems.length) {
    showRunMessage(problems[0].message);
    showLoop(problems[0].loop);
    return;
  }
  const request = buildRequest();
  let response;
  try {
    response = await api.analyze(request);
  } catch (error) {
    showRunMessage(error.message);
    return;
  }
  state.job = { id: response.job, request, timeLimit: response.time_limit, logOffset: 0, log: "", started: performance.now() };
  setRunning(true);
  setBadge("running");
  state.analysis.card?.stop();
  state.analysis.card = null;
  state.analysis.step = null;
  editor.showStep(null);
  selectTab("result");
  showRunning({ elapsed: 0, timeLimit: state.job.timeLimit, progress: [], log: "" });
  poll();
}

async function poll() {
  const job = state.job;
  if (!job) return;
  let status;
  try {
    status = await api.job(job.id, job.logOffset);
  } catch (error) {
    finish({ status: "error", error: { type: "connection", message: error.message } }, 0);
    return;
  }
  if (state.job !== job) return;
  job.log += status.log;
  job.logOffset = status.log_offset;
  $("run-status").replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }),
    h("span", {}, `${formatSeconds(status.elapsed)}`));
  if (status.state === "starting" || status.state === "running") {
    showRunning({ elapsed: status.elapsed, timeLimit: status.time_limit, progress: status.progress, log: job.log });
    pollTimer = setTimeout(poll, status.elapsed < 2 ? 150 : 400);
    return;
  }
  finish({ ...(status.result || { status: status.state, error: { message: status.state } }), log: (status.result?.log ?? job.log),
    __job: job.id }, status.elapsed);
}

/** The progress view in the Result panel. */
function showRunning(progress) {
  const change = state.job ? changeSince(state.job.request, buildRequest()) : null;
  renderRunning($("panel-result"), progress, { strategyTitles: state.titles, change });
}

function finish(result, elapsed) {
  clearTimeout(pollTimer);
  const request = state.job?.request;
  state.job = null;
  setRunning(false);
  // the result belongs to the request of its run: program or setup changed meanwhile stay as they
  // are, and the result is shown as outdated (#310)
  state.result = { ...result, __elapsed: elapsed, __request: request };
  state.resultChange = changeSince(request, buildRequest());
  state.resultStale = Boolean(state.resultChange);
  renderResultPanel();
  updateAnnotations();
  const message = result.status === "ok" ? null : (result.error?.message || result.status);
  if (message) showRunMessage(message, result.status === "cancelled" ? "info" : "error");
}

async function stop() {
  const job = state.job;
  if (!job) return;
  try {
    const status = await api.cancel(job.id);
    if (state.job === job) finish({ ...(status.result || {}), log: job.log }, status.elapsed);
  } catch (error) {
    showRunMessage(error.message);
  }
}

// ---- examples -----------------------------------------------------------------------------------------

async function loadExample(example, syntax = "ppy") {
  if (!(await documents.confirmClose())) return;
  let data;
  try {
    data = await api.example(example.id, syntax);
  } catch (error) {
    showRunMessage(error.message);
    return;
  }
  editor.setSource(data.source, data.syntax || "ppy");
  const suggestion = data.suggestion || {};
  state.setup.decisions = {};
  for (const [key, decision] of Object.entries(suggestion.decisions || {})) {
    if (key === "*") continue;
    state.setup.decisions[key] = { strategy: decision.strategy, params: { ...(decision.params || {}) } };
  }
  state.pendingDefault = suggestion.decisions?.["*"] || null;
  state.setup.input = suggestion.input_distribution || "";
  state.result = null;
  state.resultStale = false;
  state.resultChange = null;
  resetAnalysis();
  documents.exampleLoaded(example, data.source);
  renderResultPanel();
  const renamed = renamingText(data.renamings);
  const note = [suggestion.note, renamed].filter(Boolean).join(" ");
  showRunMessage(note ? `${note} ⌘Z restores your program.` : "Example loaded. ⌘Z restores your program.", "info");
  await runCheck();
  if (state.pendingDefault) {
    for (const loop of state.outline?.loops || []) {
      if (!state.setup.decisions[String(loop.id)] || !Object.keys(suggestion.decisions || {}).includes(String(loop.id))) {
        state.setup.decisions[String(loop.id)] = { strategy: state.pendingDefault.strategy, params: { ...(state.pendingDefault.params || {}) } };
      }
    }
    state.pendingDefault = null;
    renderSetupPanel();
    updateAnnotations();
  }
  selectTab("setup");
}

// ---- syntax: pGCL or .ppy ------------------------------------------------------------------------

function renamingText(renamings) {
  if (!renamings?.length) return "";
  const list = renamings.map((r) => `${r.from} → ${r.to}`).join(", ");
  return `Renamed ${list} (keywords in .ppy).`;
}

function renderSyntax() {
  const syntax = editor.getSyntax();
  $("syntax-label").textContent = SYNTAXES[syntax].label;
  $("syntax-label").setAttribute("aria-label", `Syntax: ${SYNTAXES[syntax].label}`);
  $("btn-convert").hidden = syntax !== "pgcl";
  $("btn-save").title = canWriteFiles ? "Save (⌘S) · Save As… (⇧⌘S)"
    : `Download the program as a ${SYNTAXES[syntax].extension} file (⌘S)`;
}

/** The syntax changed (New, Open…, an example, Convert to .ppy, or undoing one of them). */
function syntaxChanged(syntax) {
  save("syntax", syntax);
  renderSyntax();
  documents?.changed();
  renderSetupPanel();
  scheduleCheck(0);
}

/** Convert to .ppy: one undoable change of text, syntax and the invariant programs in the setup
 * (they ride along as the change's companion, so undo and redo restore all three together). */
async function convertProgram() {
  if (editor.getSyntax() !== "pgcl") return;
  const source = editor.getSource();
  let converted;
  try {
    converted = await api.convert(source);
  } catch (error) {
    showRunMessage(error.message);
    if (error.data?.line) editor.revealLine(error.data.line);
    return;
  }
  if (source !== editor.getSource()) return;
  const before = {};
  const after = {};
  for (const [key, decision] of Object.entries(state.setup.decisions)) {
    const invariant = decision.strategy === "invariant" ? decision.params?.invariant : null;
    if (typeof invariant !== "string" || !invariant.trim()) continue;
    try {
      after[key] = (await api.convert(invariant)).source;
      before[key] = invariant;
    } catch {
      // left as it is: the setup reports it when the analysis runs
    }
  }
  if (source !== editor.getSource() || editor.getSyntax() !== "pgcl") return;
  editor.setSource(converted.source, "ppy", { companion: { before, after } });
  const renamed = renamingText(converted.renamings);
  notify(`Converted to .ppy.${renamed ? ` ${renamed}` : ""} ⌘Z undoes it.`, "info");
  editor.focus();
}

/** A conversion was applied, undone or redone: the setup's invariant programs follow. */
function companionApplied(change) {
  if (!swapInvariants(state.setup.decisions, change)) return;
  save("setup", state.setup);
  renderSetupPanel();
}

// ---- boot -----------------------------------------------------------------------------------------------

async function boot() {
  initTabs();
  initDivider();
  try {
    state.meta = await api.meta();
  } catch (error) {
    $("panel-setup").replaceChildren(h("p", { class: "notice error" }, icon("error"), h("span", {}, error.message)));
    return;
  }
  state.strategies = state.meta.strategies;
  state.titles = Object.fromEntries(state.strategies.map((s) => [s.name, s.title]));
  const saved = load("setup", null);
  state.setup = defaultSetup(state.meta);
  if (saved && typeof saved === "object") {
    state.setup.input = saved.input || "";
    state.setup.decisions = saved.decisions || {};
    Object.assign(state.setup.options, saved.options || {});
    if (!state.meta.engines.some((e) => e.name === state.setup.options.engine && e.available)) {
      state.setup.options.engine = defaultSetup(state.meta).options.engine;
    }
    state.setup.options.timeLimit = Math.min(state.setup.options.timeLimit || 120, state.meta.limits.time_limit);
  }

  const storedSource = load("source", null);
  editor = createEditor($("editor"), {
    doc: storedSource ?? WELCOME,
    // a program kept by a version without .ppy support is pGCL
    syntax: load("syntax", storedSource === null ? "ppy" : "pgcl") === "ppy" ? "ppy" : "pgcl",
    onSyntaxChange: syntaxChanged,
    onCompanion: companionApplied,
    onChange: (source) => {
      save("source", source);
      documents?.changed();
      refreshStale();
      scheduleCheck();
    },
    onRun: run,
    onStop: stop,
    onBreakpointsChange: (lines) => {
      state.analysis.breakpoints = lines;
      if (state.result && state.result.steps?.length) renderResultPanel();
    },
  });
  state.analysis.watches = (load("watches", []) || []).filter((w) => typeof w === "string").slice(0, 8);
  save("syntax", editor.getSyntax());
  documents = createDocument({ editor, initialText: editor.getSource(), onReplace: programReplaced, notify,
    onChange: renderTitle });
  documents.changed();

  const library = createLibrary($("library"), { api, onLoad: loadExample });
  $("btn-new").addEventListener("click", () => documents.new());
  $("btn-open").addEventListener("click", () => documents.open());
  $("btn-save").addEventListener("click", () => documents.save());
  $("btn-convert").addEventListener("click", convertProgram);
  renderSyntax();
  $("btn-examples").addEventListener("click", () => library.open());
  renderEngine();
  $("engine-select").addEventListener("change", engineChanged);
  $("btn-run").addEventListener("click", run);
  $("btn-stop").addEventListener("click", stop);
  document.addEventListener("keydown", (event) => {
    if (analysisKey(event)) return;
    const mod = event.metaKey || event.ctrlKey;
    const key = event.key.toLowerCase();
    const modal = document.querySelector("dialog[open]");
    if (mod && event.key === "Enter") { event.preventDefault(); run(); }
    else if (mod && !event.altKey && key === "s") {
      event.preventDefault();
      if (!modal) documents.save({ as: event.shiftKey });
    } else if (mod && !event.altKey && !event.shiftKey && key === "o") {
      event.preventDefault();
      if (!modal) documents.open();
    } else if ((mod && event.key === ".") || (event.key === "Escape" && state.job && !modal)) {
      if (state.job) { event.preventDefault(); stop(); }
    }
  });

  renderProblems();
  renderSetupPanel();
  renderResultPanel();
  await runCheck();
  editor.focus();
}

boot();
