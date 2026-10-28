// Prodigy web GUI: wires the editor, the setup panel, the analysis jobs and the result view.
import { api } from "./api.js";
import { createEditor } from "./editor.js";
import { createLibrary } from "./library.js";
import { defaultDecision, focusLoopCard, renderSetup, setupProblems } from "./setup.js";
import { load, save } from "./store.js";
import { h, icon } from "./components/dom.js";
import { BOUND_TEXT, formatSeconds, formatValue, qualify } from "./components/format.js";
import { renderResult, renderRunning } from "./components/result-view.js";

const WELCOME = `// A fair coin is flipped until it shows heads; c counts the tails.
// Press Analyze (⌘↩). How the loop is analysed is set in Setup.
nat x;
nat c;

x := 1
while (x = 1) {
    {x := 0} [1/2] {c := c + 1}
}

?Pr[c = 0]
?Pr[c < 3]
?Ex[c]
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
  resultStale: false,
  title: load("title", null),
  optionsOpen: false,
};

let editor;
let checkTimer = null;
let checkController = null;
let pollTimer = null;

// ---- tabs ------------------------------------------------------------------------------

function selectTab(name, { focus = false } = {}) {
  for (const tab of ["setup", "result"]) {
    const button = $(`tab-${tab}`);
    const selected = tab === name;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(`panel-${tab}`).hidden = !selected;
    if (selected && focus) button.focus();
  }
}

function initTabs() {
  for (const tab of ["setup", "result"]) {
    $(`tab-${tab}`).addEventListener("click", () => selectTab(tab));
    $(`tab-${tab}`).addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        selectTab(tab === "setup" ? "result" : "setup", { focus: true });
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

function setTitle(title, subtitle = "") {
  state.title = title ? { title, subtitle } : null;
  save("title", state.title);
  renderTitle();
}

function renderTitle() {
  const title = state.title?.title || "Untitled";
  $("doc-title").textContent = title;
  $("doc-subtitle").textContent = state.title?.subtitle || "";
  document.title = `${title} — Prodigy`;
}

// ---- checking -------------------------------------------------------------------------------

function scheduleCheck(delay = 350) {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(runCheck, delay);
}

async function runCheck() {
  const source = editor.getSource();
  checkController?.abort();
  checkController = new AbortController();
  try {
    const result = await api.check(source, checkController.signal);
    if (source !== editor.getSource()) return;
    state.diagnostics = result.diagnostics;
    editor.setDiagnostics(result.diagnostics);
    renderProblems();
    const wasStale = state.outlineStale;
    state.outlineStale = !result.outline;
    if (result.outline) state.outline = result.outline;
    if (result.outline || wasStale !== state.outlineStale) renderSetupPanel();
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
    outlineStale: Boolean(state.outlineStale && state.outline),
    optionsOpen: state.optionsOpen,
    onOptionsToggle: (open) => { state.optionsOpen = open; },
    onChange: () => {
      save("setup", state.setup);
      markStale();
      updateAnnotations();
    },
  });
  save("setup", state.setup);
}

function showLoop(loopId) {
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId);
}

// ---- annotations in the editor ------------------------------------------------------------------

function shortStrategy(decision) {
  const p = decision.params || {};
  switch (decision.strategy) {
    case "unroll": return `Unroll, N = ${p.iterations ?? "?"}`;
    case "mass_threshold": return `Until mass ${p.threshold ?? "?"}`;
    case "evt_iterate": return `EVT iteration, N = ${p.iterations ?? "?"}`;
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
      if (!query.line || !["probability", "expectation"].includes(query.kind)) continue;
      const { glyph, text } = qualify(query, { withValue: true });
      queries.push({ line: query.line, kind: query.bound, glyph: glyph || "", text,
        title: `${(BOUND_TEXT[query.bound] || BOUND_TEXT.unknown).label}: ${query.value ?? ""}` });
    }
  }
  editor.setAnnotations({ loops, queries });
}

// ---- results ----------------------------------------------------------------------------------------

function markStale() {
  if (state.result && !state.resultStale) {
    state.resultStale = true;
    renderResultPanel();
  }
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
    panel.replaceChildren(h("div", { class: "result-view" }, h("p", { class: "help" },
      "No result yet. Press Analyze (⌘↩) to compute the output distribution and the answers to the queries.")));
    setBadge(null);
    return;
  }
  renderResult(panel, state.result, {
    stale: state.resultStale,
    strategyTitles: state.titles,
    onSelectLoop: showLoop,
    onSelectLine: (line) => editor.revealLine(line),
    elapsed: state.result.__elapsed,
  });
  const kind = state.result.status === "ok" ? state.result.bounds?.kind : "error";
  setBadge(state.resultStale ? null : kind);
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

function buildRequest() {
  const options = state.setup.options;
  const decisions = {};
  for (const loop of state.outline?.loops || []) {
    const decision = state.setup.decisions[String(loop.id)];
    if (decision) decisions[String(loop.id)] = { strategy: decision.strategy, params: decision.params };
  }
  return {
    source: editor.getSource(),
    input_distribution: state.setup.input?.trim() || null,
    engine: options.engine,
    normalize: options.normalize,
    simplification: options.simplification,
    decisions,
    coefficients: options.coefficients,
    record_steps: options.recordSteps,
    synthesis_strategy: options.synthesis,
    solver: options.solver,
    time_limit: options.timeLimit,
  };
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
  let response;
  try {
    response = await api.analyze(buildRequest());
  } catch (error) {
    showRunMessage(error.message);
    return;
  }
  state.job = { id: response.job, timeLimit: response.time_limit, logOffset: 0, log: "", started: performance.now() };
  setRunning(true);
  setBadge("running");
  selectTab("result");
  renderRunning($("panel-result"), { elapsed: 0, timeLimit: state.job.timeLimit, progress: [], log: "" }, { strategyTitles: state.titles });
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
    renderRunning($("panel-result"), { elapsed: status.elapsed, timeLimit: status.time_limit, progress: status.progress, log: job.log },
      { strategyTitles: state.titles });
    pollTimer = setTimeout(poll, status.elapsed < 2 ? 150 : 400);
    return;
  }
  finish({ ...(status.result || { status: status.state, error: { message: status.state } }), log: (status.result?.log ?? job.log) }, status.elapsed);
}

function finish(result, elapsed) {
  clearTimeout(pollTimer);
  state.job = null;
  setRunning(false);
  state.result = { ...result, __elapsed: elapsed };
  state.resultStale = false;
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

async function loadExample(example) {
  let data;
  try {
    data = await api.example(example.id);
  } catch (error) {
    showRunMessage(error.message);
    return;
  }
  editor.setSource(data.source);
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
  setTitle(example.title, example.category_title);
  renderResultPanel();
  showRunMessage(suggestion.note ? `${suggestion.note} ⌘Z restores your program.` : "Example loaded. ⌘Z restores your program.", "info");
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

// ---- boot -----------------------------------------------------------------------------------------------

async function boot() {
  initTabs();
  initDivider();
  renderTitle();
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

  editor = createEditor($("editor"), {
    doc: load("source", WELCOME),
    onChange: (source) => {
      save("source", source);
      if (state.title && !state.title.subtitle?.endsWith("edited")) {
        setTitle(state.title.title, `${state.title.subtitle ? `${state.title.subtitle} · ` : ""}edited`);
      }
      markStale();
      scheduleCheck();
    },
    onRun: run,
    onStop: stop,
  });

  const library = createLibrary($("library"), { api, onLoad: loadExample });
  $("btn-examples").addEventListener("click", () => library.open());
  $("btn-run").addEventListener("click", run);
  $("btn-stop").addEventListener("click", stop);
  document.addEventListener("keydown", (event) => {
    const mod = event.metaKey || event.ctrlKey;
    if (mod && event.key === "Enter") { event.preventDefault(); run(); }
    else if ((mod && event.key === ".") || (event.key === "Escape" && state.job && !$("library").open)) {
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
