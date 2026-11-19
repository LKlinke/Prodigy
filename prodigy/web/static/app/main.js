// Prodigy web GUI: wires the editor, the setup panel, the analysis jobs and the result view.
import { api } from "./api.js";
import { canWriteFiles, createDocument } from "./document.js";
import { createEditor, SYNTAXES } from "./editor.js";
import { createLibrary } from "./library.js";
import { applyInputProblem, defaultDecision, focusLoopCard, renderSetup, rekeyDecisions, setupProblems,
  swapInvariants } from "./setup.js";
import { changeSince } from "./staleness.js";
import { SETUP_KEYS, applyCompareDefaults, compareDefault, requestDecisions, restoreSetup, setupInput } from "./mode-setup.js";
import { load, save } from "./store.js";
import { h, icon } from "./components/dom.js";
import { BOUND_TEXT, formatSeconds, formatValue, qualify } from "./components/format.js";
import { liveChartStates, renderResult, renderRunning } from "./components/result-view.js";
import { renderEquivalence } from "./components/equivalence-view.js";
import { continueDecision, continueOffer } from "./components/equivalence-model.js";

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
  graveyard: load("graveyard", {}), // decisions of loops that are gone for now, by loop signature (#277); kept across reloads
  diagnostics: [],
  // each mode keeps its own setup (loop decisions, options; Analyze also the input); only the programs are shared
  // (mode-setup.js)
  setups: { analyze: null, compare: null },
  /** The setup of the mode that is shown. */
  get setup() { return this.setups[this.mode]; },
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
  // the mode of the window (#154): "analyze" one program or "compare" two; each mode keeps its own result
  mode: "analyze",
  results: { analyze: null, compare: null },
  // program B of Compare mode: diagnostics, outline and loop decisions of its own, and which column a narrow
  // editor pane shows
  compare: { diagnostics: [], outline: null, outlineStale: false, decisions: {}, graveyard: {}, active: "first",
    // decisions of program A's loops in Compare that are gone for now (Analyze has state.graveyard)
    graveyardA: {},
    // "Compare this loop" (#399): the program the loop was taken from ({source, syntax, decisions, input, loop,
    // program, programLoop, whole}), restored when going back to Analyze; kept across reloads
    origin: null,
    // program B as it was before its last edit: invariants of program A's loops that equal it follow its edits
    lastOther: "" },
};

/** Strategies Analyze mode does not offer: checking a loop against a program is a comparison (#399). */
const COMPARE_ONLY = new Set(["invariant"]);

const TABS = ["setup", "result"];

let editor;
let otherEditor;
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
  forgetOrigin();
  // the decisions and inputs of both modes belonged to the program that is gone
  for (const mode of Object.keys(state.setups)) {
    state.setups[mode].decisions = {};
    state.setups[mode].input = "";
    saveSetup(mode);
  }
  state.pendingDefault = null;
  state.graveyard = {};
  state.compare.graveyardA = {};
  state.outline = null;
  state.result = null;
  state.resultStale = false;
  state.resultChange = null;
  resetAnalysis();
  saveSetup();
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
    const result = await api.check(source, syntax, checkController.signal, inputToCheck());
    if (source !== editor.getSource() || syntax !== editor.getSyntax()) return;
    showInputProblem(result);
    state.diagnostics = result.diagnostics;
    editor.setDiagnostics(result.diagnostics);
    renderProblems();
    const wasStale = state.outlineStale;
    state.outlineStale = !result.outline;
    if (result.outline) {
      // decisions follow their loops when loops are inserted, removed or commented out (#277); program A is
      // shared, so the decisions of both modes follow it
      if (rekeyDecisions(state.setups.analyze.decisions, state.outline, result.outline, state.graveyard)) {
        saveSetup("analyze");
        save("graveyard", state.graveyard);
      }
      if (rekeyDecisions(state.setups.compare.decisions, state.outline, result.outline, state.compare.graveyardA)) {
        saveSetup("compare");
      }
      state.outline = result.outline;
      if (state.mode === "compare") compareDefaults();
    }
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

/** The initial distribution the server should check along with the program ({} if none). */
function inputToCheck() {
  const text = setupInput(state.setup, state.mode);
  return text ? { input_distribution: text, engine: state.setup.options.engine } : {};
}

/** Shows at the Setup field whether the server refused the initial distribution (#307), without
 * re-rendering the panel (the person is typing in it). */
function showInputProblem(result) {
  const check = result?.input_distribution;
  state.inputProblem = check?.status === "refused" ? check.message
    : check?.status === "timeout" ? "Could not check yet; it will be checked when you analyze." : "";
  const field = $("panel-setup").querySelector(".input-distribution");
  if (field) applyInputProblem(field, state.inputProblem);
}

/** An input-distribution check of its own: the program is cached on the server, so typing in the
 * field does not re-parse it, and the outline and diagnostics stay as they are. */
let inputCheckTimer = null;
let inputController = null;
function scheduleInputCheck() {
  clearTimeout(inputCheckTimer);
  inputController?.abort();
  if (!setupInput(state.setup, state.mode)) { showInputProblem(null); return; }
  inputCheckTimer = setTimeout(async () => {
    inputController = new AbortController();
    const source = editor.getSource();
    try {
      const result = await api.check(source, editor.getSyntax(), inputController.signal, inputToCheck());
      if (source === editor.getSource()) showInputProblem(result);
    } catch (error) {
      if (error.name !== "AbortError") showInputProblem(null);
    }
  }, 500);
}

function renderProblems(failure = null) {
  renderProblemsButton($("problems"), state.diagnostics, editor, failure);
}

function renderProblemsButton(button, diagnostics, target, failure = null) {
  const problems = diagnostics.filter((d) => d.severity !== "info");
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
  button.onclick = () => {
    if (target === otherEditor) showColumn("second");
    target.revealLine(first.line);
  };
}

// ---- modes: Analyze one program, Compare two (#154) ----------------------------------------------------

const MODES = ["analyze", "compare"];

/** The mode the URL asks for: `…/compare` is Compare, everything else (`/`, `/analyze`) Analyze. */
function modeFromUrl() {
  return /\/compare\/?$/.test(window.location.pathname) ? "compare" : "analyze";
}

/**
 * Switches the window to `mode`. Program A (the first editor, its setup) stays; each mode keeps its own
 * result. `history` is "push" (a choice of the person), "replace" or null (the URL already says it).
 * While a job runs the mode stays (its result belongs to the mode it was started in); returns whether it
 * switched. Going back to Analyze from a loop opened with "Compare this loop" brings back the program it came
 * from (see restoreOrigin), with `continueWith` as the decision of that loop if given.
 */
function setMode(mode, { focus = false, history: how = "push", continueWith = null } = {}) {
  if (!MODES.includes(mode)) mode = "analyze";
  if (state.job && mode !== state.mode) {
    notify(`Stop the running ${state.job.comparing ? "check" : "analysis"} before switching to ${mode === "compare" ? "Compare" : "Analyze"}.`);
    return false;
  }
  const leavingCompare = state.mode === "compare" && mode === "analyze";
  if (mode !== state.mode) {
    state.results[state.mode] = { result: state.result };
    state.result = state.results[mode]?.result || null;
    state.analysis.card?.stop();
    state.analysis.card = null;
    state.analysis.step = null;
    editor.showStep(null);
  }
  state.mode = mode;
  if (leavingCompare) restoreOrigin(continueWith);
  const compare = mode === "compare";
  $("editor-pane").dataset.compare = String(compare);
  $("column-second").hidden = !compare;
  $("compare-tabs").hidden = !compare;
  $("program-heading").textContent = compare ? "Program A" : "Program";
  for (const key of MODES) {
    const button = $(`mode-${key}`);
    button.setAttribute("aria-checked", String(key === mode));
    button.tabIndex = key === mode ? 0 : -1;
  }
  if (compare && !otherEditor.getSource().trim()) {
    // a copy of program A is the usual start: change it and check
    otherEditor.setSource(editor.getSource(), editor.getSyntax());
  }
  if (!compare) showColumn("first");
  if (how) {
    const url = new URL(compare ? "compare" : "./", document.baseURI);
    url.search = window.location.search;
    url.hash = window.location.hash;
    if (url.href !== window.location.href) window.history[how === "push" ? "pushState" : "replaceState"]({ mode }, "", url);
  }
  renderRunButton();
  // the engine belongs to the setup of the mode
  if (state.meta) $("engine-select").value = state.setup.options.engine;
  if (compare && state.outline && state.compare.outline) compareDefaults();
  state.inputProblem = "";
  scheduleInputCheck();
  state.resultChange = state.result ? changeSince(state.result.__request, buildRequest()) : null;
  state.resultStale = Boolean(state.resultChange);
  renderSetupPanel();
  renderResultPanel();
  updateAnnotations();
  if (compare) scheduleOtherCheck(0);
  if (focus) $(`mode-${mode}`).focus();
  return true;
}

function initModeSwitch() {
  for (const mode of MODES) {
    const button = $(`mode-${mode}`);
    button.addEventListener("click", () => setMode(mode));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      // a radio group: the arrow keys move the choice
      setMode(mode === "analyze" ? "compare" : "analyze", { focus: true });
    });
  }
  window.addEventListener("popstate", () => {
    const wanted = modeFromUrl();
    if (wanted !== state.mode && !setMode(wanted, { history: null })) {
      // a job runs: the URL goes back to the mode that is shown
      window.history.pushState({ mode: state.mode }, "", new URL(state.mode === "compare" ? "compare" : "./", document.baseURI));
    }
  });
}

/** ⌥⌘1 / ⌥⌘2 (Ctrl+Alt+1 / 2 elsewhere) choose the mode; returns whether the key was one of them. */
function modeKey(event) {
  if (!event.altKey || !(event.metaKey || event.ctrlKey) || event.shiftKey) return false;
  const mode = { Digit1: "analyze", Digit2: "compare" }[event.code];
  if (!mode) return false;
  event.preventDefault();
  setMode(mode, { focus: true });
  return true;
}

function renderRunButton() {
  const compare = state.mode === "compare";
  $("btn-run").querySelector("span").textContent = compare ? "Check" : "Analyze";
  $("btn-run").title = compare ? "Check whether programs A and B are equivalent (⌘↩)" : "Analyze the program (⌘↩)";
  $("btn-stop").title = compare ? "Stop the check (⌘.)" : "Stop the analysis (⌘.)";
}

/** Which program a narrow editor pane shows (the tabs above the editors). */
function showColumn(which) {
  state.compare.active = which;
  $("editor-pane").dataset.active = which;
  for (const key of ["first", "second"]) {
    const tab = $(`tab-${key}`);
    tab.setAttribute("aria-selected", String(key === which));
    tab.tabIndex = key === which ? 0 : -1;
  }
}

function initCompareTabs() {
  for (const key of ["first", "second"]) {
    const tab = $(`tab-${key}`);
    tab.addEventListener("click", () => {
      showColumn(key);
      (key === "first" ? editor : otherEditor).focus();
    });
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const next = key === "first" ? "second" : "first";
      showColumn(next);
      $(`tab-${next}`).focus();
    });
  }
}

let otherCheckTimer = null;
let otherCheckController = null;

function scheduleOtherCheck(delay = 350) {
  clearTimeout(otherCheckTimer);
  otherCheckTimer = setTimeout(runOtherCheck, delay);
}

/** Diagnostics and outline of program B, as for program A (syntax and type errors, language warnings, loops). */
async function runOtherCheck() {
  if (state.mode !== "compare") return;
  const source = otherEditor.getSource();
  const syntax = otherEditor.getSyntax();
  otherCheckController?.abort();
  otherCheckController = new AbortController();
  try {
    const result = await api.check(source, syntax, otherCheckController.signal);
    if (source !== otherEditor.getSource() || syntax !== otherEditor.getSyntax()) return;
    state.compare.diagnostics = result.diagnostics;
    otherEditor.setDiagnostics(result.diagnostics);
    renderProblemsButton($("other-problems"), result.diagnostics, otherEditor);
    const compare = state.compare;
    const wasStale = compare.outlineStale;
    compare.outlineStale = !result.outline;
    if (result.outline) {
      if (rekeyDecisions(compare.decisions, compare.outline, result.outline, compare.graveyard)) {
        save("compareDecisionsB", compare.decisions);
      }
      const changed = JSON.stringify(compare.outline?.loops || null) !== JSON.stringify(result.outline.loops);
      compare.outline = result.outline;
      if (compareDefaults() || changed) renderSetupPanel();
      refreshStale();
    } else if (wasStale !== compare.outlineStale) {
      renderSetupPanel();
    }
    updateAnnotations();
  } catch (error) {
    if (error.name === "AbortError") return;
    state.compare.diagnostics = [];
    renderProblemsButton($("other-problems"), [], otherEditor, error.message);
  }
}

function renderOtherSyntax() {
  const syntax = otherEditor.getSyntax();
  $("other-syntax-label").textContent = SYNTAXES[syntax].label;
  $("other-syntax-label").setAttribute("aria-label", `Syntax of program B: ${SYNTAXES[syntax].label}`);
}

// ---- setup ------------------------------------------------------------------------------------

/** Keeps the setup of the shown mode (or of `mode`) in the browser. */
function saveSetup(mode = state.mode) {
  if (state.setups[mode]) save(SETUP_KEYS[mode], state.setups[mode]);
}

/** Compare: every loop of programs A and B without a chosen strategy gets the Compare default (mode-setup.js);
 * defaults follow the programs. Returns whether a decision changed. */
function compareDefaults() {
  const changed = applyCompareDefaults(
    { outline: state.outline, source: editor.getSource(), decisions: state.setups.compare.decisions },
    { outline: state.compare.outline, source: otherEditor.getSource(), decisions: state.compare.decisions },
    state.strategies);
  if (changed) {
    saveSetup("compare");
    save("compareDecisionsB", state.compare.decisions);
  }
  return changed;
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
  // a deprecated engine (SymEngine, #416): say so when it is chosen; every result of it carries the warning
  const deprecation = state.meta.engines.find((e) => e.name === state.setup.options.engine)?.deprecation;
  if (deprecation) notify(deprecation, "info");
  saveSetup();
  scheduleInputCheck();
  refreshStale();
  updateAnnotations();
}

function showLoop(loopId, { field = false } = {}) {
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId, { field });
}

/** "Use as EVT invariant candidate" of an exploration: the loop switches to Check EVT invariant with the
 * candidate, and the Setup shows its field to review or edit before analysing again (#324). */
function useEvtCandidate(loopId, candidate) {
  state.setup.decisions[String(loopId)] = { strategy: "evt_invariant", params: { invariant: candidate } };
  saveSetup();
  refreshStale();
  renderSetupPanel();
  updateAnnotations();
  selectTab("setup");
  focusLoopCard($("panel-setup"), loopId, { field: true });
}

/** "Open in Compare" of a parametric invariant (#154): the loop becomes program A (the program itself if
 * the loop is all it does) with the invariant as the strategy of that loop, the invariant becomes program B. */
function openInCompare(details) {
  if (!details?.invariant || state.job) return;
  const syntax = editor.getSyntax();
  const source = editor.getSource();
  const loop = details.loop ?? details.program_loop ?? 0;
  if (details.program && state.mode === "analyze") {
    rememberOrigin({ source, syntax, loop, program: details.program, programLoop: details.program_loop ?? loop,
      whole: details.program === source });
  }
  if (!setMode("compare")) return;
  const decision = { strategy: "invariant", params: { invariant: details.invariant } };
  if (details.program && details.program !== editor.getSource()) {
    // undoable (⌘Z); the decisions belong to the loops of the new program
    editor.setSource(details.program, syntax);
    state.setup.decisions = {};
    state.outline = null;
    state.compare.graveyardA = {};
  }
  state.setup.decisions[String(details.program_loop ?? details.loop ?? 0)] = decision;
  saveSetup();
  setOtherSource(details.invariant, syntax);
  state.compare.decisions = {};
  state.compare.outline = null;
  showColumn("first");
  renderSetupPanel();
  refreshStale();
  updateAnnotations();
  scheduleCheck(0);
  scheduleOtherCheck(0);
  notify("Loop and invariant opened in Compare as programs A and B. Press Check (⌘↩) to find the parameter values.");
}

// ---- a loop against a loop-free program: Compare this loop, Continue in Analyze (#399) ---------------------

/** Program B gets `source`; invariants of program A's loops that were program B do not follow this. */
function setOtherSource(source, syntax) {
  state.compare.lastOther = source;
  otherEditor.setSource(source, syntax);
  state.compare.lastOther = otherEditor.getSource();
}

/** Remembers the program a loop was taken from (the current program A, its decisions and input). */
function rememberOrigin({ source, syntax, loop, program, programLoop, whole }) {
  state.compare.origin = { source, syntax, decisions: structuredClone(state.setup.decisions), input: state.setup.input || "",
    loop, program, programLoop, whole };
  save("compareOrigin", state.compare.origin);
}

function forgetOrigin() {
  state.compare.origin = null;
  save("compareOrigin", null);
}

/** Whether program A is still the loop that "Compare this loop" took out of a program. */
function originActive() {
  const origin = state.compare.origin;
  return Boolean(origin) && editor.getSource() === origin.program && editor.getSyntax() === origin.syntax;
}

/**
 * Back in Analyze after "Compare this loop": if program A is still that loop, the program it came from comes back
 * (undoable with ⌘Z), with its decisions and initial distribution, and `decision` for the loop if given (the
 * continue flow). If program A was edited into something else, it stays as it is.
 */
function restoreOrigin(decision) {
  const origin = state.compare.origin;
  if (!originActive()) return;
  forgetOrigin();
  if (!origin.whole) {
    editor.setSource(origin.source, origin.syntax);
    state.outline = null;
    state.graveyard = {};
  }
  state.setup.decisions = origin.decisions || {};
  if (decision) state.setup.decisions[String(origin.loop)] = decision;
  state.setup.input = origin.input || "";
  saveSetup();
  scheduleCheck(0);
}

/**
 * "Compare this loop" of a loop card in Analyze: the loop becomes program A of Compare mode, on its own (with the
 * declarations; the code before it is left out because a comparison holds for every input, see
 * prodigy.analysis.session.isolate_loop), and program B is the loop-free program it continues with, or the
 * declarations to write one. The program comes back when going back to Analyze.
 */
async function compareLoop(loopId) {
  if (state.job) { notify(`Stop the running ${state.job.comparing ? "check" : "analysis"} first.`); return; }
  const source = editor.getSource();
  const syntax = editor.getSyntax();
  let isolated;
  try {
    isolated = await api.loopProgram(source, syntax, loopId);
  } catch (error) {
    showRunMessage(error.message);
    return;
  }
  if (source !== editor.getSource() || syntax !== editor.getSyntax() || state.mode !== "analyze") return;
  // only a loop-free program the loop continues with comes along (as program B); the strategy chosen for the
  // analysis stays in Analyze, and Compare gives the loop its own default (mode-setup.js: compareDefault)
  const current = state.setup.decisions[String(loopId)];
  const invariant = current?.strategy === "invariant" ? current.params?.invariant || "" : "";
  rememberOrigin({ source, syntax, loop: loopId, program: isolated.program, programLoop: isolated.loop,
    whole: isolated.whole });
  // program B before the switch (an empty one would become a copy of program A)
  setOtherSource(invariant || isolated.declarations
    || (syntax === "ppy" ? "# Program B: a loop-free program\n" : "// Program B: a loop-free program\n"), syntax);
  state.compare.decisions = {};
  state.compare.outline = null;
  state.compare.graveyard = {};
  if (!setMode("compare")) { forgetOrigin(); return; }
  if (!isolated.whole) {
    editor.setSource(isolated.program, syntax);
    state.outline = null;
    state.compare.graveyardA = {};
  }
  // the loop starts with the Compare default; Compare has no input (the check covers every input)
  state.setup.decisions = {};
  saveSetup();
  state.result = null;
  state.resultStale = false;
  state.resultChange = null;
  renderResultPanel();
  renderSetupPanel();
  updateAnnotations();
  scheduleCheck(0);
  scheduleOtherCheck(0);
  selectTab("setup");
  const what = isolated.whole ? `Loop ${loopId} is program A` : `Loop ${loopId} is program A, on its own`;
  if (invariant) {
    showColumn("first");
    $("btn-run").focus();
    notify(`${what}; its loop-free program is program B. Press Check (⌘↩). Analyze brings your program back.`);
  } else {
    showColumn("second");
    // the cursor after the declarations, where program B's instructions go
    otherEditor.revealLine(otherEditor.getSource().split("\n").length);
    notify(`${what}. Write the loop-free program B and press Check (⌘↩). Analyze brings your program back.`);
  }
}

/** What the verdict card offers for continuing in Analyze (see continueOffer), or null. */
function continueTarget() {
  const result = state.result;
  if (state.mode !== "compare" || !result?.equivalence || state.resultStale) return null;
  const loops = state.outline?.loops || [];
  const offer = continueOffer(result.equivalence, { loopsOfA: loops.length, loopsOfB: (state.compare.outline?.loops || []).length });
  if (!offer?.guarantee) return offer;
  const origin = originActive() ? state.compare.origin : null;
  const loopInA = loops[0].id;
  const target = origin && !origin.whole ? `It replaces loop ${origin.loop} of your program; the code before the loop runs as before.`
    : "Program A runs with it in Analyze.";
  return { guarantee: offer.guarantee, target, onContinue: () => continueInAnalyze(offer.guarantee, loopInA) };
}

/** "Continue in Analyze with B as the loop's result": the loop's decision follows the verdict (see
 * continueDecision), the program the loop came from comes back, and Analyze runs. */
function continueInAnalyze(guarantee, loopInA) {
  const checked = state.result?.__request?.decisions?.[String(loopInA)] || state.setup.decisions[String(loopInA)];
  const decision = continueDecision(guarantee, otherEditor.getSource(), checked);
  const fromOrigin = originActive();
  if (!setMode("analyze", { continueWith: fromOrigin ? decision : null })) return;
  if (!fromOrigin) {
    state.setup.decisions[String(loopInA)] = decision;
    saveSetup();
  }
  renderSetupPanel();
  updateAnnotations();
  run();
}

/** In Compare mode, a loop of program A whose invariant is program B (as "Compare this loop" and the library set
 * it up) keeps checking program B while it is edited. */
function followProgramB(source) {
  const before = state.compare.lastOther;
  state.compare.lastOther = source;
  if (state.mode !== "compare" || before === source) return;
  let changed = false;
  for (const decision of Object.values(state.setup.decisions)) {
    if (decision?.strategy === "invariant" && decision.params?.invariant === before) {
      decision.params.invariant = source;
      changed = true;
    }
  }
  if (changed) {
    saveSetup();
    const field = [...$("panel-setup").querySelectorAll("textarea")].find((t) => t.value === before);
    if (field) field.value = source;
  }
}

/** Why saving is not possible right now: program A is a loop taken out of the program, so saving would write only
 * the loop into the file (null if saving is fine). */
function saveBlocked() {
  if (state.mode !== "compare" || !originActive() || state.compare.origin.whole) return null;
  return "Program A is one loop of your program. Switch to Analyze (⌥⌘1) to get your program back, then save.";
}

// ---- annotations in the editor ------------------------------------------------------------------

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

function updateAnnotations() {
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
  error: "Analysis failed", running: "Analysis running", equivalent: "Equivalent", not_equivalent: "Not equivalent",
  undecided: "Equivalence not decided" };

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

/** Runs a grid job for two variables of the result of analysis `jobId` and resolves to the grid;
 * `onPartial(stream)` gets the cells computed so far on each poll (#405). */
async function loadJoint(jobId, payload, signal, onPartial = null) {
  const key = `${jobId} ${JSON.stringify(payload)}`;
  if (jointCache.has(key)) return jointCache.get(key);
  const response = await api.joint(jobId, payload);
  for (;;) {
    if (signal.aborted) {
      api.cancel(response.job).catch(() => {});
      throw new DOMException("cancelled", "AbortError");
    }
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

// ---- running --------------------------------------------------------------------------------------

function setRunning(running) {
  $("btn-run").hidden = running;
  $("btn-stop").hidden = !running;
  if (!running) $("run-status").replaceChildren();
}

let runMessageCounter = 0;

function showRunMessage(message, tone = "error") {
  const status = $("run-status");
  const text = h("span", { class: "run-message-text" }, message);
  const outcome = h("span", { class: tone === "error" ? "outcome" : "", "data-kind": tone === "error" ? "error" : null },
    icon(tone === "error" ? "error" : "info"), " ", text);
  const wrap = h("span", { class: "run-message" }, outcome);
  status.replaceChildren(wrap);
  // a long message is cut off in the toolbar; the whole text opens in a popover that the keyboard reaches
  // (a `title` tooltip does not, #426)
  requestAnimationFrame(() => {
    if (wrap.isConnected && text.scrollWidth > text.clientWidth) addMessageDisclosure(wrap, message);
  });
}

function addMessageDisclosure(wrap, message) {
  const id = `run-message-${++runMessageCounter}`;
  const popover = h("div", { class: "run-message-popover", id, hidden: true, tabindex: "-1" }, message);
  const button = h("button", { type: "button", class: "link-button run-message-more", "aria-expanded": "false",
    "aria-controls": id }, "Show full message");
  const close = (focus) => {
    popover.hidden = true;
    button.setAttribute("aria-expanded", "false");
    button.textContent = "Show full message";
    delete wrap.dataset.open;
    document.removeEventListener("pointerdown", onOutside, true);
    if (focus) button.focus();
  };
  const onOutside = (event) => { if (!wrap.contains(event.target)) close(false); };
  button.addEventListener("click", () => {
    if (!popover.hidden) { close(false); return; }
    popover.hidden = false;
    button.setAttribute("aria-expanded", "true");
    button.textContent = "Hide full message";
    wrap.dataset.open = "";
    document.addEventListener("pointerdown", onOutside, true);
  });
  wrap.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !popover.hidden) { event.stopPropagation(); close(true); }
  });
  wrap.append(" ", button, popover);
}

/** A short notice (opened, saved); information disappears after a few seconds. */
function notify(message, tone = "info") {
  showRunMessage(message, tone);
  const shown = $("run-status").firstChild;
  if (tone === "info") {
    setTimeout(() => {
      if (!state.job && $("run-status").firstChild === shown && !("open" in shown.dataset)) {
        $("run-status").replaceChildren();
      }
    }, 5000);
  }
}

function buildRequest() {
  const options = state.setup.options;
  if (state.mode === "compare") {
    // the check for every input: no initial distribution, and only the options the check uses
    return {
      source: editor.getSource(),
      syntax: editor.getSyntax(),
      engine: options.engine,
      decisions: requestDecisions(state.setup.decisions, state.outline),
      time_limit: options.timeLimit,
      other_source: otherEditor.getSource(),
      other_syntax: otherEditor.getSyntax(),
      other_decisions: requestDecisions(state.compare.decisions, state.compare.outline),
    };
  }
  const steps = Boolean(options.recordSteps);
  return {
    source: editor.getSource(),
    syntax: editor.getSyntax(),
    input_distribution: setupInput(state.setup, state.mode),
    engine: options.engine,
    normalize: options.normalize,
    simplification: options.simplification,
    decisions: requestDecisions(state.setup.decisions, state.outline),
    coefficients: options.coefficients,
    record_steps: steps,
    ...(steps && state.analysis.watches.length ? { watches: state.analysis.watches } : {}),
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
    showRunMessage(state.mode === "compare" ? `Fix line ${errors[0].line} of program A first.` : `Fix line ${errors[0].line} first.`);
    if (state.mode === "compare") showColumn("first");
    editor.revealLine(errors[0].line);
    return;
  }
  const problems = setupProblems(state.outline, state.setup, state.strategies, editor.getSyntax());
  if (problems.length) {
    showRunMessage(state.mode === "compare" ? `Program A: ${problems[0].message}` : problems[0].message);
    showLoop(problems[0].loop, { field: Boolean(problems[0].param) });
    return;
  }
  const comparing = state.mode === "compare";
  if (comparing) {
    clearTimeout(otherCheckTimer);
    await runOtherCheck();
    const otherErrors = state.compare.diagnostics.filter((d) => d.severity === "error");
    if (!otherEditor.getSource().trim()) {
      showRunMessage("Program B is empty.");
      showColumn("second");
      otherEditor.focus();
      return;
    }
    if (otherErrors.length) {
      showRunMessage(`Fix line ${otherErrors[0].line} of program B first.`);
      showColumn("second");
      otherEditor.revealLine(otherErrors[0].line);
      return;
    }
    const otherProblems = setupProblems(state.compare.outline, { decisions: state.compare.decisions }, state.strategies,
      otherEditor.getSyntax());
    if (otherProblems.length) {
      showRunMessage(`Program B: ${otherProblems[0].message}`);
      showLoop(`b-${otherProblems[0].loop}`, { field: Boolean(otherProblems[0].param) });
      return;
    }
  }
  const request = buildRequest();
  let response;
  try {
    response = await (comparing ? api.equivalence(request) : api.analyze(request));
  } catch (error) {
    showRunMessage(error.message);
    return;
  }
  state.job = { id: response.job, request, timeLimit: response.time_limit, logOffset: 0, log: "", started: performance.now(),
    comparing };
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
    showRunning({ elapsed: status.elapsed, timeLimit: status.time_limit, progress: status.progress, log: job.log,
      stream: status.stream });
    pollTimer = setTimeout(poll, status.elapsed < 2 ? 150 : 400);
    return;
  }
  finish({ ...(status.result || { status: status.state, error: { message: status.state } }), log: (status.result?.log ?? job.log),
    __job: job.id }, status.elapsed);
}

/** The progress view in the Result panel. */
function showRunning(progress) {
  const change = state.job ? changeSince(state.job.request, buildRequest()) : null;
  renderRunning($("panel-result"), progress, { strategyTitles: state.titles, change,
    label: state.job?.comparing ? "Checking" : "Analyzing" });
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
  state.liveCharts = result.status === "ok" && !result.equivalence ? liveChartStates($("panel-result")) : null;
  renderResultPanel();
  updateAnnotations();
  // a comparison shows its verdict and why in the result
  const message = result.status === "ok" || result.equivalence ? null : (result.error?.message || result.status);
  // parameter synthesis in Analyze is no failure but a pointer to Compare (#154)
  const neutral = result.status === "cancelled" || result.error?.details?.type === "parameter_synthesis";
  if (message) showRunMessage(message, neutral ? "info" : "error");
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
  forgetOrigin();
  editor.setSource(data.source, data.syntax || "ppy");
  if (data.other) {
    // an equivalence example: the second program goes into the second editor (#154)
    setOtherSource(data.other.source, data.other.syntax || "ppy");
    state.compare.decisions = {};
    state.compare.outline = null;
    state.compare.graveyard = {};
    setMode("compare");
    showColumn("first");
  } else {
    setMode("analyze");
  }
  const suggestion = data.suggestion || {};
  state.setup.decisions = {};
  for (const [key, decision] of Object.entries(suggestion.decisions || {})) {
    if (key === "*") continue;
    state.setup.decisions[key] = { strategy: decision.strategy, params: { ...(decision.params || {}) } };
  }
  state.pendingDefault = suggestion.decisions?.["*"] || null;
  // Compare checks every input: an input the example suggests is Analyze's
  if (state.mode === "analyze") state.setup.input = suggestion.input_distribution || "";
  // the suggested decisions belong to the example's loops: they must not be re-keyed from the loops of the
  // program it replaces (#277 follows edits, not replacements)
  state.outline = null;
  state.graveyard = {};
  state.compare.graveyardA = {};
  save("graveyard", state.graveyard);
  saveSetup();
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
  saveSetup();
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
  // each mode has its own setup (mode-setup.js); a setup kept by an earlier version is Analyze's
  for (const mode of Object.keys(state.setups)) state.setups[mode] = restoreSetup(load(SETUP_KEYS[mode], null), state.meta, mode);

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
  otherEditor = createEditor($("editor-other"), {
    doc: load("otherSource", "") || "",
    syntax: load("otherSyntax", "ppy") === "pgcl" ? "pgcl" : "ppy",
    label: "Second program",
    breakpoints: false,
    onSyntaxChange: (syntax) => { save("otherSyntax", syntax); renderOtherSyntax(); scheduleOtherCheck(0); refreshStale(); },
    onChange: (source) => {
      save("otherSource", source);
      followProgramB(source);
      refreshStale();
      scheduleOtherCheck();
    },
    onRun: run,
    onStop: stop,
  });
  renderOtherSyntax();
  initCompareTabs();
  initModeSwitch();
  // (a new key: what an earlier version kept there were the defaults of Analyze, e.g. "auto")
  state.compare.decisions = load("compareDecisionsB", {}) || {};
  state.compare.origin = load("compareOrigin", null);
  state.compare.lastOther = otherEditor.getSource();
  state.analysis.watches = (load("watches", []) || []).filter((w) => typeof w === "string").slice(0, 8);
  save("syntax", editor.getSyntax());
  documents = createDocument({ editor, initialText: editor.getSource(), onReplace: programReplaced, notify,
    onChange: renderTitle, saveBlocked });
  documents.changed();

  const library = createLibrary($("library"), { api, onLoad: loadExample, onLoadFile: (entry) => documents.openRecentFile(entry) });
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
    if (analysisKey(event) || modeKey(event)) return;
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
  setMode(modeFromUrl(), { history: null });
  await runCheck();
  editor.focus();
}

boot();
