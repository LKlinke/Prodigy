// The two modes of the window (#154): switching between Analyze and Compare, program B, and the ways a loop
// moves between them (Open in Compare, Compare this loop, Continue in Analyze, #399).
import { api } from "./api.js";
import { $, editor, otherEditor, saveSetup, state } from "./app-state.js";
import { renderProblemsButton, scheduleCheck, scheduleInputCheck } from "./checking.js";
import { SYNTAXES } from "./editor.js";
import { applyCompareDefaults } from "./mode-setup.js";
import { renderSetupPanel, selectTab, updateAnnotations } from "./panels.js";
import { refreshStale, renderResultPanel } from "./result-panel.js";
import { buildRequest, notify, run, showRunMessage } from "./run.js";
import { rekeyDecisions } from "./setup.js";
import { changeSince } from "./staleness.js";
import { save } from "./store.js";
import { continueDecision, continueOffer } from "./components/equivalence-model.js";

const MODES = ["analyze", "compare"];

/** The mode the URL asks for: `…/compare` is Compare, everything else (`/`, `/analyze`) Analyze. */
export function modeFromUrl() {
  return /\/compare\/?$/.test(window.location.pathname) ? "compare" : "analyze";
}

/**
 * Switches the window to `mode`. Program A (the first editor, its setup) stays; each mode keeps its own
 * result. `history` is "push" (a choice of the person), "replace" or null (the URL already says it).
 * While a job runs the mode stays (its result belongs to the mode it was started in); returns whether it
 * switched. Going back to Analyze from a loop opened with "Compare this loop" brings back the program it came
 * from (see restoreOrigin), with `continueWith` as the decision of that loop if given.
 */
export function setMode(mode, { focus = false, history: how = "push", continueWith = null } = {}) {
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

export function initModeSwitch() {
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
export function modeKey(event) {
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
export function showColumn(which) {
  state.compare.active = which;
  $("editor-pane").dataset.active = which;
  for (const key of ["first", "second"]) {
    const tab = $(`tab-${key}`);
    tab.setAttribute("aria-selected", String(key === which));
    tab.tabIndex = key === which ? 0 : -1;
  }
}

export function initCompareTabs() {
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

export function scheduleOtherCheck(delay = 350) {
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

/** Checks program B now (before a check). */
export function checkOtherNow() {
  clearTimeout(otherCheckTimer);
  return runOtherCheck();
}

export function renderOtherSyntax() {
  const syntax = otherEditor.getSyntax();
  $("other-syntax-label").textContent = SYNTAXES[syntax].label;
  $("other-syntax-label").setAttribute("aria-label", `Syntax of program B: ${SYNTAXES[syntax].label}`);
}

/** Compare: every loop of programs A and B without a chosen strategy gets the Compare default (mode-setup.js);
 * defaults follow the programs. Returns whether a decision changed. */
export function compareDefaults() {
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

/** "Open in Compare" of a parametric invariant (#154): the loop becomes program A (the program itself if
 * the loop is all it does) with the invariant as the strategy of that loop, the invariant becomes program B. */
export function openInCompare(details) {
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
export function setOtherSource(source, syntax) {
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

export function forgetOrigin() {
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
export async function compareLoop(loopId) {
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
export function continueTarget() {
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
export function followProgramB(source) {
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
export function saveBlocked() {
  if (state.mode !== "compare" || !originActive() || state.compare.origin.whole) return null;
  return "Program A is one loop of your program. Switch to Analyze (⌥⌘1) to get your program back, then save.";
}
