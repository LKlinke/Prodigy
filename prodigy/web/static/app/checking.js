// Checking program A while it is typed: diagnostics, the outline of its loops, the initial distribution.
import { api } from "./api.js";
import { $, editor, otherEditor, saveSetup, state } from "./app-state.js";
import { compareDefaults, showColumn } from "./compare-mode.js";
import { setupInput } from "./mode-setup.js";
import { renderSetupPanel, updateAnnotations } from "./panels.js";
import { refreshStale } from "./result-panel.js";
import { applyInputProblem, rekeyDecisions } from "./setup.js";
import { save } from "./store.js";
import { h, icon } from "./components/dom.js";

let checkTimer = null;
let checkController = null;

export function scheduleCheck(delay = 350) {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(runCheck, delay);
}

export async function runCheck() {
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

/** Checks program A now (before a run, or after the program was replaced). */
export function checkNow() {
  clearTimeout(checkTimer);
  return runCheck();
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
export function scheduleInputCheck() {
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

export function renderProblems(failure = null) {
  renderProblemsButton($("problems"), state.diagnostics, editor, failure);
}

export function renderProblemsButton(button, diagnostics, target, failure = null) {
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
