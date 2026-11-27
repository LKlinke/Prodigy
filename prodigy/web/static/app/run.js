// Running an analysis or a check as a job on the server: the request, polling, the toolbar's messages.
import { api } from "./api.js";
import { $, editor, otherEditor, state } from "./app-state.js";
import { checkNow } from "./checking.js";
import { checkOtherNow, showColumn } from "./compare-mode.js";
import { requestDecisions, setupInput } from "./mode-setup.js";
import { selectTab, showLoop, updateAnnotations } from "./panels.js";
import { renderResultPanel, setBadge } from "./result-panel.js";
import { setupProblems } from "./setup.js";
import { changeSince } from "./staleness.js";
import { h, icon } from "./components/dom.js";
import { formatSeconds } from "./components/format.js";
import { liveChartStates, renderRunning } from "./components/result-view.js";

let pollTimer = null;

function setRunning(running) {
  $("btn-run").hidden = running;
  $("btn-stop").hidden = !running;
  if (!running) $("run-status").replaceChildren();
}

let runMessageCounter = 0;

export function showRunMessage(message, tone = "error", actions = []) {
  const status = $("run-status");
  const text = h("span", { class: "run-message-text" }, message);
  const outcome = h("span", { class: tone === "error" ? "outcome" : "", "data-kind": tone === "error" ? "error" : null },
    icon(tone === "error" ? "error" : "info"), " ", text);
  const wrap = h("span", { class: "run-message" }, outcome, ...actions);
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
  wrap.firstElementChild.after(" ", button, popover); // next to the text, before any action button
}

/** The server is busy (429): what is running, with a button that cancels it, so that the user can start again. */
function showBusy(message, running) {
  const cancel = h("button", { type: "button", class: "button small", id: "btn-cancel-busy",
    title: "Cancel what is running on the server", onclick: async () => {
      cancel.disabled = true;
      await Promise.allSettled(running.map((job) => api.cancel(job.id)));
      showRunMessage("Cancelled. Press Run to start again.", "info");
    } }, `Cancel ${running.length === 1 ? "it" : "them"}`);
  // like every run message: a long text is cut off in the toolbar and opens in full on request (#426)
  showRunMessage(message, "error", [cancel]);
  cancel.focus();
}

/** A short notice (opened, saved); information disappears after a few seconds. */
export function notify(message, tone = "info") {
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

export function buildRequest() {
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
export async function requestSuggestion(loopId, form, { signal, onProgress }) {
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

export async function run() {
  if (state.job) return;
  await checkNow();
  const errors = state.diagnostics.filter((d) => d.severity === "error");
  if (errors.length) {
    showRunMessage(state.mode === "compare" ? `Fix line ${errors[0].line} of program A first.` : `Fix line ${errors[0].line} first.`);
    if (state.mode === "compare") showColumn("first");
    editor.revealLine(errors[0].line);
    return;
  }
  const problems = setupProblems(state.outline, state.setup, state.strategies);
  if (problems.length) {
    showRunMessage(state.mode === "compare" ? `Program A: ${problems[0].message}` : problems[0].message);
    showLoop(problems[0].loop, { field: Boolean(problems[0].param) });
    return;
  }
  const comparing = state.mode === "compare";
  if (comparing) {
    await checkOtherNow();
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
    const otherProblems = setupProblems(state.compare.outline, { decisions: state.compare.decisions }, state.strategies);
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
    if (error.status === 429 && error.data?.running?.length) showBusy(error.message, error.data.running);
    else showRunMessage(error.message);
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

export async function stop() {
  const job = state.job;
  if (!job) return;
  try {
    const status = await api.cancel(job.id);
    if (state.job === job) finish({ ...(status.result || {}), log: job.log }, status.elapsed);
  } catch (error) {
    showRunMessage(error.message);
  }
}
