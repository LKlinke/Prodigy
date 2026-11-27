// Prodigy web GUI: boots the window (editors, document, library, keyboard) and wires the app's modules.
import { api } from "./api.js";
import { $, documents, editor, otherEditor, saveSetup, setDocuments, setEditors, state } from "./app-state.js";
import { checkNow, renderProblems, runCheck, scheduleCheck } from "./checking.js";
import { followProgramB, forgetOrigin, initCompareTabs, initModeSwitch, modeFromUrl, modeKey, renderOtherSyntax,
  saveBlocked, scheduleOtherCheck, setMode, setOtherSource, showColumn } from "./compare-mode.js";
import { canWriteFiles, createDocument } from "./document.js";
import { createEditor, SYNTAXES } from "./editor.js";
import { createLibrary } from "./library.js";
import { restoreSetup, SETUP_KEYS } from "./mode-setup.js";
import { engineChanged, initTabs, renderEngine, renderSetupPanel, selectTab, updateAnnotations } from "./panels.js";
import { analysisKey, refreshStale, renderResultPanel, resetAnalysis } from "./result-panel.js";
import { notify, run, showRunMessage, stop } from "./run.js";
import { cancelSuggestions, swapInvariants } from "./setup.js";
import { load, save } from "./store.js";
import { h, icon } from "./components/dom.js";

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
  await checkNow();
  renderSetupPanel();
  updateAnnotations();
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
  const note = suggestion.note;
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

// ---- syntax (.ppy) ------------------------------------------------------------------------

function renderSyntax() {
  const syntax = editor.getSyntax();
  $("syntax-label").textContent = SYNTAXES[syntax].label;
  $("syntax-label").setAttribute("aria-label", `Syntax: ${SYNTAXES[syntax].label}`);
  $("btn-save").title = canWriteFiles ? "Save (⌘S) · Save As… (⇧⌘S)"
    : `Download the program as a ${SYNTAXES[syntax].extension} file (⌘S)`;
}

/** The syntax changed (New, Open…, an example, or undoing one of them). */
function syntaxChanged(syntax) {
  save("syntax", syntax);
  renderSyntax();
  documents?.changed();
  renderSetupPanel();
  scheduleCheck(0);
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
  const first = createEditor($("editor"), {
    doc: storedSource ?? WELCOME,
    // a program kept by an earlier version may be .pgcl; the server then reports that .pgcl was removed
    syntax: "ppy",
    onSyntaxChange: syntaxChanged,
    onCompanion: companionApplied,
    onChange: (source) => {
      save("source", source);
      cancelSuggestions();
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
  const second = createEditor($("editor-other"), {
    doc: load("otherSource", "") || "",
    syntax: "ppy",
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
  setEditors(first, second);
  renderOtherSyntax();
  initCompareTabs();
  initModeSwitch();
  // (a new key: what an earlier version kept there were the defaults of Analyze, e.g. "auto")
  state.compare.decisions = load("compareDecisionsB", {}) || {};
  state.compare.origin = load("compareOrigin", null);
  state.compare.lastOther = otherEditor.getSource();
  state.analysis.watches = (load("watches", []) || []).filter((w) => typeof w === "string").slice(0, 8);
  save("syntax", editor.getSyntax());
  setDocuments(createDocument({ editor, initialText: editor.getSource(), onReplace: programReplaced, notify,
    onChange: renderTitle, saveBlocked }));
  documents.changed();

  const library = createLibrary($("library"), { api, onLoad: loadExample, onLoadFile: (entry) => documents.openRecentFile(entry) });
  $("btn-new").addEventListener("click", () => documents.new());
  $("btn-open").addEventListener("click", () => documents.open());
  $("btn-save").addEventListener("click", () => documents.save());
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
