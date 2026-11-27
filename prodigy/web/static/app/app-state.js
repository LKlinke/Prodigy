// The state of the window shared by the app's modules: the person's programs and setups, the result shown,
// the running job, and the two editors and the document (set once by main.js at boot).
import { SETUP_KEYS } from "./mode-setup.js";
import { load, save } from "./store.js";

export const $ = (id) => document.getElementById(id);

export const state = {
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

export let editor = null;
export let otherEditor = null;
export let documents = null;

/** main.js creates the editors and the document at boot. */
export function setEditors(first, second) {
  editor = first;
  otherEditor = second;
}

export function setDocuments(value) {
  documents = value;
}

/** Keeps the setup of the shown mode (or of `mode`) in the browser. */
export function saveSetup(mode = state.mode) {
  if (state.setups[mode]) save(SETUP_KEYS[mode], state.setups[mode]);
}
