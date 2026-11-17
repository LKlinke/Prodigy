// The setups of the two modes of the window (#154). Analyze and Compare share only the programs (program A is the
// first editor in both); each mode keeps its own setup: the input, the strategy of each loop and the options. So
// nothing set up for an analysis (e.g. the strategy "auto") reaches a comparison, and back. Program B and the
// strategies of its loops belong to Compare. Plain data, no DOM, so that Node can test it.

/** Where each mode keeps its setup in the browser (store.js). */
export const SETUP_KEYS = { analyze: "setup", compare: "compareSetup" };

/** The options a setup has; Compare has no step trace (a comparison records no steps). */
const OPTION_KEYS = ["engine", "normalize", "simplification", "coefficients", "timeLimit", "synthesis", "solver",
  "templateHeuristic", "positivityHeuristic", "recordSteps"];

/** The engine the server uses by default (the first available one otherwise). */
function defaultEngine(meta) {
  const engines = meta?.engines || [];
  return (engines.find((e) => e.default) || engines.find((e) => e.available))?.name || "sympy";
}

/** A fresh setup of `mode` ("analyze" or "compare"). */
export function defaultSetup(meta, mode = "analyze") {
  const options = {
    engine: defaultEngine(meta), normalize: true, simplification: true, coefficients: 30,
    timeLimit: Math.min(120, meta?.limits?.time_limit ?? 120), synthesis: "default", solver: "sympy",
    templateHeuristic: "default", positivityHeuristic: "default",
  };
  if (mode === "compare") {
    // the check covers every input at once; `input` is the one input the plots show when `inputKind` is "given"
    return { input: "", inputKind: "default", decisions: {}, options };
  }
  return { input: "", decisions: {}, options: { ...options, recordSteps: false } };
}

/** The setup of `mode` from what the browser kept (`saved`, may be anything), checked against the server. */
export function restoreSetup(saved, meta, mode = "analyze") {
  const setup = defaultSetup(meta, mode);
  if (!saved || typeof saved !== "object") return setup;
  setup.input = typeof saved.input === "string" ? saved.input : "";
  if (mode === "compare") setup.inputKind = saved.inputKind === "given" ? "given" : "default";
  if (saved.decisions && typeof saved.decisions === "object") setup.decisions = saved.decisions;
  for (const key of OPTION_KEYS) {
    if (Object.hasOwn(setup.options, key) && saved.options && Object.hasOwn(saved.options, key)) {
      setup.options[key] = saved.options[key];
    }
  }
  if (!(meta?.engines || []).some((e) => e.name === setup.options.engine && e.available)) {
    setup.options.engine = defaultEngine(meta);
  }
  const limit = meta?.limits?.time_limit ?? 120;
  setup.options.timeLimit = Math.min(Number(setup.options.timeLimit) || 120, limit);
  return setup;
}

/** The initial distribution a setup runs on (null: the default input). */
export function setupInput(setup, mode = "analyze") {
  if (mode === "compare" && setup?.inputKind !== "given") return null;
  return (setup?.input || "").trim() || null;
}

// ---- the strategies of loops in Compare ----------------------------------------------------------------

/**
 * The strategy a loop of one program gets in Compare when nobody chose one. An equivalence verdict needs results
 * that are proven, and the check runs every program on all inputs at once, where unrolling or a synthesis gives
 * no verdict. So the default is "Check invariant" (Park's lemma): with the other program as the invariant when
 * this program is that one loop and the other program is loop-free (as `prodigy check_equality` and "Compare
 * this loop" do), otherwise with an invariant program still to be written. `outline` is the outline of the
 * loop's program, `other` {outline, source} the other program. `byDefault` marks the decision as a default: it
 * follows the programs until somebody changes it.
 */
export function compareDefault(loop, outline, other) {
  const loops = outline?.loops || [];
  const source = other?.source || "";
  const otherLoopFree = Boolean(other?.outline) && !(other.outline.loops || []).length && source.trim() !== "";
  const invariant = loops.length === 1 && !loop.function && otherLoopFree ? source : "";
  return { strategy: "invariant", params: { invariant }, byDefault: true };
}

/** Why a loop has its default (shown on its card), or null if it is no default. `which` is "A" or "B". */
export function defaultReason(decision, which) {
  if (!decision?.byDefault) return null;
  const otherName = which === "A" ? "B" : "A";
  return decision.params?.invariant
    ? `Default in Compare: program ${otherName} is this loop's invariant (Check invariant). It follows program ${otherName} until you change it.`
    : `Default in Compare: an equivalence needs proven results, so write the loop-free program this loop equals `
      + `(its invariant), or choose another strategy.`;
}

/**
 * Gives every loop of both programs a decision: a loop without one, with an unknown strategy, or with a default
 * gets the Compare default (compareDefault), so defaults follow the programs. `first` and `second` are
 * {outline, source, decisions} of programs A and B. Returns whether a decision changed.
 */
export function applyCompareDefaults(first, second, strategies) {
  const known = new Set((strategies || []).map((s) => s.name));
  let changed = false;
  for (const [own, other] of [[first, second], [second, first]]) {
    for (const loop of own.outline?.loops || []) {
      const key = String(loop.id);
      const current = own.decisions[key];
      if (current && known.has(current.strategy) && !current.byDefault) continue;
      const next = compareDefault(loop, own.outline, other);
      if (JSON.stringify(current) !== JSON.stringify(next)) {
        own.decisions[key] = next;
        changed = true;
      }
    }
  }
  return changed;
}

/** The decisions of `outline`'s loops as a request sends them: strategy and parameters only. */
export function requestDecisions(decisions, outline) {
  const sent = {};
  for (const loop of outline?.loops || []) {
    const decision = decisions[String(loop.id)];
    if (decision) sent[String(loop.id)] = { strategy: decision.strategy, params: decision.params };
  }
  return sent;
}
