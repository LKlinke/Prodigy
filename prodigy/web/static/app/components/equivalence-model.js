// The comparison of two programs (issue #154) as plain data: verdict texts, what the check found and
// the witness. No DOM, so that Node can test it;
// equivalence-view.js draws it. The input is the `equivalence` part of a result (see
// prodigy/analysis/equivalence/verdict.py and docs/gui.md).

/** Each verdict has its own glyph and word: status is never shown by colour alone. `tone` picks the
 * colour token (exact: proven equal; data: proven different, which is no failure; unknown). */
export const VERDICT_TEXT = {
  equivalent: { glyph: "=", title: "Equivalent", tone: "exact", proven: true },
  not_equivalent: { glyph: "≠", title: "Not equivalent", tone: "data", proven: true },
  unknown: { glyph: "?", title: "Not decided", tone: "unknown", proven: false },
};

const CHECK_ANSWERS = {
  equal: "The second-order generating functions of the two programs are equal.",
  different: "The solver found no way to make the second-order generating functions equal.",
  undecided: "Equality of the second-order generating functions could be neither proven nor refuted.",
  failed: "The check did not finish.",
};

/** The verdict of an `equivalence` part, with its texts; an unknown verdict counts as `unknown`. */
export function verdictOf(equivalence) {
  const key = Object.hasOwn(VERDICT_TEXT, equivalence?.verdict) ? equivalence.verdict : "unknown";
  const text = VERDICT_TEXT[key];
  const error = equivalence?.check?.error;
  const notImplemented = key === "unknown" && error?.type === "NotImplementedError";
  return {
    key,
    ...text,
    // a "not implemented" stays one: its own title, never "not equivalent"
    title: notImplemented ? "Not decided: not implemented" : text.title,
    notImplemented,
    subtitle: key === "unknown" ? (notImplemented ? "The analysis cannot do this yet."
      : equivalence?.relation ? "One direction is proven." : "No proof either way.")
      : key === "equivalent" ? "Proven for every input." : "Proven by a witness.",
  };
}

/** One sentence for the relation that was proven when only one side is exact. */
export function relationText(relation) {
  if (relation === "first_below_second") return "Program A ⪯ Program B (coefficient-wise, for every input)";
  if (relation === "second_below_first") return "Program B ⪯ Program A (coefficient-wise, for every input)";
  return null;
}

/** "p = 1/3" for each solution of a parameter synthesis. */
export function constraintTexts(constraints) {
  return (constraints || []).map((c) => Object.entries(c).map(([name, value]) => `${name} = ${value}`).join(", "))
    .filter(Boolean);
}

export function stateText(state) {
  const entries = Object.entries(state || {});
  return entries.length ? entries.map(([k, v]) => `${k} = ${v}`).join(", ") : "the empty state";
}

/** "1/3" for an exact value, "[a, b]" for an interval; long exact numbers as decimals (`format` turns an
 * exact value and its float into text, e.g. formatValue). */
export function intervalText(side, format = (exact) => exact) {
  if (!side) return "—";
  const lower = cellText(side.lower, format(side.lower, side.lower_float));
  const upper = cellText(side.upper, format(side.upper, side.upper_float));
  return side.lower === side.upper ? lower : `[${lower}, ${upper}]`;
}

/** Lines that say what the check for every input found (the "How it was checked" disclosure). */
export function checkLines(check) {
  if (!check) return [];
  const lines = [CHECK_ANSWERS[check.answer] || check.answer];
  if (check.answer !== "failed") {
    lines.push(`Program A analysed ${boundWord(check.first)}; program B analysed ${boundWord(check.second)}.`);
  }
  if (check.counterexample) lines.push(`Counterexample of the solver: input ${inputText(check.counterexample)}.`);
  if (check.error?.message) {
    const prefix = check.error.type === "NotImplementedError" ? "Not implemented: " : "";
    lines.push(prefix + check.error.message);
  }
  return lines;
}

function boundWord(kind) {
  return { exact: "exactly", lower: "as a lower bound", upper: "as an upper bound (Park's lemma)",
    unknown: "without guarantee" }[kind] || kind;
}

/** The input of a witness (a state, every variable listed) in words: "x = 1, n = 3", or "every variable 0". */
export function inputText(input) {
  const set = Object.entries(input || {}).filter(([, value]) => value);
  return set.length ? set.map(([name, value]) => `${name} = ${value}`).join(", ") : "every variable 0";
}

/** A probability for a table cell: a short exact value as it is, else the decimal. */
export function cellText(exact, decimal) {
  return exact && exact.length <= 10 && !/[a-zA-Z]/.test(exact) ? exact : decimal;
}

// ---- continuing in Analyze with program B as the loop's result (#399) ---------------------------------

/** What Analyze gets for the loop when it continues with program B: the words of the guarantee. */
export const CONTINUE_GUARANTEE = {
  exact: { text: "The loop's result is program B's. The loop keeps the strategy that proved this; Analyze runs it again on your input, and its result shows what it proves there." },
  upper: { text: "Program B over-approximates the loop (Park's lemma, checked again in Analyze); exact if the loop terminates almost surely." },
};

/**
 * Whether a comparison lets Analyze continue after a loop with program B as the loop's result, and with which
 * guarantee. `loopsOfA` / `loopsOfB` count the loops of the two programs: program A must be one loop and
 * program B loop-free. Returns {guarantee: "exact" | "upper"}, {reason} when a proven relation still does not
 * allow it (shown as a note), or null.
 */
export function continueOffer(equivalence, { loopsOfA = 0, loopsOfB = 0 } = {}) {
  if (!equivalence) return null;
  const guarantee = equivalence.verdict === "equivalent" ? "exact"
    : equivalence.verdict === "unknown" && equivalence.relation === "first_below_second" ? "upper" : null;
  if (equivalence.verdict === "unknown" && equivalence.relation === "second_below_first") {
    return { reason: "Only Program B ⪯ Program A is proven, so program B is a lower bound of the loop; Analyze "
      + "cannot continue with it. For lower bounds, unroll the loop." };
  }
  if (!guarantee || loopsOfA !== 1) return null;
  if ((equivalence.constraints || []).length) {
    return { reason: "Program B holds for the parameter values above; put them into program B to continue in Analyze." };
  }
  if (loopsOfB) return { reason: "Program B has loops; only a loop-free program B can be the loop's result in Analyze." };
  return { guarantee };
}

/**
 * The decision of the loop in Analyze when it continues with program B (`programB`, its text). For an upper
 * bound it is the strategy `invariant` with B, which Analyze checks again (Park's lemma); for an exact result the
 * loop keeps `checked`, the decision that proved it equal to B, since `invariant` alone only gives an upper
 * bound. `compared` records where the decision came from (the Setup shows it; it is not sent).
 */
export function continueDecision(guarantee, programB, checked) {
  if (guarantee === "exact" && checked?.strategy && checked.strategy !== "invariant") {
    return { strategy: checked.strategy, params: structuredClone(checked.params || {}),
      compared: { guarantee, program: programB } };
  }
  return { strategy: "invariant", params: { invariant: programB }, compared: { guarantee: "upper", program: programB } };
}
