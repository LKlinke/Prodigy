// The comparison of two programs (issue #154) as plain data: verdict texts, what the check found,
// and the marginals of the two output distributions side by side. No DOM, so that Node can test it;
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
  if (check.counterexample) lines.push(`Counterexample of the solver: input ${stateText(check.counterexample)}.`);
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

/** An input distribution in words: "1" is the point where every variable is 0. */
export function inputWords(distribution) {
  if (!distribution || String(distribution).trim() === "1") return "the input where every variable is 0";
  return `the input ${distribution}`;
}

/** Where the input of the shown distributions comes from. */
export function inputText(equivalence) {
  const input = equivalence?.input || {};
  if (input.source === "counterexample") return `Both programs on ${inputWords(input.distribution)} (the solver's counterexample).`;
  if (input.distribution) return `Both programs on the initial distribution ${input.distribution}.`;
  return "Both programs on the default input (every variable 0).";
}

/** A probability for a table cell: a short exact value as it is, else the decimal. */
export function cellText(exact, decimal) {
  return exact && exact.length <= 10 && !/[a-zA-Z]/.test(exact) ? exact : decimal;
}

/**
 * The interval in which the probability of one value of a marginal lies (floats), given the guarantee of the
 * result: exact [p, p], lower bound [p, upper end], upper bound [0, p]; null without a guarantee. A value without
 * a row has probability 0 (a lower bound: at most the residual mass).
 */
export function rowInterval(result, row) {
  const kind = result?.bounds?.kind;
  const p = row ? row.p : 0;
  if (p === null || p === undefined || !Number.isFinite(p)) return null;
  if (kind === "exact") return [p, p];
  if (kind === "upper") return [0, p];
  if (kind === "lower") {
    const residual = result?.bounds?.residual_mass_float;
    const upper = row ? row.upper_p : (Number.isFinite(residual) ? residual : 1);
    return [p, Math.min(1, Number.isFinite(upper) ? upper : 1)];
  }
  return null;
}

/**
 * The marginal of `variable` in both results, row by row: [{value, first, second, differs}], where
 * first/second are table rows ({prob, p}) or null. `differs` is a hint where to look, not a proof (the witness is
 * the proof): true if both results are exact and the probabilities differ, or if the intervals of the two
 * probabilities (rowInterval) do not overlap; false if both are exact and agree; null if no comparison is possible.
 */
export function compareMarginals(first, second, variable) {
  const table = (result) => result?.status === "ok" ? result.distribution?.marginals?.[variable] : null;
  const one = table(first);
  const two = table(second);
  const exact = first?.bounds?.kind === "exact" && second?.bounds?.kind === "exact";
  const values = new Map();
  for (const [index, rows] of [[0, one?.rows], [1, two?.rows]]) {
    for (const row of rows || []) {
      const entry = values.get(row.value) || [null, null];
      entry[index] = row;
      values.set(row.value, entry);
    }
  }
  const complete = (t, value) => t && !t.error && (!t.truncated || value <= (t.rows?.at(-1)?.value ?? -1));
  const close = (a, b) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));
  return [...values.entries()].sort((a, b) => a[0] - b[0]).map(([value, [a, b]]) => {
    let differs = null;
    if (complete(one, value) && complete(two, value)) {
      if (exact) {
        const pa = a ? a.p : 0;
        const pb = b ? b.p : 0;
        const same = (a?.prob ?? "0") === (b?.prob ?? "0") || (pa !== null && pb !== null && close(pa, pb));
        differs = !same;
      } else {
        const ia = rowInterval(first, a);
        const ib = rowInterval(second, b);
        if (ia && ib && ((ia[1] < ib[0] && !close(ia[1], ib[0])) || (ib[1] < ia[0] && !close(ib[1], ia[0])))) differs = true;
      }
    }
    return { value, first: a, second: b, differs };
  });
}

/** The variables of the two results (the first's order, then any only the second has). */
export function comparedVariables(first, second) {
  const names = [...(first?.distribution?.variables || [])];
  for (const name of second?.distribution?.variables || []) if (!names.includes(name)) names.push(name);
  return names;
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
