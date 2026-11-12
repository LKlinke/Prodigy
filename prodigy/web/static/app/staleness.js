// Whether a result still matches the program and the setup: the request of its analysis is
// remembered and compared with the request the current program and setup would send (#310).

/** JSON with sorted object keys, so that equal requests give equal text. */
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().filter((key) => value[key] !== undefined)
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value ?? null);
}

/** The program part (source, syntax) and the setup part (everything else) of a request. */
function split(request) {
  const { source, syntax, ...setup } = request || {};
  return { program: canonical({ source, syntax }), setup: canonical(setup) };
}

/**
 * What changed between the request of a result (`analyzed`) and the current one: null if nothing,
 * else "program", "setup" or "both". A result without a remembered request counts as current.
 */
export function changeSince(analyzed, current) {
  if (!analyzed) return null;
  const before = split(analyzed);
  const now = split(current);
  const program = before.program !== now.program;
  const setup = before.setup !== now.setup;
  if (program && setup) return "both";
  return program ? "program" : setup ? "setup" : null;
}

const WHAT = { program: "the program", setup: "the setup", both: "the program and the setup" };

/** The notice above an outdated result. */
export function staleText(change) {
  return `This result is for an earlier version of ${WHAT[change] || WHAT.both}.`;
}

/** The note in the progress view when the program or setup changed after the analysis started. */
export function changedWhileRunningText(change) {
  return `You changed ${WHAT[change] || WHAT.both} after this analysis started; its result will be for the earlier version.`;
}
