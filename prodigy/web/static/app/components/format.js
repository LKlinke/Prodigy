// Number and value formatting shared by the result components.

const SUPERSCRIPT = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

function scientific(x, digits) {
  const [mantissa, exponent] = x.toExponential(digits - 1).split("e");
  const exp = String(Number(exponent)).split("").map((c) => SUPERSCRIPT[c] || c).join("");
  return `${trimZeros(mantissa)} × 10${exp}`;
}

function trimZeros(text) {
  return text.includes(".") ? text.replace(/0+$/, "").replace(/\.$/, "") : text;
}

/** A state (or any valuation) in words: "x = 0, n = 1", or `empty` when it has no variable. */
export function stateText(state, empty = "the empty state") {
  const entries = Object.entries(state || {});
  return entries.length ? entries.map(([name, value]) => `${name} = ${value}`).join(", ") : empty;
}

/** A float for display: 4 significant digits, scientific notation for very small values. */
export function formatNumber(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return x > 0 ? "∞" : "−∞";
  if (x === 0) return "0";
  const abs = Math.abs(x);
  if (abs < 1e-4 || abs >= 1e7) return scientific(x, digits);
  if (abs >= 1000) return x.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return trimZeros(x.toPrecision(digits)).replace(/^-/, "−");
}

/**
 * Display text for an exact value with its float. Values that round to 1 (or 0) but are not
 * exactly 1 (or 0) are shown as 1 − ε (or ε) so that rounding never fakes exactness.
 */
export function formatValue(exact, number, digits = 4) {
  if (number === null || number === undefined) return exact ?? "—";
  const shown = formatNumber(number, digits);
  if (exact !== null && exact !== undefined) {
    if (shown === "1" && exact !== "1") {
      const rest = 1 - number;
      return rest > 0 ? `1 − ${formatNumber(rest, 2)}` : "≈ 1";
    }
    if (shown === "0" && exact !== "0") return `≈ ${scientific(number, 2)}`;
  }
  return shown;
}

/**
 * Both ends of an interval. If they would look identical after rounding, the upper end is
 * written as "lower + width", so that a tiny but non-zero width stays visible.
 */
export function formatInterval(lower, lowerFloat, upper, upperFloat) {
  const a = formatValue(lower, lowerFloat);
  const b = formatValue(upper, upperFloat);
  if (a === b && lowerFloat !== null && upperFloat !== null && upperFloat > lowerFloat) {
    return [a, `${a} + ${formatNumber(upperFloat - lowerFloat, 2)}`];
  }
  return [a, b];
}

// ---- probabilities as intervals (#405) ---------------------------------------------------------

const TEN = 10n;

// an exact rational: an integer, a fraction or a terminating decimal ("3/7", "1", "0.031684")
const RATIONAL = /^(-?)(?:(\d+)(?:\/(\d+))?|(\d*)\.(\d+))$/;

/** [negative, numerator, denominator] of an exact rational text, or null. */
function parseRational(text) {
  const match = typeof text === "string" ? RATIONAL.exec(text.trim()) : null;
  if (!match) return null;
  if (match[2] !== undefined) return [match[1] === "-", BigInt(match[2]), BigInt(match[3] ?? "1")];
  return [match[1] === "-", BigInt(`${match[4] || "0"}${match[5]}`), TEN ** BigInt(match[5].length)];
}

/** The decimal text of mantissa × 10^exponent (mantissa: digits as a string), plain or scientific. */
function decimalText(negative, mantissa, exponent) {
  const digits = mantissa.replace(/^0+(?=\d)/, "");
  if (/^0+$/.test(digits)) return "0";
  const magnitude = digits.length - 1 + exponent;
  let text;
  if (magnitude < -4 || magnitude >= 7) {
    const head = trimZeros(`${digits[0]}.${digits.slice(1)}`);
    const exp = String(magnitude).split("").map((c) => SUPERSCRIPT[c] || c).join("");
    text = `${head} × 10${exp}`;
  } else if (exponent >= 0) {
    text = digits + "0".repeat(exponent);
  } else {
    const padded = digits.padStart(-exponent + 1, "0");
    text = trimZeros(`${padded.slice(0, padded.length + exponent)}.${padded.slice(padded.length + exponent)}`);
  }
  return negative ? `−${text}` : text;
}

/**
 * An endpoint of an interval as a decimal with `digits` significant digits, rounded outward:
 * `direction` "down" for a lower end, "up" for an upper end, so that the printed interval still
 * contains the true value. An exact rational ("3/7", "1") is rounded exactly (BigInt); any other
 * value is rounded from its float, and the last digit moves outward unless the decimal lies strictly
 * on the safe side of the float (the float is within half a unit in the last place of the value).
 */
export function outwardDecimal(exact, number, direction, digits = 6) {
  const rational = parseRational(exact);
  if (rational) {
    const [negative, numerator, denominator] = rational;
    if (numerator === 0n) return "0";
    // a negative value is rounded the other way in magnitude
    const down = (direction === "down") !== negative;
    // e = floor(log10(n / d))
    let e = numerator.toString().length - denominator.toString().length;
    const atLeast = (k) => (k >= 0 ? numerator >= denominator * TEN ** BigInt(k) : numerator * TEN ** BigInt(-k) >= denominator);
    if (!atLeast(e)) e -= 1;
    const shift = digits - 1 - e;
    const top = shift >= 0 ? numerator * TEN ** BigInt(shift) : numerator;
    const bottom = shift >= 0 ? denominator : denominator * TEN ** BigInt(-shift);
    let q = top / bottom;
    if (!down && q * bottom !== top) q += 1n;
    return decimalText(negative, q.toString(), -shift);
  }
  if (number === null || number === undefined || !Number.isFinite(number)) return formatNumber(number);
  if (number === 0) return "0";
  const [mantissaText, exponentText] = Math.abs(number).toExponential(digits - 1).split("e");
  let mantissa = BigInt(mantissaText.replace(".", ""));
  let exponent = Number(exponentText) - (digits - 1);
  const nearest = Number(`${mantissaText}e${exponentText}`);
  const magnitudeDown = (direction === "down") === (number > 0);
  if (magnitudeDown && nearest >= Math.abs(number)) mantissa -= 1n;
  if (!magnitudeDown && nearest <= Math.abs(number)) mantissa += 1n;
  if (mantissa === 0n) return "0";
  return decimalText(number < 0, mantissa.toString(), exponent);
}

/** Whether an exact value is worth showing next to its decimal (short and not the same text). */
function shortExactText(exact, number) {
  return typeof exact === "string" && exact.length <= 24 && exactIsInformative(exact, number) ? exact : null;
}

/**
 * The interval of a coefficient of a result with guarantee `bound` (prodigy/analysis/values.py:
 * rows of univariate_table, cells of joint_grid):
 *   exact    a single value: "1/4 (0.25)"
 *   lower    [p, min(1, p + r)] from the row's `upper` (the residual mass r included, capped at 1)
 *   upper    [0, p], or [lower, p] if the row has a `lower` end (sandwich)
 *   unknown  no interval: "unknown (computed 0.25, no guarantee)"
 * Exact rationals are shown with the decimals of the interval, rounded outward. Values with
 * parameters (no float) are shown as their expressions.
 */
export function probabilityText(bound, { prob = null, p = null, upper = null, upper_p = null, lower = null, lower_p = null } = {}, digits = 6) {
  if (bound === "unknown") {
    return `unknown (computed ${p === null ? prob ?? "—" : formatNumber(p, digits)}, no guarantee)`;
  }
  if (bound !== "lower" && bound !== "upper") {
    if (p === null) return prob ?? "—";
    const exact = shortExactText(prob, p);
    const decimal = formatNumber(p, digits);
    return exact ? `${exact} (${decimal})` : decimal;
  }
  const ends = bound === "lower"
    ? [{ exact: prob, p }, upper !== null || upper_p !== null ? { exact: upper, p: upper_p } : { exact: "1", p: 1 }]
    : [lower !== null || lower_p !== null ? { exact: lower, p: lower_p } : { exact: "0", p: 0 }, { exact: prob, p }];
  if (ends.some((end) => end.p === null || end.p === undefined)) {
    return `[${ends.map((end) => end.exact ?? "—").join(", ")}]`;
  }
  const decimals = `[${outwardDecimal(ends[0].exact, ends[0].p, "down", digits)}, ${outwardDecimal(ends[1].exact, ends[1].p, "up", digits)}]`;
  const exacts = ends.map((end) => (typeof end.exact === "string" && end.exact.length <= 24 && parseRational(end.exact) ? end.exact : null));
  const exactText = exacts.every(Boolean) ? `[${exacts.join(", ")}]` : null;
  return exactText && exactText !== decimals ? `${exactText} (${decimals})` : decimals;
}

/**
 * Where the upper end of a lower bound's interval comes from (#451), in words, or null: a value's
 * own certificate, or its lower end plus the mass not yet accounted for.
 */
export function upperSourceText(bound, { upper_source = null } = {}) {
  if (bound !== "lower") return null;
  if (upper_source === "certificate") return "upper end from a certificate of its own";
  if (upper_source === "residual") return "upper end = lower end + the mass not yet accounted for";
  return null;
}

/** The guarantee of the coefficients in words (the interval notation is not left to imply it). */
export const PROBABILITY_GUARANTEE = {
  exact: "Exact probabilities.",
  lower: "Bounds: each probability lies in [lower end, upper end]; the upper end adds the mass not yet accounted for (at most 1).",
  upper: "Bounds: each probability lies in [0, upper end].",
  unknown: "No guarantee: computed values that are neither exact nor bounds.",
};

/** How a bar chart draws the intervals (charts.js: marginalChart). */
export const BAR_LEGEND = {
  lower: "Bars are solid up to the lower end; hover or select a bar for the dashed part up to its upper end. The hatched “?” bar is the mass not yet accounted for.",
  upper: "Hatched bars reach up to the upper end.",
};

/** The guarantee and, for bounds, how the bars draw it. */
export function chartLegend(bound) {
  return [PROBABILITY_GUARANTEE[bound] || PROBABILITY_GUARANTEE.unknown, BAR_LEGEND[bound]].filter(Boolean).join(" ");
}

/**
 * How a query value is qualified: glyph (or null) and text, e.g. {glyph: "✓", text: "exact"}
 * or {glyph: null, text: "∈ [0.5, 0.5 + 3 × 10⁻⁸]"}. The interval notation carries its own
 * symbol, so no extra glyph is added.
 */
export function qualify(query, { withValue = false } = {}) {
  const value = formatValue(query.value, query.value_float);
  switch (query.bound) {
    case "exact":
      return { glyph: "✓", text: withValue ? `= ${value}` : "exact" };
    case "lower":
      if (query.upper !== null && query.upper !== undefined) {
        const [a, b] = formatInterval(query.lower, query.lower_float, query.upper, query.upper_float);
        return { glyph: null, text: `∈ [${a}, ${b}]` };
      }
      return { glyph: null, text: `≥ ${formatValue(query.lower, query.lower_float)}` };
    case "upper":
      return { glyph: null, text: `≤ ${formatValue(query.upper, query.upper_float)}` };
    default:
      return { glyph: "!", text: withValue ? `≈ ${value}, no guarantee` : "no guarantee" };
  }
}

/** Exact value, shortened for display (the full text goes into a tooltip). */
export function shortExact(exact, max = 28) {
  if (exact === null || exact === undefined) return "";
  return exact.length > max ? `${exact.slice(0, max - 1)}…` : exact;
}

/** Whether the exact text adds information beyond the decimal display. */
export function exactIsInformative(exact, number) {
  if (!exact) return false;
  if (number === null || number === undefined) return true;
  return exact !== formatNumber(number) && exact !== String(number);
}

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  if (seconds <= 120 || Number.isInteger(seconds)) return seconds % 60 === 0 ? `${seconds / 60} min` : `${Math.round(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds % 60)} s`;
}

export const BOUND_TEXT = {
  exact: { glyph: "✓", label: "Exact" },
  lower: { glyph: "≥", label: "Lower bound" },
  upper: { glyph: "≤", label: "Upper bound" },
  unknown: { glyph: "!", label: "No guarantee" },
};
