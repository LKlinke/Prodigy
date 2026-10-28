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
