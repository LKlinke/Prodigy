// The scale and the colours of the plots (#452): one probability scale from linear to logarithmic and
// three colour maps, shared by the heatmap, the 3D view (joint-view.js) and the marginal bars
// (charts.js). Pure functions without DOM, so that tests can run them in Node.

// ---- the probability scale --------------------------------------------------------------------

/** At most this many decades are shown on a logarithmic scale; smaller values get the minimal height. */
export const LOG_DECADES = 12;

/** The positions the scale slider snaps to, from linear to logarithmic. */
export const SCALE_STOPS = Object.freeze([
  Object.freeze({ lambda: 1, name: "Linear", short: "Linear" }),
  Object.freeze({ lambda: 0.5, name: "Square root", short: "√ Root" }),
  Object.freeze({ lambda: 0, name: "Logarithmic", short: "Log" }),
]);

/** Below this distance the slider snaps to a stop. */
const SNAP = 0.04;

/** λ clamped to [0, 1] and snapped to Linear, √ or Log when it is close to one of them. */
export function snapLambda(lambda) {
  const value = Number.isFinite(lambda) ? Math.max(0, Math.min(1, lambda)) : 1;
  const stop = SCALE_STOPS.find((s) => Math.abs(s.lambda - value) < SNAP);
  return stop ? stop.lambda : Math.round(value * 100) / 100;
}

/** The scale in words: "Linear (λ = 1)", "Square root (λ = 0.50)", "Logarithmic (λ → 0)", "Power (λ = 0.30)". */
export function scaleName(lambda) {
  if (lambda === 1) return "Linear (λ = 1)";
  if (lambda === 0) return "Logarithmic (λ → 0)";
  const stop = SCALE_STOPS.find((s) => s.lambda === lambda);
  return `${stop ? stop.name : "Power"} (λ = ${lambda.toFixed(2)})`;
}

/** The short label at the axis of a chart: "" for linear, "log", "√" or "λ 0.30". */
export function scaleTag(lambda) {
  if (lambda === 1) return "";
  if (lambda === 0) return "log";
  if (lambda === 0.5) return "√";
  return `λ ${lambda.toFixed(2)}`;
}

/** The Box-Cox transform (p^λ − 1)/λ; its limit λ → 0 is ln p. Monotone in p for every λ. */
export function boxCox(p, lambda) {
  return lambda === 0 ? Math.log(p) : (p ** lambda - 1) / lambda;
}

function inverseBoxCox(y, lambda) {
  return lambda === 0 ? Math.exp(y) : Math.max(0, lambda * y + 1) ** (1 / lambda);
}

/** The smallest "nice" number ≥ value (1, 2, 2.5 or 5 times a power of ten); 1 for value ≤ 0. */
export function niceMax(value) {
  if (!(value > 0)) return 1;
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * base) return step * base;
  }
  return 10 * base;
}

/** The number 1, 2 or 5 times a power of ten that is closest to p (on a logarithmic axis). */
function niceRound(p) {
  const exponent = Math.floor(Math.log10(p));
  const base = 10 ** exponent;
  const candidates = [1, 2, 5, 10].map((m) => m * base);
  return candidates.reduce((best, c) => (Math.abs(Math.log(c / p)) < Math.abs(Math.log(best / p)) ? c : best));
}

/**
 * The scale of a plot's probabilities. It maps p to t(p) in [0, 1] with the Box-Cox transform B_λ,
 * normalised between a floor f and the largest value M:
 *
 *     t(p) = (B_λ(p) − B_λ(f)) / (B_λ(M) − B_λ(f))
 *
 * λ = 1 is linear, λ = ½ the square root, λ = 0 the logarithm. The floor of the logarithm f₀ is the
 * smallest positive value, at most LOG_DECADES decades below M (`decades`: the whole decade below it,
 * and M the whole decade above the largest value, as the axis of a bar chart wants). For λ > 0 the
 * floor is f₀·(1 − λ): it is 0 for the linear scale, so that the linear scale starts at 0 as before,
 * and it moves continuously to f₀ as λ goes to 0. For every λ, t is continuous and increasing in p,
 * and the scale depends continuously on λ.
 *
 * `values` are the probabilities to show (null and non-numbers are ignored), `atLeast` raises M (the
 * frames of an animation share one scale, #289). Returns
 *   {lambda, min: f, max: M, t, height, ticks}
 * where t(p) is clamped to [0, 1] and null for 0 and for values without a number; height(p) is the
 * same without the upper clamp (a bar above M, such as the "?" tower, stays taller); ticks are
 * probabilities in increasing order between f and M (at λ = 0 whole decades, at λ = 1 0, M/2 and M,
 * in between round numbers at even distances of t). Without a positive value M is 0 and t is null.
 */
export function probabilityScale(values, { lambda = 1, atLeast = 0, decades = false } = {}) {
  const lam = snapLambda(lambda);
  const positive = values.filter((p) => Number.isFinite(p) && p > 0);
  if (!positive.length) {
    return { lambda: lam, min: 0, max: 0, t: () => null, height: () => null, ticks: [0] };
  }
  const largest = Math.max(atLeast, ...positive);
  let max;
  let floor0;
  if (decades) {
    const hi = Math.ceil(Math.log10(largest) - 1e-9);
    // strictly below the smallest value, so that it keeps a visible bar
    const lo = Math.max(Math.floor(Math.log10(Math.min(...positive)) - 1e-9), hi - LOG_DECADES);
    max = lam === 0 ? 10 ** hi : niceMax(largest);
    floor0 = 10 ** lo;
  } else {
    max = largest;
    floor0 = 10 ** Math.max(Math.log10(Math.min(...positive)), Math.log10(max) - LOG_DECADES);
  }
  const floor = floor0 * (1 - lam);
  const b0 = lam === 0 || floor > 0 ? boxCox(floor, lam) : -1 / lam;
  const span = boxCox(max, lam) - b0;
  const height = (p) => {
    if (!Number.isFinite(p) || !(p > 0)) return null;
    if (!(span > 0)) return 1;
    return p <= floor ? 0 : (boxCox(p, lam) - b0) / span;
  };
  const t = (p) => {
    const value = height(p);
    return value === null ? null : Math.min(1, value);
  };
  const at = (u) => inverseBoxCox(b0 + u * span, lam);
  return { lambda: lam, min: floor, max, t, height, ticks: scaleTicks(lam, floor, max, decades, t, at) };
}

function scaleTicks(lam, floor, max, decades, t, at) {
  if (!(max > floor)) return [max];
  if (lam === 1) return [0, max / 2, max];
  const lo = Math.round(Math.log10(floor));
  const hi = Math.round(Math.log10(max));
  if (lam === 0 && decades) {
    // the axis of a bar chart: whole decades from the top down, at most about five of them
    const step = Math.ceil((hi - lo) / 4);
    const ticks = [];
    for (let k = hi; k >= lo; k -= step) ticks.push(10 ** k);
    return ticks.reverse();
  }
  // both ends are labelled (#307); between them one round number near the middle of the scale, if
  // one leaves room for the labels (a tick closer than a third of the scale to an end would overlap
  // its label): a whole decade on the logarithmic scale, else 1, 2 or 5 times a power of ten
  const candidates = lam === 0
    ? Array.from({ length: Math.max(0, hi - lo + 1) }, (_, i) => 10 ** (lo + i))
    : [0.5, 0.4, 0.6].map((u) => niceRound(at(u)));
  const middle = candidates
    .map((p) => ({ p, u: t(p) }))
    .filter(({ u }) => u !== null && u > 0.35 && u < 0.65)
    .sort((a, b) => Math.abs(a.u - 0.5) - Math.abs(b.u - 0.5))[0];
  return middle ? [floor, middle.p, max] : [floor, max];
}

// ---- colour maps ------------------------------------------------------------------------------

/** The colour maps of the plots: perceptually uniform, the first two readable with colour-vision deficiencies. */
export const COLOUR_MAPS = Object.freeze({
  viridis: Object.freeze({ name: "Viridis", stops: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"] }),
  cividis: Object.freeze({ name: "Cividis", stops: ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#fee838"] }),
  // for print: more ink for more probability
  greyscale: Object.freeze({ name: "Greyscale", stops: ["#e6e6e6", "#141414"] }),
});

const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const toHex = (rgb) => `#${rgb.map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("")}`;

/** The colour of map `name` (viridis if unknown) at t in [0, 1] as #rrggbb. */
export function colourAt(name, t) {
  const stops = (COLOUR_MAPS[name] || COLOUR_MAPS.viridis).stops;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const [a, b] = [hex(stops[i]), hex(stops[i + 1])];
  return toHex(a.map((v, k) => v + (b[k] - v) * (x - i)));
}

/** The viridis colour at t in [0, 1]. */
export function viridis(t) {
  return colourAt("viridis", t);
}

/** A colour #rrggbb times `factor` (the shaded sides of a 3D bar). */
export function shade(colour, factor) {
  return toHex(hex(colour).map((v) => v * factor));
}

/** The relative luminance of #rrggbb (WCAG 2). */
export function luminance(colour) {
  const [r, g, b] = hex(colour).map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Text colour (black or white) with the higher contrast on the background #rrggbb. */
export function textOn(colour) {
  const l = luminance(colour);
  return (l + 0.05) / 0.05 > 1.05 / (l + 0.05) ? "#000000" : "#ffffff";
}
