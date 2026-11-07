// Transitions of the stepper (issue #198): two consecutive views are keyframes, and the values in
// between are interpolated with ease-in/ease-out on requestAnimationFrame. Interpolated numbers
// are only a transition: they are shown muted and hidden from assistive technology, and the exact
// value replaces them at the end. With prefers-reduced-motion, only the end state is shown.
import { h } from "./dom.js";
import { formatNumber } from "./format.js";

export const reducedMotion = () => Boolean(globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

/** Ease-in/ease-out (cubic). */
export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2);

const lerp = (a, b, t) => a + (b - a) * t;

/**
 * Runs frame(e) with the eased progress e from 0 to 1 over `duration` ms, then done().
 * Returns {cancel()}; with reduced motion it runs the last frame at once.
 */
export function tween(duration, frame, done = null) {
  let handle = 0;
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    cancelAnimationFrame(handle);
    frame(1);
    done?.();
  };
  if (reducedMotion() || duration <= 0 || typeof requestAnimationFrame === "undefined") {
    finish();
    return { cancel: () => {}, finish };
  }
  let start = null;
  const step = (now) => {
    if (finished) return;
    if (start === null) start = now;
    const t = Math.min(1, (now - start) / duration);
    if (t >= 1) { finish(); return; }
    frame(ease(t));
    handle = requestAnimationFrame(step);
  };
  frame(0);
  handle = requestAnimationFrame(step);
  return {
    // stop where it is (the next transition starts from the current state)
    cancel: () => { finished = true; cancelAnimationFrame(handle); },
    finish,
  };
}

// ---- bar charts --------------------------------------------------------------------------

/** The current state of a marginal chart (charts.js), possibly in the middle of a transition. */
export function chartState(svg) {
  if (!svg?.dataset.max) return null;
  const width = Number(svg.dataset.width);
  const bars = new Map();
  for (const group of svg.querySelectorAll(":scope > .mc-bar[data-key]")) {
    const rect = group.querySelector("rect");
    bars.set(group.dataset.key, {
      p: Number(rect.dataset.currentP ?? rect.dataset.p),
      cx: Number(rect.dataset.currentCx ?? rect.dataset.cx) / width,
      w: Number(rect.dataset.currentW ?? rect.dataset.w) / width,
      className: group.getAttribute("class"),
      fill: rect.getAttribute("fill"),
    });
  }
  return { variable: svg.dataset.variable, max: Number(svg.dataset.currentMax ?? svg.dataset.max), bars };
}

/** Applies the state between `from` (chartState of the previous chart) and the chart's own state at e. */
function applyChart(svg, from, e) {
  const width = Number(svg.dataset.width);
  const top = Number(svg.dataset.top);
  const plotH = Number(svg.dataset.plotH);
  const max = lerp(from.max, Number(svg.dataset.max), e);
  svg.dataset.currentMax = String(max);
  for (const group of svg.querySelectorAll(":scope > .mc-bar[data-key]")) {
    const rect = group.querySelector("rect");
    const before = from.bars.get(group.dataset.key);
    const p1 = Number(rect.dataset.p);
    const cx1 = Number(rect.dataset.cx);
    const w1 = Number(rect.dataset.w);
    const p = before ? lerp(before.p, p1, e) : p1 * e;
    const cx = before ? lerp(before.cx * width, cx1, e) : cx1;
    const w = before ? lerp(before.w * width, w1, e) : w1;
    const height = Math.min(plotH, Math.max(p > 0 ? 1 : 0, plotH * (p / max)));
    rect.setAttribute("x", String(cx - w / 2));
    rect.setAttribute("width", String(w));
    rect.setAttribute("y", String(top + plotH - height));
    rect.setAttribute("height", String(height));
    rect.dataset.currentP = String(p);
    rect.dataset.currentCx = String(cx);
    rect.dataset.currentW = String(w);
    group.style.opacity = before ? "" : String(e);
    const label = group.querySelector("text");
    if (label) label.setAttribute("x", String(cx));
  }
  // bars that are gone shrink and fade out
  let ghosts = svg.querySelector(":scope > .mc-ghosts");
  if (!ghosts) {
    ghosts = document.createElementNS("http://www.w3.org/2000/svg", "g");
    ghosts.setAttribute("class", "mc-ghosts");
    ghosts.setAttribute("aria-hidden", "true");
    const present = new Set([...svg.querySelectorAll(":scope > .mc-bar[data-key]")].map((g) => g.dataset.key));
    for (const [key, bar] of from.bars) {
      if (present.has(key)) continue;
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", bar.className);
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      // a hatched bar takes the pattern of the new chart (the old one is gone)
      const pattern = svg.querySelector("pattern")?.id;
      if (bar.fill?.startsWith("url(") && pattern) rect.setAttribute("fill", `url(#${pattern})`);
      rect.dataset.p = String(bar.p);
      rect.dataset.cx = String(bar.cx * width);
      rect.dataset.w = String(bar.w * width);
      group.append(rect);
      ghosts.append(group);
    }
    svg.append(ghosts);
  }
  for (const group of ghosts.children) {
    const rect = group.firstChild;
    const p = Number(rect.dataset.p) * (1 - e);
    const height = Math.min(plotH, plotH * (p / max));
    const w = Number(rect.dataset.w);
    rect.setAttribute("x", String(Number(rect.dataset.cx) - w / 2));
    rect.setAttribute("width", String(w));
    rect.setAttribute("y", String(top + plotH - height));
    rect.setAttribute("height", String(height));
    group.style.opacity = String(1 - e);
  }
  if (e >= 1) ghosts.remove();
  // the axis rescales with the bars; its labels (of the new scale) fade in
  if (Math.abs(from.max - Number(svg.dataset.max)) > 1e-12) {
    for (const grid of svg.querySelectorAll(":scope > .mc-grid")) grid.style.opacity = String(e);
  }
}

/**
 * Tweens the chart that `find()` returns from the state `from`. `find` is asked on every frame,
 * because a chart may be redrawn (e.g. at its real width) while the transition runs.
 */
export function tweenChart(find, from, duration) {
  if (!from) return null;
  return tween(duration, (e) => {
    const svg = find();
    if (svg?.dataset.max && svg.dataset.variable === from.variable) applyChart(svg, from, e);
  });
}

// ---- numbers -------------------------------------------------------------------------------

/**
 * Counts the number shown in `element` from `from` to `to`. While it counts, a muted, approximate
 * number is shown and hidden from assistive technology; the element's own (exact) content stays
 * in the document for screen readers and is shown again at the end.
 */
export function countNumber(element, from, to, duration, { digits = 4, integer = false } = {}) {
  if (!element || !Number.isFinite(from) || !Number.isFinite(to) || from === to) return null;
  const exact = h("span", { class: "tw-exact" }, ...element.childNodes);
  const live = h("span", { class: "tw-live", "aria-hidden": "true" });
  element.replaceChildren(exact, live);
  element.classList.add("tw-counting");
  return tween(duration, (e) => {
    const value = lerp(from, to, e);
    element.dataset.current = String(value);
    live.textContent = integer ? Math.round(value).toLocaleString("en-US") : `≈ ${formatNumber(value, digits)}`;
  }, () => {
    element.classList.remove("tw-counting");
    delete element.dataset.current;
    element.replaceChildren(...exact.childNodes);
  });
}

// ---- closed forms ----------------------------------------------------------------------------

/**
 * Cross-fades the formula box `box` (formula-box.js) from the content `previous` (a clone of the
 * previous box's scroll area). The box keeps its size: the old content lies over the new one.
 */
export function crossFade(box, previous, duration) {
  const frame = box?.querySelector(".formula-frame");
  const scroller = box?.querySelector(".formula-scroll");
  if (!frame || !previous) return null;
  const ghost = h("div", { class: "formula-ghost", "aria-hidden": "true" }, previous.node);
  frame.append(ghost);
  ghost.scrollTop = previous.scrollTop;
  ghost.scrollLeft = previous.scrollLeft;
  return tween(duration, (e) => {
    ghost.style.opacity = String(1 - e);
    scroller.style.opacity = String(e);
  }, () => {
    ghost.remove();
    scroller.style.opacity = "";
  });
}

/** What crossFade needs of the box that is about to be replaced. */
export function formulaState(box) {
  const scroller = box?.querySelector(".formula-scroll");
  if (!scroller) return null;
  const node = scroller.querySelector(".formula-content")?.cloneNode(true);
  if (!node) return null;
  // in the middle of a cross-fade, continue from what is visible: the newer content
  return { node, scrollTop: scroller.scrollTop, scrollLeft: scroller.scrollLeft };
}
