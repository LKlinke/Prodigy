// The view options of the plots (#452): one "View options" button next to the view switch opens a
// popover with the scale (a slider from linear to logarithmic), the colour map and the display of
// intervals. The choice applies to every plot (heatmap, 3D view, marginal bars) and is remembered per
// viewer in localStorage; it is not part of an analysis. Like parts.js this depends on no GUI module.
import { h, s } from "./dom.js";
import { COLOUR_MAPS, SCALE_STOPS, colourAt, scaleName, snapLambda } from "./plot-scale.js";

const KEY = "prodigy.gui.viewOptions";
// the linear/log switch before #452; its choice is taken over once
const OLD_SCALE_KEY = "prodigy.gui.chartScale";
const EVENT = "prodigy:view-options";

export const INTERVAL_STYLES = Object.freeze({ glass: "Glass", outline: "Outline" });

/** Outline instead of glass when the system asks for less transparency or more contrast. */
function prefersOutline() {
  try {
    return Boolean(globalThis.matchMedia?.("(prefers-reduced-transparency: reduce)").matches
      || globalThis.matchMedia?.("(prefers-contrast: more)").matches);
  } catch {
    return false;
  }
}

function storage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

/**
 * The options as {lambda, colours, intervals, unaccounted}: λ of the scale in [0, 1] (1 linear,
 * 0 logarithmic), the colour map (a key of COLOUR_MAPS), "glass" or "outline", and whether the
 * mass not yet accounted for is drawn as a "?" bar. Unknown or missing entries get their defaults.
 */
export function loadViewOptions() {
  let stored = {};
  let oldScale = null;
  try {
    stored = JSON.parse(storage()?.getItem(KEY) || "{}") || {};
    oldScale = storage()?.getItem(OLD_SCALE_KEY);
  } catch {
    stored = {};
  }
  return normaliseOptions(stored, oldScale === "log" ? 0 : 1);
}

/** `raw` with every field checked against its allowed values (stored text is untrusted). */
export function normaliseOptions(raw, defaultLambda = 1) {
  const value = raw && typeof raw === "object" ? raw : {};
  return {
    lambda: snapLambda(typeof value.lambda === "number" ? value.lambda : defaultLambda),
    colours: Object.hasOwn(COLOUR_MAPS, value.colours) ? value.colours : "viridis",
    intervals: Object.hasOwn(INTERVAL_STYLES, value.intervals) ? value.intervals : (prefersOutline() ? "outline" : "glass"),
    unaccounted: typeof value.unaccounted === "boolean" ? value.unaccounted : true,
  };
}

/** Stores a change of the options and tells every plot (onViewOptionsChange). */
export function saveViewOptions(change) {
  const options = normaliseOptions({ ...current(), ...change });
  try {
    storage()?.setItem(KEY, JSON.stringify(options));
  } catch {
    // ignore: storage is a convenience only; the change still applies to this page
    memory = options;
  }
  globalThis.dispatchEvent?.(new CustomEvent(EVENT, { detail: options }));
  return options;
}

// the options of this page when localStorage refuses to store them
let memory = null;
const current = () => memory || loadViewOptions();

/**
 * Calls `callback(options)` after each change while `node` is in the document; the listener removes
 * itself once the node is gone (the plots are drawn anew often, so they must not pile up).
 */
export function onViewOptionsChange(node, callback) {
  const listener = (event) => {
    if (!node.isConnected) {
      globalThis.removeEventListener(EVENT, listener);
      return;
    }
    callback(event.detail);
  };
  globalThis.addEventListener?.(EVENT, listener);
}

/** The options in force (stored, or kept in memory when storage is not available). */
export function viewOptions() {
  return current();
}

// ---- the control ------------------------------------------------------------------------------

let counter = 0;

function adjustmentsIcon() {
  return s("svg", { class: "icon", viewBox: "0 0 16 16", "aria-hidden": "true" },
    s("path", { d: "M2 4.5h7 M12.5 4.5H14 M2 11.5h2 M7.5 11.5H14" }),
    s("circle", { cx: 10.75, cy: 4.5, r: 1.75 }), s("circle", { cx: 5.75, cy: 11.5, r: 1.75 }));
}

function gradient(name) {
  const stops = Array.from({ length: 6 }, (_, i) => `${colourAt(name, i / 5)} ${i * 20}%`).join(", ");
  return h("span", { class: "vo-ramp", "aria-hidden": "true", style: `background: linear-gradient(to right, ${stops})` });
}

function radio(group, value, label, checked, extra = null) {
  return h("label", { class: "vo-choice" },
    h("input", { type: "radio", name: group, value, checked }), extra, h("span", {}, label));
}

/**
 * The "View options" button and its popover. The popover is not modal: Escape and a click outside
 * close it, and focus returns to the button; Tab leaves it and closes it. Every change applies at
 * once (saveViewOptions). `label` names what the options apply to, for screen readers.
 */
export function viewOptionsControl({ label = "View options of the plots" } = {}) {
  const id = `vo-${++counter}`;
  const button = h("button", { type: "button", class: "button small vo-button", "aria-haspopup": "dialog",
    "aria-expanded": "false", "aria-controls": id, title: label },
  adjustmentsIcon(), h("span", { class: "vo-button-text" }, "View options"));

  const options = current();
  const slider = h("input", { type: "range", class: "vo-slider", min: "0", max: "100", step: "1",
    "aria-labelledby": `${id}-scale`, "aria-describedby": `${id}-scale-value` });
  const scaleValue = h("output", { class: "vo-value", id: `${id}-scale-value`, for: `${id}-slider` });
  slider.id = `${id}-slider`;
  const stops = h("div", { class: "vo-stops" }, ...SCALE_STOPS.map((stop) => h("button", { type: "button",
    class: "vo-stop", "data-lambda": stop.lambda, title: scaleName(stop.lambda), "aria-label": `${stop.name} scale`,
    onclick: () => { saveViewOptions({ lambda: stop.lambda }); } }, stop.short)));
  const colours = h("div", { class: "vo-choices", role: "radiogroup", "aria-labelledby": `${id}-colour` },
    ...Object.entries(COLOUR_MAPS).map(([key, map]) => radio(`${id}-colours`, key,
      key === "cividis" ? `${map.name} (colour-vision deficiencies)` : key === "greyscale" ? `${map.name} (print)` : map.name,
      options.colours === key, gradient(key))));
  const intervals = h("div", { class: "vo-choices vo-row", role: "radiogroup", "aria-labelledby": `${id}-intervals` },
    ...Object.entries(INTERVAL_STYLES).map(([key, name]) => radio(`${id}-intervals`, key, name, options.intervals === key,
      h("span", { class: `vo-glyph vo-glyph-${key}`, "aria-hidden": "true" }))));
  const unaccounted = h("input", { type: "checkbox", checked: options.unaccounted });
  const intervalsNote = h("p", { class: "vo-note" },
    "Glass: a translucent part with dashed edges from the lower to the upper end of an interval. Outline: the same without the fill.");

  const popover = h("div", { class: "vo-popover", id, role: "dialog", "aria-label": "View options", hidden: true },
    h("section", { class: "vo-group" },
      h("h4", { class: "vo-title", id: `${id}-scale` }, "Scale"),
      slider, stops, scaleValue),
    h("section", { class: "vo-group" },
      h("h4", { class: "vo-title", id: `${id}-colour` }, "Colour"), colours),
    h("section", { class: "vo-group" },
      h("h4", { class: "vo-title", id: `${id}-intervals` }, "Intervals"), intervals, intervalsNote,
      h("label", { class: "vo-choice vo-check" }, unaccounted,
        h("span", { class: "vo-glyph vo-glyph-residual", "aria-hidden": "true" }, "?"),
        h("span", {}, "Show unaccounted mass (?)"))));

  const sync = (o) => {
    slider.value = String(Math.round((1 - o.lambda) * 100));
    slider.setAttribute("aria-valuetext", scaleName(o.lambda));
    scaleValue.textContent = scaleName(o.lambda);
    for (const stop of stops.children) stop.setAttribute("aria-pressed", String(Number(stop.dataset.lambda) === o.lambda));
    for (const input of colours.querySelectorAll("input")) input.checked = input.value === o.colours;
    for (const input of intervals.querySelectorAll("input")) input.checked = input.value === o.intervals;
    unaccounted.checked = o.unaccounted;
  };
  sync(options);
  slider.addEventListener("input", () => {
    const lambda = snapLambda(1 - Number(slider.value) / 100);
    saveViewOptions({ lambda });
  });
  colours.addEventListener("change", (event) => saveViewOptions({ colours: event.target.value }));
  intervals.addEventListener("change", (event) => saveViewOptions({ intervals: event.target.value }));
  unaccounted.addEventListener("change", () => saveViewOptions({ unaccounted: unaccounted.checked }));

  const wrapper = h("span", { class: "vo" }, button, popover);
  onViewOptionsChange(wrapper, sync);

  const close = ({ refocus = true } = {}) => {
    if (popover.hidden) return;
    popover.hidden = true;
    button.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", outside, true);
    if (refocus) button.focus();
  };
  const outside = (event) => {
    if (!wrapper.contains(event.target)) close({ refocus: false });
  };
  const open = () => {
    sync(current());
    popover.hidden = false;
    button.setAttribute("aria-expanded", "true");
    // the popover opens below the button, aligned to its right edge, and is moved to stay inside the window
    popover.style.transform = "";
    const box = popover.getBoundingClientRect();
    const room = document.documentElement.clientWidth || globalThis.innerWidth || box.right;
    const shift = box.left < 16 ? 16 - box.left : Math.min(0, room - 16 - box.right);
    if (shift) popover.style.transform = `translateX(${Math.round(shift)}px)`;
    document.addEventListener("pointerdown", outside, true);
    slider.focus();
  };
  button.addEventListener("click", () => (popover.hidden ? open() : close()));
  wrapper.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !popover.hidden) {
      event.preventDefault();
      event.stopPropagation();
      close();
    }
  });
  wrapper.addEventListener("focusout", (event) => {
    if (!popover.hidden && event.relatedTarget && !wrapper.contains(event.relatedTarget)) close({ refocus: false });
  });
  return wrapper;
}
