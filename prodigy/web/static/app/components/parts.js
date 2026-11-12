// Small parts shared by the result view (result-view.js) and the analysis card (analysis-card.js):
// badges, buttons, segmented controls, cards and the figure of one marginal. Like those modules it
// needs only the result JSON and the CSS custom properties of result-view.css.
import { copyText, h, icon } from "./dom.js";
import { marginalChart, marginalTable } from "./charts.js";
import { BOUND_TEXT, formatValue } from "./format.js";
import { formulaBox } from "./formula-box.js";

export function guaranteeBadge(kind, label = null) {
  const text = BOUND_TEXT[kind] || BOUND_TEXT.unknown;
  return h("span", { class: "rv-badge", "data-kind": kind },
    h("span", { class: "glyph", "aria-hidden": "true" }, text.glyph), label ?? text.label);
}

/** A long expression as text in a bounded box (see formula-box.js). */
export function expressionBox(text, label) {
  return formulaBox({ kind: "text", text: String(text ?? ""), label });
}

export function copyButton(label, text) {
  const button = h("button", { type: "button", class: "button small", title: label },
    icon("copy"), h("span", {}, label));
  button.addEventListener("click", async () => {
    const ok = await copyText(text);
    button.querySelector("span").textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => { button.querySelector("span").textContent = label; }, 1400);
  });
  return button;
}

let cardCounter = 0;

/** A card of the result view: a titled section on the inspector's background. */
export function card(className, title, ...content) {
  const id = `rv-card-${++cardCounter}`;
  return h("section", { class: `rv-card ${className}`, "aria-labelledby": id },
    h("h3", { class: "rv-card-title", id }, title), ...content);
}

/** A segmented control of mutually exclusive buttons; `onChoose(key)` runs on a click. */
export function segmented(label, choices, onChoose, title = () => null) {
  const group = h("div", { class: "segmented small", role: "group", "aria-label": label });
  for (const [key, text] of choices) {
    group.append(h("button", { type: "button", "data-key": key, "aria-pressed": "false", title: title(key),
      onclick: () => onChoose(key) }, text));
  }
  return group;
}

export function press(group, key) {
  for (const button of group.children) button.setAttribute("aria-pressed", String(button.dataset.key === key));
}

// The scale of the charts (linear | log), remembered per browser. Same key as store.js would use;
// kept here because these components must not depend on the GUI's modules (VS Code webview).
const SCALE_KEY = "prodigy.gui.chartScale";

export function loadScale() {
  try {
    return window.localStorage.getItem(SCALE_KEY) === "log" ? "log" : "linear";
  } catch {
    return "linear";
  }
}

export function saveScale(scale) {
  try {
    window.localStorage.setItem(SCALE_KEY, scale);
  } catch {
    // ignore: storage is a convenience only
  }
}

/** The figure of one marginal with its own chart/table and scale switches (queries, dist values). */
export function distributionFigure(table, result, variable, mode = "chart") {
  const residual = result.bounds?.kind === "lower" ? result.bounds?.residual_mass_float : null;
  const residualExact = result.bounds?.residual_mass;
  const bound = result.bounds?.kind || "exact";
  const holder = h("div", { class: "rv-figure-body" });
  const toggle = h("div", { class: "segmented small", role: "group", "aria-label": `Show the distribution of ${variable} as` });
  const views = { chart: "Chart", table: "Table" };
  const scaleToggle = h("div", { class: "segmented small", role: "group", "aria-label": `Scale of the probability axis of ${variable}` });
  let current = mode;
  let scale = loadScale();
  let drawnWidth = 0;
  const show = (which) => {
    current = which;
    drawnWidth = holder.clientWidth || 560;
    holder.replaceChildren(which === "chart"
      ? marginalChart(table, { residual, residualExact, variable, bound, width: drawnWidth, scale })
      : marginalTable(table, { variable, bound }));
    for (const button of toggle.children) button.setAttribute("aria-pressed", String(button.dataset.view === which));
    for (const button of scaleToggle.children) button.setAttribute("aria-pressed", String(button.dataset.scale === scale));
    scaleToggle.hidden = which !== "chart";
  };
  for (const [key, label] of Object.entries({ linear: "Linear", log: "Log" })) {
    scaleToggle.append(h("button", { type: "button", "data-scale": key, "aria-pressed": "false",
      title: key === "log" ? "Logarithmic probability axis: small probabilities stay visible" : "Linear probability axis",
      onclick: () => { scale = key; saveScale(key); show("chart"); } }, label));
  }
  if (typeof ResizeObserver !== "undefined") {
    // draw the chart at its real size so that labels keep their font size
    new ResizeObserver(() => {
      if (current === "chart" && holder.clientWidth && Math.abs(holder.clientWidth - drawnWidth) > 16) show("chart");
    }).observe(holder);
  }
  for (const [key, label] of Object.entries(views)) {
    toggle.append(h("button", { type: "button", "data-view": key, "aria-pressed": "false", onclick: () => show(key) }, label));
  }
  show(mode);
  const caption = [];
  if (table.truncated && table.rows?.length) {
    caption.push(`Values 0–${table.rows[table.rows.length - 1].value}` +
      (table.tail_mass_float ? `; ${formatValue(table.tail_mass, table.tail_mass_float)} of the mass lies beyond.` : "."));
  }
  if (residual) caption.push(`Hatched: ${formatValue(residualExact, residual)} not yet accounted for.`);
  if (bound === "upper") caption.push("Hatched bars are upper bounds.");
  if (bound === "unknown") caption.push("Approximate values without a guarantee.");
  return h("figure", { class: "rv-figure" },
    h("div", { class: "rv-figure-head" }, scaleToggle, toggle),
    holder,
    caption.length ? h("figcaption", { class: "rv-note" }, caption.join(" ")) : null);
}
