// Small parts shared by the result view (result-view.js) and the analysis card (analysis-card.js):
// badges, buttons, segmented controls, cards and the figure of one marginal. Like those modules it
// needs only the result JSON and the CSS custom properties of result-view.css.
import { copyText, h, icon } from "./dom.js";
import { chartFocus, marginalChart, marginalTable } from "./charts.js";
import { BOUND_TEXT, formatValue } from "./format.js";
import { onViewOptionsChange, viewOptions, viewOptionsControl } from "./view-options.js";
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

/** The figure of one marginal with its own chart/table switch and the view options (queries, dist values). */
export function distributionFigure(table, result, variable, mode = "chart") {
  const residual = result.bounds?.kind === "lower" ? result.bounds?.residual_mass_float : null;
  const residualExact = result.bounds?.residual_mass;
  const bound = result.bounds?.kind || "exact";
  const holder = h("div", { class: "rv-figure-body" });
  const toggle = h("div", { class: "segmented small", role: "group", "aria-label": `Show the distribution of ${variable} as` });
  const views = { chart: "Chart", table: "Table" };
  const optionsControl = viewOptionsControl({ label: `View options of the chart of ${variable}: scale, colours and intervals` });
  let current = mode;
  let drawnWidth = 0;
  // the selected bar stays selected when the chart is drawn anew (#462)
  const selection = {};
  const show = (which) => {
    current = which;
    drawnWidth = holder.clientWidth || 560;
    const refocus = chartFocus(holder);
    holder.replaceChildren(which === "chart"
      ? marginalChart(table, { residual, residualExact, variable, bound, width: drawnWidth, view: viewOptions(), selection })
      : marginalTable(table, { variable, bound }));
    refocus();
    for (const button of toggle.children) button.setAttribute("aria-pressed", String(button.dataset.view === which));
    optionsControl.setAvailable(which === "chart");
  };
  onViewOptionsChange(holder, () => { if (current === "chart") show("chart"); });
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
    h("div", { class: "rv-figure-head" }, optionsControl.button, toggle),
    optionsControl.frame(holder),
    caption.length ? h("figcaption", { class: "rv-note" }, caption.join(" ")) : null);
}
