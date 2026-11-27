// Layout parts shared by the lenses of the analysis card (analysis-card.js, cost-lenses.js, checks-lens.js):
// where a step is, the fact line under a figure, and a disclosure for the details.
import { h } from "./dom.js";

export function where(step) {
  if (!step) return "";
  if (step.kind === "normalize") return "end of the program";
  return step.line ? `line ${step.line}` : "generated";
}

export function facts(...items) {
  return h("dl", { class: "ac-facts" }, ...items.filter(Boolean).map(([label, value, title]) =>
    h("div", {}, h("dt", {}, label), h("dd", { title: title || "" }, value))));
}

export function disclosure(summary, ...body) {
  const content = body.flat().filter(Boolean);
  if (!content.length) return null;
  return h("details", { class: "disclosure" }, h("summary", {}, summary), h("div", { class: "disclosure-body" }, ...content));
}
