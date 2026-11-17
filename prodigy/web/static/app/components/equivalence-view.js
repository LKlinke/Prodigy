// The result of comparing two programs (issue #154): the verdict, why, a witness if there is one,
// the two output distributions side by side, and below the full result of either program (the
// result view of result-view.js). Like result-view.js it needs only the result JSON.
import { h, icon } from "./dom.js";
import { formatSeconds, formatValue } from "./format.js";
import { card, expressionBox, guaranteeBadge, segmented, press } from "./parts.js";
import { renderResult } from "./result-view.js";
import {
  cellText, checkLines, compareMarginals, comparedVariables, constraintTexts, inputText, inputWords, intervalText,
  relationText, stateText, verdictOf,
} from "./equivalence-model.js";
import { staleText } from "../staleness.js";

const PROGRAMS = [["first", "Program A"], ["second", "Program B"]];

function verdictCard(result, equivalence) {
  const verdict = verdictOf(equivalence);
  const check = equivalence.check || {};
  const children = [
    h("div", { class: "eq-verdict-head" },
      h("span", { class: "eq-glyph", "aria-hidden": "true" }, verdict.glyph),
      h("div", { class: "eq-verdict-title" },
        h("h3", { class: "eq-title" }, verdict.title),
        h("span", { class: "eq-subtitle" }, verdict.subtitle)),
      check.time !== undefined ? h("span", { class: "rv-meta" }, `check ${formatSeconds(check.time)}`) : null),
    h("p", { class: "eq-reason" }, equivalence.reason || ""),
  ];
  const relation = relationText(equivalence.relation);
  if (relation) {
    children.push(h("p", { class: "eq-relation" }, h("span", { class: "rv-muted" }, "Proven: "),
      h("span", { class: "mono" }, relation)));
  }
  const constraints = constraintTexts(equivalence.constraints);
  if (constraints.length) {
    children.push(h("div", { class: "eq-constraints" },
      h("span", { class: "rv-muted" }, constraints.length === 1 ? "Parameter values: " : "Parameter values (any of): "),
      h("ul", {}, ...constraints.map((text) => h("li", { class: "mono" }, text)))));
  }
  if (equivalence.witness) children.push(witnessTable(equivalence.witness));
  children.push(checkDetails(check));
  return h("section", { class: "rv-card eq-verdict", "data-verdict": verdict.key, "data-tone": verdict.tone,
    "aria-label": `Verdict: ${verdict.title}` }, ...children);
}

function witnessTable(witness) {
  const row = (label, side) => h("tr", {}, h("th", { scope: "row" }, label),
    h("td", { class: "num", title: side.lower === side.upper ? side.lower : `[${side.lower}, ${side.upper}]` },
      intervalText(side, (exact, number) => formatValue(exact, number, 6))),
    h("td", {}, guaranteeBadge(side.kind)));
  return h("div", { class: "eq-witness" },
    h("h4", { class: "rv-block-title" }, "Witness"),
    h("p", { class: "rv-note" }, `Probability of the output state ${stateText(witness.state)}`,
      ` on ${inputWords(witness.input)}`,
      ". The intervals do not overlap, so the programs differ."),
    h("div", { class: "table-wrap" }, h("table", { class: "rv-table eq-witness-table" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, ""), h("th", { scope: "col" }, "Probability"),
        h("th", { scope: "col" }, "Guarantee"))),
      h("tbody", {}, row("Program A", witness.first), row("Program B", witness.second)))));
}

function checkDetails(check) {
  const body = checkLines(check).map((line) => h("p", {}, line));
  if (check.difference) body.push(expressionBox(check.difference, "Difference of the second-order generating functions"));
  return h("details", { class: "disclosure eq-check" },
    h("summary", {}, "How it was checked"),
    h("div", { class: "disclosure-body" },
      h("p", { class: "rv-muted" }, "Prodigy runs both programs on one generating function that stands for every "
        + "input at once (second-order check), then both on the input shown below to look for a witness."),
      ...body));
}

/** The marginals of both programs, side by side, with the rows where exact values differ marked. */
function distributionsCard(result, equivalence) {
  const other = equivalence.other || {};
  const variables = comparedVariables(result, other);
  const children = [h("p", { class: "rv-note" }, inputText(equivalence))];
  const badges = h("div", { class: "eq-badges" },
    ...PROGRAMS.map(([key, label]) => {
      const r = key === "first" ? result : other;
      return h("span", { class: "eq-badge-item" }, h("span", { class: "rv-muted" }, `${label}: `),
        r.status === "ok" ? guaranteeBadge(r.bounds?.kind || "unknown") : guaranteeBadge("error", "Failed"));
    }));
  children.push(badges);
  if (!variables.length || (result.status !== "ok" && other.status !== "ok")) {
    children.push(h("p", { class: "rv-muted" }, "No output distribution to compare; see the results below."));
    return card("eq-distributions", "Output distributions", ...children);
  }
  for (const variable of variables) {
    const rows = compareMarginals(result, other, variable);
    if (!rows.length) continue;
    const anyDiffers = rows.some((r) => r.differs);
    const cell = (row) => row ? h("td", { class: "num", title: row.prob ?? "" }, cellText(row.prob, formatValue(row.prob, row.p, 6)))
      : h("td", { class: "num rv-muted" }, "0");
    const body = rows.map((r) => h("tr", { "data-differs": r.differs ? "true" : null },
      h("td", { class: "num" }, String(r.value)), cell(r.first), cell(r.second),
      h("td", { class: "eq-diff" }, r.differs === true ? [h("span", { class: "glyph", "aria-hidden": "true" }, "≠"), " differs"]
        : r.differs === false ? h("span", { class: "rv-muted" }, "same") : "")));
    children.push(h("div", { class: "eq-marginal" },
      h("h4", { class: "rv-block-title" }, `Pr[${variable} = k]`,
        anyDiffers ? h("span", { class: "eq-marginal-flag" }, " · values differ") : null),
      h("div", { class: "table-wrap" }, h("table", { class: "rv-table eq-table", "aria-label": `Distribution of ${variable} in programs A and B` },
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "k"), h("th", { scope: "col" }, "A"),
          h("th", { scope: "col" }, "B"), h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Comparison")))),
        h("tbody", {}, ...body)))));
  }
  if (result.bounds?.kind !== "exact" || other.bounds?.kind !== "exact") {
    children.push(h("p", { class: "rv-note" }, "Rows are compared only when both results are exact; bounds may "
      + "differ without the programs differing."));
  }
  return card("eq-distributions", "Output distributions", ...children);
}

/**
 * Renders a comparison result (an analysis result with an `equivalence` part) into `container`.
 * options: those of renderResult (for the result of program A) plus `programView` ({which}) to
 * remember the chosen program, `onSelectOtherLoop(id)` for the loops of program B, `stale`, `change`, `onRun`. Returns {analysis} of the shown result.
 */
export function renderEquivalence(container, result, options = {}) {
  const equivalence = result.equivalence;
  const view = options.programView || { which: "first" };
  const children = [];
  if (options.stale) {
    children.push(h("div", { class: "notice rv-stale", role: "status" }, icon("info"),
      h("span", {}, staleText(options.change)),
      options.onRun ? h("button", { type: "button", class: "button small", onclick: () => options.onRun() },
        "Check Again") : null));
  }
  children.push(verdictCard(result, equivalence), distributionsCard(result, equivalence));
  const holder = h("div", { class: "eq-program-result" });
  let analysis = null;
  const choose = (which) => {
    view.which = which;
    press(switcher, which);
    analysis?.stop?.();
    const shown = which === "second" ? equivalence.other || {} : result;
    // the editor callbacks (lines, loops) belong to the first program's editor
    const own = which === "first" ? options
      : { strategyTitles: options.strategyTitles, onSelectLoop: options.onSelectOtherLoop };
    ({ analysis } = renderResult(holder, shown, { ...own, stale: false, loadJoint: undefined }));
  };
  const switcher = segmented("Result of", PROGRAMS, choose);
  children.push(h("div", { class: "eq-program-head" }, h("h3", { class: "rv-card-title" }, "Result of"), switcher), holder);
  container.replaceChildren(h("div", { class: "result-view eq-view", "data-stale": options.stale ? "true" : "false" },
    ...children));
  choose(view.which === "second" ? "second" : "first");
  return { get analysis() { return analysis; } };
}
