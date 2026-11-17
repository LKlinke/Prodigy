// The result of comparing two programs (issue #154): the verdict, why, a witness if there is one,
// the two output distributions side by side, and below the full result of either program (the
// result view of result-view.js). Like result-view.js it needs only the result JSON.
import { h, icon } from "./dom.js";
import { formatSeconds, formatValue } from "./format.js";
import { card, expressionBox, guaranteeBadge, segmented, press } from "./parts.js";
import { renderResult } from "./result-view.js";
import {
  CONTINUE_GUARANTEE, checkLines, constraintTexts, inputText, inputWords, intervalText, relationText, stateText, verdictOf,
} from "./equivalence-model.js";
import { comparePlots } from "./compare-plots.js";
import { staleText } from "../staleness.js";

const PROGRAMS = [["first", "Program A"], ["second", "Program B"]];

function verdictCard(result, equivalence, continueTo = null) {
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
  const next = continueAction(continueTo);
  if (next) children.push(next);
  children.push(checkDetails(check));
  return h("section", { class: "rv-card eq-verdict", "data-verdict": verdict.key, "data-tone": verdict.tone,
    "aria-label": `Verdict: ${verdict.title}` }, ...children);
}

/** "Continue in Analyze with B as the loop's result" (#399), with the guarantee Analyze will have, or why a
 * proven relation does not allow it. `continueTo`: {guarantee, target, onContinue} or {reason}. */
function continueAction(continueTo) {
  if (!continueTo) return null;
  if (!continueTo.guarantee) return continueTo.reason ? h("p", { class: "rv-note eq-continue-note" }, continueTo.reason) : null;
  const guarantee = CONTINUE_GUARANTEE[continueTo.guarantee];
  const helpId = "eq-continue-help";
  return h("div", { class: "eq-continue" },
    h("button", { type: "button", class: "button", "aria-describedby": helpId, onclick: () => continueTo.onContinue() },
      icon("forward"), h("span", {}, "Continue in Analyze with B as the loop's result")),
    h("p", { class: "rv-note", id: helpId }, guaranteeBadge(continueTo.guarantee === "exact" ? "exact" : "upper"), " ",
      guarantee.text, continueTo.target ? ` ${continueTo.target}` : ""));
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

/** The output distributions of both programs, plotted side by side (compare-plots.js). */
function distributionsCard(result, equivalence, options) {
  return card("eq-distributions", "Output distributions",
    h("p", { class: "rv-note" }, inputText(equivalence)),
    comparePlots(result, equivalence.other || {}, options.plots || {}, { loadJoint: options.loadJoint, maxJointAxis: options.maxJointAxis }));
}

/**
 * Renders a comparison result (an analysis result with an `equivalence` part) into `container`.
 * options: those of renderResult (for the result of program A) plus `programView` ({which}) to
 * remember the chosen program, `plots` (the choices of the distribution plots, kept across re-renders),
 * `loadJoint(payload, signal, onPartial)` for the grids of either program (the payload names it: `program` is
 * "first" or "second"), `onSelectOtherLoop(id)` for the loops of program B, `stale`, `change`, `onRun`,
 * and `continueTo` ({guarantee, target, onContinue} or {reason}, see continueOffer) for the continue flow (#399).
 * Returns {analysis} of the shown result.
 */
export function renderEquivalence(container, result, options = {}) {
  const equivalence = result.equivalence;
  const view = options.programView || { which: "first" };
  // the choices of each program's analysis card (lens, variable, view), kept while switching between them
  const views = view.cards?.result === result ? view.cards : (view.cards = { result, first: { result }, second: { result: equivalence.other } });
  const children = [];
  if (options.stale) {
    children.push(h("div", { class: "notice rv-stale", role: "status" }, icon("info"),
      h("span", {}, staleText(options.change)),
      options.onRun ? h("button", { type: "button", class: "button small", onclick: () => options.onRun() },
        "Check Again") : null));
  }
  children.push(verdictCard(result, equivalence, options.stale ? null : options.continueTo),
    distributionsCard(result, equivalence, options));
  const holder = h("div", { class: "eq-program-result" });
  let analysis = null;
  const choose = (which) => {
    view.which = which;
    press(switcher, which);
    analysis?.stop?.();
    const shown = which === "second" ? equivalence.other || {} : result;
    // the editor callbacks (lines, loops) belong to the first program's editor
    const own = which === "first" ? options
      : { strategyTitles: options.strategyTitles, onSelectLoop: options.onSelectOtherLoop, maxJointAxis: options.maxJointAxis };
    const loadJoint = options.loadJoint
      ? (payload, signal, onPartial) => options.loadJoint({ ...payload, program: which }, signal, onPartial) : undefined;
    ({ analysis } = renderResult(holder, shown, { ...own, stale: false, loadJoint, analysisState: views[which] }));
  };
  const switcher = segmented("Result of", PROGRAMS, choose);
  children.push(h("div", { class: "eq-program-head" }, h("h3", { class: "rv-card-title" }, "Result of"), switcher), holder);
  container.replaceChildren(h("div", { class: "result-view eq-view", "data-stale": options.stale ? "true" : "false" },
    ...children));
  choose(view.which === "second" ? "second" : "first");
  return { get analysis() { return analysis; } };
}
