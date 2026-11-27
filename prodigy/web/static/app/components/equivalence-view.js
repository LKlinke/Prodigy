// The result of comparing two programs (issue #154): whether they are equivalent for every input. The verdict,
// why, the parameter values of a synthesis, and for "not equivalent" the witness that proves it. Compare runs no
// program on an input of its own, so there are no output distributions here. Needs only the result JSON.
import { h, icon } from "./dom.js";
import { formatSeconds, formatValue, stateText } from "./format.js";
import { expressionBox, guaranteeBadge } from "./parts.js";
import {
  CONTINUE_GUARANTEE, checkLines, constraintTexts, inputText, intervalText, relationText, verdictOf,
} from "./equivalence-model.js";
import { staleText } from "../staleness.js";

function verdictCard(equivalence, continueTo = null) {
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

/** The witness: on one input, one output state whose probabilities in A and B provably differ. */
function witnessTable(witness) {
  // an interval whose ends round to the same decimals gets more digits, so that it never reads as one value
  const digits = (side) => [6, 9, 12].find((d) => side.lower === side.upper
    || formatValue(side.lower, side.lower_float, d) !== formatValue(side.upper, side.upper_float, d)) ?? 12;
  const row = (label, side) => h("tr", {}, h("th", { scope: "row" }, label),
    h("td", { class: "num", title: side.lower === side.upper ? side.lower : `[${side.lower}, ${side.upper}]` },
      intervalText(side, (exact, number) => formatValue(exact, number, digits(side)))),
    h("td", {}, guaranteeBadge(side.kind)));
  return h("div", { class: "eq-witness" },
    h("h4", { class: "rv-block-title" }, "Witness"),
    h("dl", { class: "eq-witness-facts" },
      h("dt", {}, "Input"), h("dd", { class: "mono" }, inputText(witness.input)),
      h("dt", {}, "Output state"), h("dd", { class: "mono" }, stateText(witness.state))),
    h("div", { class: "table-wrap" }, h("table", { class: "rv-table eq-witness-table" },
      h("caption", { class: "visually-hidden" }, `Probability of the output state ${stateText(witness.state)}`),
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, ""), h("th", { scope: "col" }, "Probability"),
        h("th", { scope: "col" }, "Guarantee"))),
      h("tbody", {}, row("Program A", witness.first), row("Program B", witness.second)))),
    h("p", { class: "rv-note" }, "The two values (or intervals) do not overlap, so the programs differ on this input."));
}

function checkDetails(check) {
  const body = checkLines(check).map((line) => h("p", {}, line));
  if (check.difference) body.push(expressionBox(check.difference, "Difference of the second-order generating functions"));
  return h("details", { class: "disclosure eq-check" },
    h("summary", {}, "How it was checked"),
    h("div", { class: "disclosure-body" },
      h("p", { class: "rv-muted" }, "Prodigy runs both programs on one generating function that stands for every "
        + "input at once (second-order check). Unless that proves them equal, it runs both on one input to look for a "
        + "witness: the solver's counterexample, the input on which an invariant was refuted, or every variable 0."),
      ...body));
}

/**
 * Renders a comparison result (with an `equivalence` part) into `container`. options: `stale`, `change`, `onRun`
 * (Check Again), and `continueTo` ({guarantee, target, onContinue} or {reason}, see continueOffer) for the continue
 * flow (#399).
 */
export function renderEquivalence(container, result, options = {}) {
  const children = [];
  if (options.stale) {
    children.push(h("div", { class: "notice rv-stale", role: "status" }, icon("info"),
      h("span", {}, staleText(options.change)),
      options.onRun ? h("button", { type: "button", class: "button small", onclick: () => options.onRun() },
        "Check Again") : null));
  }
  children.push(verdictCard(result.equivalence, options.stale ? null : options.continueTo));
  container.replaceChildren(h("div", { class: "result-view eq-view", "data-stale": options.stale ? "true" : "false" },
    ...children));
}
