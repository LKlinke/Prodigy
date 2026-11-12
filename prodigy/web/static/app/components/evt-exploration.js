// The exploration of a loop analysed with "Expected visiting times H_N" (strategy evt_iterate, #324).
//
// The iterates H_N of the EVT operator are no output distribution: the coefficient of a state is the
// expected number of times the loop head is reached in that state within N iterations. The card shows,
// for a chosen N, the closed form of H_N and, for the first states, how the coefficients grow with N (a
// table whose cells are shaded by their value; the value is always written, so the shading is only an aid).
// "Use as EVT invariant candidate" switches the loop to Check EVT invariant, prefilled with a guess.
import { h, icon } from "./dom.js";
import { formulaBox } from "./formula-box.js";

let counter = 0;

/** The explorations in a result: [{loop, exploration}] for every loop with one. */
export function explorations(result) {
  const found = [];
  for (const loop of result?.loops || []) {
    const cert = (loop.certificates || []).find((c) => c?.type === "evt_iteration" && c.exploration);
    if (cert) found.push({ loop, exploration: cert.exploration });
  }
  return found;
}

function stateText(state) {
  return Object.entries(state).map(([name, value]) => `${name} = ${value}`).join(", ");
}

function shortNumber(value, exact) {
  if (value === null || value === undefined) return exact;
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.001) ? value.toExponential(2) : value.toFixed(3);
}

/** The candidate offered for Check EVT invariant: the rational fit, else the closed form of the chosen H_N. */
export function candidateFor(exploration, index) {
  if (exploration.guess?.candidate) return { text: exploration.guess.candidate, kind: "fit" };
  const iterate = exploration.iterates[index];
  if (!iterate || iterate.truncated) return null;
  return { text: iterate.closed_form, kind: "iterate", n: iterate.n };
}

/**
 * The card for one loop. options: { onUseCandidate(loopId, candidate), onSelectLoop(id), state: {} }
 * (`state` keeps the chosen N across re-renderings).
 */
export function explorationCard(loop, exploration, options = {}) {
  counter += 1;
  const id = `evt-explore-${counter}`;
  const iterates = exploration.iterates || [];
  const memory = options.state || {};
  let index = Math.min(memory.index ?? iterates.length - 1, iterates.length - 1);

  const sliderId = `${id}-n`;
  const slider = h("input", { type: "range", id: sliderId, min: "0", max: String(Math.max(0, iterates.length - 1)),
    step: "1", class: "evt-slider" });
  slider.value = String(index);
  const nLabel = h("output", { for: sliderId, class: "evt-n num" });
  const formSlot = h("div", { class: "evt-form" });
  const tableSlot = h("div", { class: "evt-table-wrap", tabindex: "0", role: "region", "aria-label": "Coefficients of H_N per state" });
  const candidateSlot = h("div", { class: "evt-candidate" });

  const max = Math.max(0, ...exploration.states.flatMap((row) => row.floats.filter((v) => typeof v === "number" && Number.isFinite(v))));

  const table = () => {
    const head = h("tr", {}, h("th", { scope: "col" }, "State"),
      ...iterates.map((it, i) => h("th", { scope: "col", class: i === index ? "current" : null, "aria-current": i === index ? "true" : null }, `N = ${it.n}`)));
    const rows = exploration.states.map((row) => h("tr", {}, h("th", { scope: "row", class: "mono" }, stateText(row.state)),
      ...row.values.map((exact, i) => {
        const value = row.floats[i];
        const share = max > 0 && typeof value === "number" ? Math.max(0, Math.min(1, value / max)) : 0;
        return h("td", { class: `num${i === index ? " current" : ""}`, title: exact,
          style: `--evt-share: ${Math.round(share * 32)}%` }, shortNumber(value, exact));
      })));
    return h("table", { class: "evt-table" }, h("caption", { class: "visually-hidden" },
      `Expected visiting times per state for N = ${iterates.map((it) => it.n).join(", ")}`),
    h("thead", {}, head), h("tbody", {}, ...rows));
  };

  const candidate = () => {
    const chosen = candidateFor(exploration, index);
    const button = h("button", { type: "button", class: "button", disabled: chosen ? null : "" }, "Use as EVT invariant candidate");
    button.addEventListener("click", () => chosen && options.onUseCandidate?.(loop.id, chosen.text));
    const what = !chosen ? "The closed form of this H_N is too long to use as a candidate; choose a smaller N."
      : chosen.kind === "fit"
        ? "Guessed from the coefficients (a rational function that matches the converged values). Only a proposal: Check EVT invariant decides."
        : `No rational guess fits the coefficients, so the candidate is H_${chosen.n} itself; it is exact only if the loop ends within N iterations. Edit it before checking.`;
    return [
      h("h5", { class: "evt-subtitle" }, "Candidate for an EVT invariant"),
      chosen ? formulaBox({ text: chosen.text, kind: "text", label: "Candidate EVT invariant" }) : null,
      h("p", { class: "help" }, what),
      h("div", { class: "evt-actions" }, button),
    ];
  };

  const update = () => {
    const iterate = iterates[index];
    nLabel.textContent = iterate ? `N = ${iterate.n}` : "";
    slider.setAttribute("aria-valuetext", iterate ? `N = ${iterate.n}` : "");
    formSlot.replaceChildren(...[
      h("h5", { class: "evt-subtitle" }, `Closed form of H_${iterate?.n ?? "N"}`),
      iterate ? formulaBox({ text: iterate.closed_form, kind: "text", label: `H_${iterate.n}` }) : null,
      iterate?.truncated ? h("p", { class: "help" }, "Only the first 3000 characters are shown.") : null].filter(Boolean));
    const scroll = tableSlot.scrollLeft;
    tableSlot.replaceChildren(table());
    tableSlot.scrollLeft = scroll;
    revealColumn();
    candidateSlot.replaceChildren(...candidate().filter(Boolean));
  };
  // keeps the column of the chosen N visible (the table scrolls sideways when it has many columns)
  function revealColumn() {
    requestAnimationFrame(() => {
      const cell = tableSlot.querySelector("thead th.current");
      const first = tableSlot.querySelector("thead th");
      if (!cell || !first) return;
      const box = tableSlot.getBoundingClientRect();
      const rect = cell.getBoundingClientRect();
      const visibleLeft = first.getBoundingClientRect().right;
      if (rect.left < visibleLeft) tableSlot.scrollLeft -= visibleLeft - rect.left;
      else if (rect.right > box.right) tableSlot.scrollLeft += rect.right - box.right;
    });
  }
  slider.addEventListener("input", () => {
    index = Number(slider.value);
    memory.index = index;
    update();
  });
  update();

  const more = exploration.states_total > exploration.states.length
    ? h("p", { class: "help" }, `The first ${exploration.states.length} of ${exploration.states_total} states.`) : null;
  const loopLink = h("button", { type: "button", class: "link-button" }, `Loop ${loop.id}`);
  loopLink.addEventListener("click", () => options.onSelectLoop?.(loop.id));
  return h("section", { class: "rv-card evt-card", "aria-labelledby": `${id}-title` },
    h("div", { class: "evt-head" },
      h("h4", { id: `${id}-title`, class: "evt-title" }, "Expected visiting times"),
      h("span", { class: "chip" }, h("span", { "aria-hidden": "true" }, "◇"), "exploration")),
    h("p", { class: "notice" }, icon("info"), h("span", {},
      loopLink, loop.line ? ` (line ${loop.line}) ` : " ",
      "was explored. These are expected visiting times H_N, not the output distribution: the coefficient of a state is how often the loop head is reached in it within N iterations. The answer above is only a lower bound.")),
    h("div", { class: "evt-slider-row" }, h("label", { for: sliderId }, "Iterate"), slider, nLabel),
    formSlot,
    h("h5", { class: "evt-subtitle" }, "How the coefficients grow with N"),
    tableSlot, more,
    candidateSlot);
}
