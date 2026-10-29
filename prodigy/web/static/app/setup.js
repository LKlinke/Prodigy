// The Setup panel: input distribution, one strategy per loop, analysis options.
import { h, icon } from "./components/dom.js";

export const GUI_DEFAULTS = { unroll: { iterations: 25 }, evt_iterate: { iterations: 25 } };

const GROUPS = [
  { title: "Bounds without termination assumption", test: (s) => s.soundness === "lower" },
  { title: "Invariants", test: (s) => s.soundness === "upper" },
  { title: "More", test: () => true },
];

const SOUNDNESS = {
  lower: { glyph: "≥", text: "lower bound" },
  upper: { glyph: "≤", text: "upper bound" },
  exact: { glyph: "✓", text: "exact" },
};

export function defaultParams(strategy) {
  const params = {};
  for (const param of strategy.params) {
    if (param.default !== null && param.default !== undefined && !param.optional) params[param.name] = param.default;
  }
  return { ...params, ...(GUI_DEFAULTS[strategy.name] || {}) };
}

export function defaultDecision(strategies) {
  const unroll = strategies.find((s) => s.name === "unroll") || strategies[0];
  return { strategy: unroll.name, params: defaultParams(unroll) };
}

/** Problems with a parameter value (empty string if fine). */
export function validateParam(param, raw) {
  const text = raw === null || raw === undefined ? "" : String(raw).trim();
  if (text === "") return param.optional ? "" : "Required.";
  if (param.kind === "int") {
    if (!/^\d+$/.test(text)) return "Enter a whole number.";
    if (param.minimum !== null && Number(text) < param.minimum) return `At least ${param.minimum}.`;
  }
  if (param.kind === "probability") {
    const match = text.match(/^(\d+(?:\.\d*)?|\.\d+)(?:\s*\/\s*(\d+))?$/);
    if (!match) return "Enter a probability such as 0.99 or 99/100.";
    const value = match[2] ? Number(match[1]) / Number(match[2]) : Number(match[1]);
    if (!(value >= 0 && value <= 1)) return "Between 0 and 1.";
  }
  return "";
}

function observeBefore(source, line) {
  const lines = source.split("\n").slice(0, Math.max(0, (line || 1) - 1));
  return lines.some((text) => /\bobserve\s*\(/.test(text.replace(/(#|\/\/).*$/, "")));
}

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;

function paramField(param, value, onInput) {
  const id = nextId(`param-${param.name}`);
  const errorId = `${id}-error`;
  const label = h("label", { for: id }, param.label, param.optional ? h("span", { class: "optional" }, " (optional)") : null);
  let control;
  if (param.kind === "pgcl") {
    control = h("textarea", { id, class: "mono", rows: "7", spellcheck: "false", placeholder: "nat x;\nif (x = 1) { … } else { skip }" });
  } else if (param.kind === "choice") {
    control = h("select", { id }, ...param.choices.map((choice) => h("option", { value: choice }, choice)));
  } else {
    const numeric = param.kind === "int";
    control = h("input", {
      id, class: `field${numeric ? "" : " mono"}`, type: numeric ? "number" : "text",
      min: numeric && param.minimum !== null ? String(param.minimum) : null, step: numeric ? "1" : null,
      inputmode: numeric ? "numeric" : null, spellcheck: "false", autocomplete: "off",
      placeholder: param.kind === "probability" ? "0.99" : param.kind === "expression" ? "1/(1 - c/2)" : "",
    });
  }
  control.value = value ?? "";
  control.setAttribute("aria-describedby", `${errorId} ${id}-help`);
  const error = h("p", { class: "field-error", id: errorId, "aria-live": "polite" });
  const update = () => {
    const problem = validateParam(param, control.value);
    error.textContent = problem;
    control.setAttribute("aria-invalid", problem ? "true" : "false");
    onInput(param.kind === "int" && /^\d+$/.test(control.value.trim()) ? Number(control.value) : control.value);
  };
  control.addEventListener("input", update);
  const problem = value !== undefined && value !== "" ? validateParam(param, control.value) : "";
  error.textContent = problem;
  if (problem) control.setAttribute("aria-invalid", "true");
  return h("div", { class: "form-row" }, label, control, error,
    param.help ? h("p", { class: "help", id: `${id}-help` }, param.help) : null);
}

function strategySelect(strategies, current, onChange) {
  const id = nextId("strategy");
  const select = h("select", { id });
  const used = new Set();
  for (const group of GROUPS) {
    const members = strategies.filter((s) => !used.has(s.name) && group.test(s));
    if (!members.length) continue;
    members.forEach((s) => used.add(s.name));
    select.append(h("optgroup", { label: group.title }, ...members.map((s) => h("option", { value: s.name }, s.title))));
  }
  select.value = current;
  select.addEventListener("change", () => onChange(select.value));
  return h("div", { class: "form-row" }, h("label", { for: id }, "Strategy"), select);
}

function loopCard(loop, ctx) {
  const { strategies, setup, onChange, source } = ctx;
  const byName = Object.fromEntries(strategies.map((s) => [s.name, s]));
  const key = String(loop.id);
  if (!setup.decisions[key] || !byName[setup.decisions[key].strategy]) {
    setup.decisions[key] = defaultDecision(strategies);
  }
  const card = h("article", { class: "card", id: `loop-card-${loop.id}`, "aria-labelledby": `loop-title-${loop.id}` });

  const render = () => {
    const decision = setup.decisions[key];
    const strategy = byName[decision.strategy];
    const soundness = SOUNDNESS[strategy.soundness];
    const params = strategy.params.filter((p) => p.kind !== "file");
    const warnings = [];
    if (strategy.requires_unconditioned && observeBefore(source, loop.line)) {
      warnings.push(h("p", { class: "notice warning" }, icon("warning"),
        h("span", {}, "This strategy does not support observe statements before the loop.")));
    }
    card.replaceChildren(
      h("div", { class: "loop-head" },
        h("h4", { id: `loop-title-${loop.id}`, class: "loop-name" }, `Loop ${loop.id} `,
          h("span", { class: "loop-line" }, loop.line ? `· line ${loop.line}` : "")),
        soundness ? h("span", { class: "chip", title: `Nominal guarantee: ${soundness.text}` },
          h("span", { "aria-hidden": "true" }, soundness.glyph), soundness.text) : null),
      h("div", { class: "loop-code", title: `while (${loop.cond})` }, `while (${loop.cond})`),
      strategySelect(strategies, decision.strategy, (name) => {
        setup.decisions[key] = { strategy: name, params: defaultParams(byName[name]) };
        render();
        onChange();
        card.querySelector("select")?.focus();
      }),
      h("p", { class: "help" }, strategy.description),
      ...warnings,
      params.length ? h("div", { class: "params" }, ...params.map((param) => paramField(param, decision.params[param.name], (value) => {
        decision.params[param.name] = value;
        onChange();
      }))) : null,
    );
  };
  render();
  return card;
}

function toggleRow(label, help, checked, onToggle) {
  const id = nextId("toggle");
  const input = h("input", { type: "checkbox", id, role: "switch" });
  input.checked = Boolean(checked);
  input.addEventListener("change", () => onToggle(input.checked));
  return h("div", { class: "form-row" },
    h("label", { class: "toggle", for: id }, h("span", {}, label), input, h("span", { class: "switch", "aria-hidden": "true" })),
    help ? h("p", { class: "help" }, help) : null);
}

function numberRow(label, help, value, { min, max }, onValue) {
  const id = nextId("number");
  const input = h("input", { type: "number", id, class: "field", min: String(min), max: String(max), step: "1", inputmode: "numeric" });
  input.value = value;
  input.style.maxWidth = "120px";
  input.addEventListener("change", () => {
    const n = Math.max(min, Math.min(max, Math.round(Number(input.value) || min)));
    input.value = n;
    onValue(n);
  });
  return h("div", { class: "form-row" }, h("label", { for: id }, label), input, help ? h("p", { class: "help" }, help) : null);
}

function selectRow(label, help, value, choices, onValue) {
  const id = nextId("select");
  const select = h("select", { id }, ...choices.map(([v, text]) => h("option", { value: v }, text)));
  select.value = value;
  select.addEventListener("change", () => onValue(select.value));
  return h("div", { class: "form-row" }, h("label", { for: id }, label), select, help ? h("p", { class: "help" }, help) : null);
}

function engineControl(meta, options, onChange) {
  const group = h("div", { class: "segmented", role: "radiogroup", "aria-label": "Engine" });
  for (const engine of meta.engines.filter((e) => e.available)) {
    const button = h("button", { type: "button", role: "radio", "aria-checked": String(options.engine === engine.name),
      "aria-pressed": String(options.engine === engine.name) }, engine.label);
    button.addEventListener("click", () => {
      options.engine = engine.name;
      for (const b of group.children) {
        const on = b === button;
        b.setAttribute("aria-checked", String(on));
        b.setAttribute("aria-pressed", String(on));
      }
      onChange();
    });
    group.append(button);
  }
  return h("div", { class: "form-row" }, h("span", { class: "label" }, "Engine"), group,
    h("p", { class: "help" }, "GiNaC is usually much faster; SymPy is the reference implementation."));
}

/** Renders the setup panel. ctx: {outline, setup, strategies, meta, source, onChange, focusLoop} */
export function renderSetup(container, ctx) {
  const { outline, setup, meta, onChange } = ctx;
  const inputId = nextId("input");
  const input = h("input", { id: inputId, class: "field mono", type: "text", spellcheck: "false", autocomplete: "off",
    placeholder: "1   (every variable starts at 0)" });
  input.value = setup.input || "";
  input.addEventListener("input", () => { setup.input = input.value; onChange(); });

  const loops = outline?.loops || [];
  const loopCards = loops.map((loop) => loopCard(loop, ctx));
  const known = new Set(loops.map((l) => String(l.id)));
  for (const key of Object.keys(setup.decisions)) {
    if (key !== "*" && !known.has(key)) delete setup.decisions[key];
  }

  const options = setup.options;
  const optionsBody = h("div", { class: "disclosure-body options-grid" },
    engineControl(meta, options, onChange),
    toggleRow("Condition on observations", "Divide by the probability that all observe statements hold.", options.normalize,
      (v) => { options.normalize = v; onChange(); }),
    toggleRow("Simplify expressions", "Slower, but gives shorter closed forms.", options.simplification,
      (v) => { options.simplification = v; onChange(); }),
    toggleRow("Record each step", "Keep the distribution after every top-level instruction.", options.recordSteps,
      (v) => { options.recordSteps = v; onChange(); }),
    numberRow("Coefficients per marginal", "How many values the charts and tables show.", options.coefficients,
      { min: 0, max: meta.limits.max_coefficients }, (v) => { options.coefficients = v; onChange(); }),
    numberRow("Time limit (seconds)", `At most ${meta.limits.time_limit} s on this server.`, options.timeLimit,
      { min: 1, max: meta.limits.time_limit }, (v) => { options.timeLimit = v; onChange(); }),
    selectRow("EVT synthesis strategy", "Templates and positivity checks used by “Synthesise EVT invariant”.",
      options.synthesis, meta.synthesis_strategies.map((s) => [s, s.replace(/_/g, " ")]), (v) => { options.synthesis = v; onChange(); }),
    selectRow("Solver", "For equation systems during invariant checks.", options.solver,
      meta.solvers.map((s) => [s, s === "z3" ? "Z3" : "SymPy"]), (v) => { options.solver = v; onChange(); }),
  );
  const optionsDisclosure = h("details", { class: "disclosure" }, h("summary", {}, "Analysis options"), optionsBody);
  optionsDisclosure.open = Boolean(ctx.optionsOpen);
  optionsDisclosure.addEventListener("toggle", () => ctx.onOptionsToggle?.(optionsDisclosure.open));

  const loopsSection = h("section", { class: "section", "aria-labelledby": "loops-title" },
    h("div", { class: "section-header" }, h("h3", { class: "section-title", id: "loops-title" }, "Loops"),
      h("span", { class: "section-note" }, loops.length ? `${loops.length}` : "")),
    ctx.outlineStale ? h("p", { class: "notice warning" }, icon("warning"),
      h("span", {}, "The program does not parse right now; the loops are those of the last version that did.")) : null,
    loops.length ? loopCards : h("p", { class: "help" },
      outline ? "This program has no while loops; its semantics is computed exactly." : "Loops appear here once the program parses."));

  container.replaceChildren(
    h("section", { class: "section", "aria-labelledby": "input-title" },
      h("div", { class: "section-header" }, h("h3", { class: "section-title", id: "input-title" }, "Input")),
      h("div", { class: "form-row" }, h("label", { for: inputId }, "Initial distribution"), input,
        h("p", { class: "help" }, "A generating function over the program variables, e.g. x^3 for x = 3, or (x + x^2)/2."))),
    loopsSection,
    h("section", { class: "section" }, optionsDisclosure),
  );
}

export function focusLoopCard(container, loopId) {
  const card = container.querySelector(`#loop-card-${loopId}`);
  if (!card) return;
  card.scrollIntoView({ block: "center", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  card.classList.add("focused");
  setTimeout(() => card.classList.remove("focused"), 1600);
  card.querySelector("select")?.focus({ preventScroll: true });
}

/** Problems that prevent running (shown next to the Analyze button). */
export function setupProblems(outline, setup, strategies) {
  const byName = Object.fromEntries(strategies.map((s) => [s.name, s]));
  const problems = [];
  for (const loop of outline?.loops || []) {
    const decision = setup.decisions[String(loop.id)];
    const strategy = decision && byName[decision.strategy];
    if (!strategy) {
      problems.push({ loop: loop.id, message: `Choose a strategy for loop ${loop.id}.` });
      continue;
    }
    for (const param of strategy.params.filter((p) => p.kind !== "file")) {
      const problem = validateParam(param, decision.params[param.name]);
      if (problem) problems.push({ loop: loop.id, message: `Loop ${loop.id}: ${param.label}: ${problem}` });
    }
  }
  return problems;
}
