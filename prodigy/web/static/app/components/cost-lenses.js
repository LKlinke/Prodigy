// The lenses Size and Time of the analysis card (analysis-card.js): the size of the closed form and the time
// per position of the timeline, with the table of all positions and the trace's hints.
import { positionChart } from "./charts.js";
import { h } from "./dom.js";
import { BOUND_TEXT, formatNumber, formatSeconds } from "./format.js";
import { disclosure, facts, where } from "./lens-parts.js";
import { press, segmented } from "./parts.js";
import { positionData, positionOfStep, sizeSeries, timeSeries } from "./timeline.js";

function stepWhere(result, model, index) {
  const position = model.positions[index];
  if (position.kind === "final") return "the final result";
  const step = result.steps[position.step];
  const text = step.summary ? ` (${step.summary.length > 32 ? `${step.summary.slice(0, 31)}…` : step.summary})` : "";
  const loop = position.kind === "iterate" ? `, iteration ${position.iteration}` : "";
  return `step ${position.step + 1}${loop}, ${where(step)}${text}`;
}

function largest(values) {
  let best = -1;
  values.forEach((v, i) => { if (Number.isFinite(v) && (best < 0 || v > values[best])) best = i; });
  return best;
}

function noSteps(ctx, what) {
  return h("div", { class: "ac-empty" },
    h("p", {}, `${what} needs the recorded steps of the analysis; this result has none.`),
    ctx.options.onRecord
      ? h("p", {}, h("button", { type: "button", class: "button", onclick: () => ctx.options.onRecord() }, "Record Steps and Analyze"))
      : null);
}

function positionsTable(ctx, sizes, times) {
  const { result, model, index, go } = ctx;
  const rows = model.positions.map((position, i) => {
    const data = positionData(result, position);
    const kind = position.kind === "final" ? result.bounds?.kind || "unknown" : data.failed ? "error" : data.snapshot?.bound || "unknown";
    const text = BOUND_TEXT[kind] || { glyph: "×", label: "Failed" };
    const name = position.kind === "final" ? "Final result"
      : position.kind === "iterate" ? `Iteration ${position.iteration}` : (data.step.summary || data.step.instruction || "");
    return h("tr", { "aria-current": i === index ? "true" : null },
      h("td", { class: "num" }, h("button", { type: "button", class: "link-button", onclick: () => go(i),
        "aria-label": `Show position ${i + 1}` }, String(i + 1))),
      h("td", { class: "num" }, data.step?.line ? String(data.step.line) : "–"),
      h("td", { class: "ac-cell-text mono", style: data.step?.depth ? `padding-left:${8 + data.step.depth * 12}px` : null }, name),
      h("td", { "data-kind": kind, class: "ac-cell-kind" }, h("span", { "aria-hidden": "true" }, text.glyph), h("span", { class: "visually-hidden" }, text.label)),
      h("td", { class: "num" }, Number.isFinite(sizes[i]) ? sizes[i].toLocaleString("en-US") : "—"),
      h("td", { class: "num" }, Number.isFinite(times[i]) ? formatSeconds(times[i]) : "—"));
  });
  return disclosure("Table of all positions", h("div", { class: "table-wrap ac-positions" }, h("table", { class: "rv-table" },
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "#"), h("th", { scope: "col" }, "Line"), h("th", { scope: "col" }, "Instruction"),
      h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Guarantee")), h("th", { scope: "col" }, "Size"), h("th", { scope: "col" }, "Time"))),
    h("tbody", {}, ...rows))));
}

function hintNote(ctx, kind) {
  const hint = (ctx.result.trace?.hints || []).find((item) => item.kind === kind);
  if (!hint) return null;
  const target = hint.step !== null && hint.step !== undefined ? positionOfStep(ctx.model, hint.step) : null;
  return h("p", { class: "rv-note" }, hint.text, " ",
    target !== null ? h("button", { type: "button", class: "link-button", onclick: () => ctx.go(target) }, "Show this step") : null);
}

function chartWidth(ctx) {
  return Math.max(300, (ctx.body.clientWidth || ctx.element.clientWidth || 600) - 4);
}

export function sizeLens(ctx) {
  const { result, model, index, view } = ctx;
  if (!result.steps?.length) return { figure: noSteps(ctx, "The size of the closed form per step") };
  const sizes = sizeSeries(result, model);
  const scale = view.sizeScale === "log" ? "log" : "linear";
  const unit = (v) => `${Math.round(v).toLocaleString("en-US")} characters`;
  const top = largest(sizes);
  const finalSize = sizes[model.final];
  const chart = positionChart({ values: sizes, segments: model.segments, current: index, kind: "line", scale, width: chartWidth(ctx),
    format: (v) => formatNumber(v, 3), onPick: ctx.go,
    label: `Size of the closed form per position, ${scale === "log" ? "logarithmic" : "linear"} scale. `
      + `${top >= 0 ? `Largest ${unit(sizes[top])} at ${stepWhere(result, model, top)}. ` : ""}`
      + `At this position ${Number.isFinite(sizes[index]) ? unit(sizes[index]) : "unknown"}. The table of all positions lists every value.` });
  const controls = h("div", { class: "rv-controls" },
    segmented("Scale", [["linear", "Linear"], ["log", "Log"]], (key) => { view.sizeScale = key; ctx.redraw(); },
      (key) => (key === "log" ? "Logarithmic scale" : "Linear scale")));
  press(controls.firstChild, scale);
  return {
    controls,
    figure: h("figure", { class: "rv-figure" }, chart,
      h("figcaption", { class: "rv-note" }, "Length of the closed form in characters, the same measure on every engine.")),
    facts: facts(["At this position", Number.isFinite(sizes[index]) ? unit(sizes[index]) : "unknown"],
      top >= 0 ? ["Largest", `${unit(sizes[top])} at ${stepWhere(result, model, top)}`] : null,
      ["Final result", Number.isFinite(finalSize) ? unit(finalSize) : "unknown"]),
    details: [hintNote(ctx, "growth"), positionsTable(ctx, sizes, timeSeries(result, model))],
  };
}

export function timeLens(ctx) {
  const { result, model, index, position, data } = ctx;
  if (!result.steps?.length) return { figure: noSteps(ctx, "The time per step") };
  const times = timeSeries(result, model);
  const slow = largest(times);
  const loopTime = (segment) => result.steps[segment.step]?.time;
  const chart = positionChart({ values: times, segments: model.segments, current: index, kind: "bars", width: chartWidth(ctx),
    format: (v) => formatSeconds(v), onPick: ctx.go,
    segmentLabel: (segment) => `loop ${segment.loop} · ${formatSeconds(loopTime(segment))}`,
    label: `Time per step, without nested steps. ${slow >= 0 ? `Slowest ${formatSeconds(times[slow])} at ${stepWhere(result, model, slow)}. ` : ""}`
      + "Iterates of loops are not timed one by one. The table of all positions lists every value." });
  let here;
  if (position.kind === "iterate") here = ["At this position", `iterates are not timed one by one; the loop took ${formatSeconds(data.step.time)}`];
  else if (position.kind === "final") here = ["At this position", "the final result"];
  else here = ["At this position", formatSeconds(times[index]) || "—",
    data.step.time !== data.step.self_time ? `${formatSeconds(data.step.time)} with nested steps` : ""];
  const snapshot = result.trace?.snapshot_time;
  return {
    figure: h("figure", { class: "rv-figure" }, chart,
      h("figcaption", { class: "rv-note" }, "Time of each step without its nested steps, so the bars add up. A loop is one step; its iterates have no bars of their own.")),
    facts: facts(here,
      slow >= 0 ? ["Slowest", `${formatSeconds(times[slow])} at ${stepWhere(result, model, slow)}`] : null,
      result.timing?.analysis !== undefined ? ["Total analysis", formatSeconds(result.timing.analysis)] : null,
      Number.isFinite(snapshot) ? ["Recording the steps", `${formatSeconds(snapshot)}, not counted in the steps`] : null),
    details: [hintNote(ctx, "time"), positionsTable(ctx, sizeSeries(result, model), times)],
  };
}
