// The timeline of the analysis card (issue #311): one position per recorded step of the trace
// (prodigy/analysis/trace.py), the recorded iterates of a loop as positions inside its segment, and
// the final result as the last position. The model functions are pure (no DOM), so that they can be
// tested on their own; `timelineControl` draws the control that moves along them.
import { h } from "./dom.js";
import { reducedMotion } from "./motion.js";

/**
 * The positions of `result`: {positions, segments, final}.
 *   positions: {kind: "step", step} (after the step; a loop's step is its exit),
 *              {kind: "iterate", step, iterate, iteration, loop} (an iterate of the loop of `step`),
 *              {kind: "final"} (the result itself, always the last one).
 *   segments:  {loop, step, from, to} the positions of one loop (its iterates and its exit).
 */
export function timelinePositions(result) {
  const steps = result?.steps || [];
  const positions = [];
  const segments = [];
  for (const step of steps) {
    const loop = step.loop;
    const from = positions.length;
    (loop?.iterates || []).forEach((it, k) => positions.push({
      kind: "iterate", step: step.index, iterate: k, iteration: it.iteration, loop: loop.id ?? null,
    }));
    positions.push({ kind: "step", step: step.index });
    if (loop) segments.push({ loop: loop.id ?? null, step: step.index, from, to: positions.length - 1 });
  }
  positions.push({ kind: "final" });
  return { positions, segments, final: positions.length - 1 };
}

function stepOf(result, position) {
  return position && position.kind !== "final" ? (result?.steps || [])[position.step] || null : null;
}

/**
 * What the analysis shows at `position`: {step, snapshot, iterate, failed}. `snapshot` is the
 * distribution of the trace (the one after the step, or before it if the step failed); null at the
 * final result, whose distribution is `result.distribution`.
 */
export function positionData(result, position) {
  const step = stepOf(result, position);
  if (!step) return { step: null, snapshot: null, iterate: null, failed: false };
  if (position.kind === "iterate") {
    const iterate = step.loop.iterates[position.iterate];
    return { step, snapshot: iterate.distribution, iterate, failed: false };
  }
  return { step, snapshot: step.after || step.before, iterate: null, failed: !step.after };
}

function loopName(id) {
  return id !== null && id !== undefined ? `loop ${id}` : "the loop";
}

/** "after step 9 of 14 · line 4", "in step 5 of 14 · loop 0, iterate 3 of 8", "final result". */
export function positionLabel(result, position) {
  const steps = result?.steps || [];
  if (!position || position.kind === "final") return "final result";
  const step = steps[position.step];
  const where = step?.kind === "normalize" ? "end of the program" : step?.line ? `line ${step.line}` : "generated";
  const count = `step ${position.step + 1} of ${steps.length}`;
  if (position.kind === "iterate") {
    const loop = step.loop;
    const of = loop.iterations ? ` of ${loop.iterations}` : "";
    return `in ${count} · ${loopName(loop.id)}, iteration ${position.iteration}${of}`;
  }
  if (!step.after) return `${count} failed · ${where}`;
  return `after ${count} · ${where}${step.loop ? ` · ${loopName(step.loop.id)} done` : ""}`;
}

/**
 * Why the marginal of `variable` is missing from the snapshot `snap`, or null if it is there. A
 * marginal that was not recorded is unknown, not 0 (prodigy/analysis/trace.py), so it must never
 * be drawn as bars of height 0.
 */
export function marginalGap(snap, variable) {
  const table = snap?.marginals?.[variable];
  if (table) return table.error ? `not available: ${table.error}` : null;
  return snap?.marginals_not_recorded?.[variable] || snap?.marginals_omitted || "not recorded";
}

/** The size of the closed form (characters) at each position, null where it is unknown. */
export function sizeSeries(result, model) {
  return model.positions.map((position) => {
    if (position.kind === "final") {
      const closed = result?.distribution?.closed_form;
      return typeof closed === "string" ? closed.length : null;
    }
    const size = positionData(result, position).snapshot?.size;
    return Number.isFinite(size) ? size : null;
  });
}

/**
 * The time of each position in seconds: a step's own time without its nested steps (so that the
 * bars add up to the analysis), null for iterates (they are not timed one by one) and the final
 * result. Snapshot time is excluded by the trace.
 */
export function timeSeries(result, model) {
  return model.positions.map((position) => {
    if (position.kind !== "step") return null;
    const step = stepOf(result, position);
    const time = step?.self_time ?? step?.time;
    return Number.isFinite(time) ? time : null;
  });
}

/** The position to show for a step index (its exit position), or the final result. */
export function positionOfStep(model, stepIndex) {
  const found = model.positions.findIndex((p) => p.kind === "step" && p.step === stepIndex);
  return found >= 0 ? found : model.final;
}

/** Index of the next (dir 1) or previous (dir -1) position on a breakpoint line, else the end. */
export function breakpointTarget(result, model, current, breakpoints, dir = 1) {
  for (let i = current + dir; i >= 0 && i < model.positions.length; i += dir) {
    const step = stepOf(result, model.positions[i]);
    if (model.positions[i].kind === "step" && step?.line && breakpoints.has(step.line)) return i;
  }
  return dir > 0 ? model.final : 0;
}

// ---- the control ----------------------------------------------------------------------------------

const PACE_MS = 650;

/**
 * The timeline control: previous, play/pause, next, a slider with one stop per position (the loop
 * segments shaded and labelled) and Final Result. `go(index)` is called for every move; `update(index)`
 * moves the control without calling it. Keys on the slider: ←/→ one position, Home/End start and
 * final result, Page Up/Down ten positions. Never plays by itself.
 * Returns {element, update(index), stop()}.
 */
export function timelineControl({ model, index, go, label, breakpoints = new Set(), stepLines = [] }) {
  const n = model.positions.length;
  const x = (i) => (n <= 1 ? 50 : (i / (n - 1)) * 100);
  let current = index;
  let timer = null;

  const done = h("span", { class: "tl-done" });
  const thumb = h("span", { class: "tl-thumb" });
  const bands = model.segments.map((segment) => {
    const left = Math.max(0, x(segment.from) - (n > 1 ? 50 / (n - 1) : 0));
    const right = Math.min(100, x(segment.to) + (n > 1 ? 50 / (n - 1) : 0));
    return h("span", { class: "tl-loop", style: `left:${left}%;width:${Math.max(0.8, right - left)}%`, "aria-hidden": "true" },
      h("span", { class: "tl-loop-label" }, loopName(segment.loop)));
  });
  // one tick per step (iterates are the shaded band); breakpoint lines get a dot
  const ticks = model.positions.map((position, i) => (position.kind === "step"
    ? h("span", { class: `tl-tick${breakpoints.has(stepLines[i]) ? " tl-break" : ""}`, style: `left:${x(i)}%`, "aria-hidden": "true" })
    : null)).filter(Boolean);
  const finalMark = h("span", { class: "tl-final-mark", style: `left:${x(model.final)}%`, "aria-hidden": "true" });
  const track = h("div", { class: "tl-track", role: "slider", tabindex: "0", "aria-label": "Position in the analysis",
    "aria-valuemin": "1", "aria-valuemax": String(n), "data-focus-key": "timeline" },
  h("span", { class: "tl-rail", "aria-hidden": "true" }), ...bands, done, ...ticks, finalMark, thumb);

  const icon = (text) => h("span", { "aria-hidden": "true" }, text);
  const prev = h("button", { type: "button", class: "button small tl-button", "aria-label": "Previous position",
    title: "Previous position (←)", "data-focus-key": "tl-prev" }, icon("◀"));
  const next = h("button", { type: "button", class: "button small tl-button", "aria-label": "Next position",
    title: "Next position (→)", "data-focus-key": "tl-next" }, icon("▶"));
  const play = h("button", { type: "button", class: "button small tl-play", "aria-pressed": "false", "data-focus-key": "tl-play" });
  const final = h("button", { type: "button", class: "button small", title: "The final result (End)", "data-focus-key": "tl-final" },
    "Final Result");

  const setPlaying = (playing) => {
    play.replaceChildren(icon(playing ? "❚❚" : "▶"), playing ? " Pause" : " Play");
    play.setAttribute("aria-pressed", String(playing));
    play.title = playing ? "Pause" : "Play the positions from here (from the start at the final result)";
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    setPlaying(false);
  };
  const move = (i) => {
    const target = Math.max(0, Math.min(n - 1, i));
    if (target !== current) go(target);
  };
  const update = (i) => {
    current = i;
    thumb.style.left = `${x(i)}%`;
    done.style.width = `${x(i)}%`;
    track.setAttribute("aria-valuenow", String(i + 1));
    track.setAttribute("aria-valuetext", `${i + 1} of ${n}: ${label(i)}`);
    prev.disabled = i === 0;
    next.disabled = i === n - 1;
    final.disabled = i === n - 1;
  };

  prev.addEventListener("click", () => { stop(); move(current - 1); });
  next.addEventListener("click", () => { stop(); move(current + 1); });
  final.addEventListener("click", () => { stop(); move(n - 1); });
  play.addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (current >= n - 1) move(0);
    setPlaying(true);
    // with reduced motion the views jump from one position to the next (no transitions), slower
    timer = setInterval(() => {
      if (!track.isConnected || current >= n - 1) { stop(); return; }
      move(current + 1);
      if (current >= n - 1) stop();
    }, reducedMotion() ? PACE_MS * 1.6 : PACE_MS);
  });

  track.addEventListener("keydown", (event) => {
    const moves = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 10, PageDown: -10 };
    let target = null;
    if (event.key in moves) target = current + moves[event.key];
    else if (event.key === "Home") target = 0;
    else if (event.key === "End") target = n - 1;
    else if (event.key === " ") { event.preventDefault(); play.click(); return; }
    if (target === null) return;
    event.preventDefault();
    stop();
    move(target);
  });
  const fromPointer = (event) => {
    const rect = track.getBoundingClientRect();
    const t = rect.width ? (event.clientX - rect.left) / rect.width : 0;
    move(Math.round(Math.max(0, Math.min(1, t)) * (n - 1)));
  };
  track.addEventListener("pointerdown", (event) => {
    stop();
    track.setPointerCapture?.(event.pointerId);
    track.focus({ preventScroll: true });
    fromPointer(event);
    const moveHandler = (e) => fromPointer(e);
    const up = () => {
      track.removeEventListener("pointermove", moveHandler);
      track.removeEventListener("pointerup", up);
      track.removeEventListener("pointercancel", up);
    };
    track.addEventListener("pointermove", moveHandler);
    track.addEventListener("pointerup", up);
    track.addEventListener("pointercancel", up);
  });

  setPlaying(false);
  update(index);
  const element = h("div", { class: "tl", role: "group", "aria-label": "Timeline of the analysis" },
    h("div", { class: "tl-row" }, prev, play, next, track, final),
    h("p", { class: "tl-hint" }, "← → one position · Home / End: start and final result",
      model.segments.length ? " · shaded: a loop, its recorded iterates are the positions inside it" : ""));
  return { element, update, stop, playing: () => Boolean(timer) };
}
