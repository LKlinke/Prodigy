// A bounded box for a formula or a long expression.
//
// The box keeps its size while the expression grows or changes: the content scrolls inside it in
// both directions (never the page), and when it is taller than the box a "Show more" button
// enlarges the box (still bounded, still scrolling). The scroll area takes the keyboard focus only
// while it has something to scroll.
import { h } from "./dom.js";
import { renderMath } from "./math.js";

let counter = 0;

/**
 * formulaBox({ latex, text, kind, label, expanded, onToggle, className })
 *   kind: "math" renders `latex` with KaTeX (falling back to `text`); "text" shows `text` as code.
 *   label: accessible name of the scroll area, e.g. "Generating function".
 *   expanded / onToggle(expanded): initial state and a callback, so that a caller can keep the
 *   state across re-renderings.
 */
export function formulaBox({ latex = null, text = "", kind = "math", label = "Expression",
  expanded = false, onToggle = null, className = "" } = {}) {
  counter += 1;
  const id = `formula-${counter}`;
  const content = kind === "math" ? h("div", { class: "formula-content formula-math" })
    : h("code", { class: "formula-content formula-text" }, text ?? "");
  if (kind === "math") renderMath(content, latex, text);
  const scroller = h("div", { class: "formula-scroll", id, role: "region", "aria-label": label }, content);
  const frame = h("div", { class: "formula-frame" }, scroller);
  // hidden until the box is measured
  const toggle = h("button", { type: "button", class: "link-button formula-toggle", "aria-controls": id, "aria-expanded": "false", hidden: true });
  const foot = h("div", { class: "formula-foot" }, toggle);
  const box = h("div", { class: `formula-box fb-${kind} ${className}`.trim(), "data-expanded": "false",
    "data-overflow": "false" }, frame, foot);

  let isExpanded = Boolean(expanded);
  let collapsedHeight = 0;
  const update = () => {
    if (!scroller.isConnected) return;
    const { scrollHeight, clientHeight, scrollWidth, clientWidth, scrollTop } = scroller;
    if (!isExpanded) collapsedHeight = clientHeight;
    // taller than the collapsed box: offer "Show more" (and "Show less" once expanded)
    const tall = isExpanded ? content.offsetHeight > collapsedHeight + 1 || collapsedHeight === 0 : scrollHeight > clientHeight + 1;
    const scrollable = scrollHeight > clientHeight + 1 || scrollWidth > clientWidth + 1;
    box.dataset.overflow = String(tall);
    box.dataset.atEnd = String(scrollTop + clientHeight >= scrollHeight - 1);
    // the toggle shows only when there is more to show
    toggle.hidden = !tall;
    if (scrollable) scroller.setAttribute("tabindex", "0");
    else scroller.removeAttribute("tabindex");
  };
  const setExpanded = (value, notify = true) => {
    isExpanded = value;
    box.dataset.expanded = String(value);
    toggle.setAttribute("aria-expanded", String(value));
    toggle.textContent = value ? "Show less" : "Show more";
    if (!value) scroller.scrollTop = 0;
    update();
    if (notify) onToggle?.(value);
  };
  toggle.addEventListener("click", () => setExpanded(!isExpanded));
  scroller.addEventListener("scroll", () => {
    box.dataset.atEnd = String(scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 1);
  }, { passive: true });
  setExpanded(isExpanded, false);
  if (typeof ResizeObserver !== "undefined") {
    // fires once the box is laid out, and again when its width, the fonts or the content change
    const observer = new ResizeObserver(() => update());
    observer.observe(scroller);
    observer.observe(content);
  } else {
    requestAnimationFrame(update);
  }
  return box;
}
