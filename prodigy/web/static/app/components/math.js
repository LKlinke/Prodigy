// Rendering of closed forms with KaTeX (falls back to the plain text form).

// Inline mode lets KaTeX break long sums after "+", display mode would not.
export function renderMath(element, latex, fallback, { display = false } = {}) {
  element.textContent = "";
  const katex = globalThis.katex;
  if (katex && latex) {
    try {
      // \displaystyle keeps fractions full size while the inline mode still allows line breaks
      katex.render(display ? latex : `\\displaystyle ${latex}`, element,
        { throwOnError: false, displayMode: display, output: "htmlAndMathml", strict: "ignore" });
      return true;
    } catch {
      // fall through to the text form
    }
  }
  element.textContent = fallback ?? latex ?? "";
  element.classList.add("math-fallback");
  return false;
}
