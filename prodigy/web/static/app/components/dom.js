// Tiny DOM helpers (no framework).

const SVG_NS = "http://www.w3.org/2000/svg";

/** h("div", {class: "x", onclick: fn}, child, "text", [more]) */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  applyAttrs(el, attrs);
  append(el, children);
  return el;
}

/** Same as h() for SVG elements. */
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyAttrs(el, attrs);
  append(el, children);
  return el;
}

function applyAttrs(el, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2), value);
    } else if (key === "dataset") {
      Object.assign(el.dataset, value);
    } else if (key === "text") {
      el.textContent = value;
    } else {
      el.setAttribute(key, value === true ? "" : String(value));
    }
  }
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function icon(name, extraClass = "") {
  const paths = {
    check: "M3.5 8.5l3 3 6-7",
    warning: "M8 2.5l6 11H2z M8 6.5v3.5 M8 12v.01",
    error: "M8 1.8a6.2 6.2 0 1 0 0 12.4A6.2 6.2 0 0 0 8 1.8z M8 5v3.6 M8 11v.01",
    info: "M8 1.8a6.2 6.2 0 1 0 0 12.4A6.2 6.2 0 0 0 8 1.8z M8 7.2v4 M8 4.9v.01",
    copy: "M5.5 5.5h7v7h-7z M3.5 10.5v-7h7",
    download: "M8 2.5v8 M4.8 7.5L8 10.7l3.2-3.2 M3 13.5h10",
    loop: "M12.5 6.5A4.5 4.5 0 1 0 12 10 M12.8 3.5v3h-3",
  };
  return s("svg", { class: `icon ${extraClass}`.trim(), viewBox: "0 0 16 16", "aria-hidden": "true" },
    s("path", { d: paths[name] || "" }));
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = h("textarea", { style: "position:fixed;opacity:0" });
    area.value = text;
    document.body.append(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    area.remove();
    return ok;
  }
}
