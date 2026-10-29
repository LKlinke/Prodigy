// Per-browser conveniences (last program, setup, pane width). Storage can be unavailable
// (private windows, blocked site data): every access is guarded and has a fallback.
const PREFIX = "prodigy.gui.";

export function load(key, fallback) {
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function save(key, value) {
  try {
    window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // ignore: storage is a convenience only
  }
}
