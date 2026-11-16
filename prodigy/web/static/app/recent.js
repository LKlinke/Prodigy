// Recently opened programs: library examples and local files, newest first, kept in localStorage
// (a convenience only: without storage the list stays empty and everything else works).
// A file is remembered with its text (browsers do not let a page reopen a file by name), so a
// large program is not kept.
import { load, save } from "./store.js";

const KEY = "recent";
export const MAX_RECENT = 8;
const MAX_TEXT = 60000;

const valid = (entry) => entry && typeof entry === "object" && typeof entry.title === "string" &&
  (entry.kind === "example" ? typeof entry.id === "string" : entry.kind === "file" && typeof entry.text === "string");

export function recentEntries() {
  const entries = load(KEY, []);
  return Array.isArray(entries) ? entries.filter(valid).slice(0, MAX_RECENT) : [];
}

const same = (a, b) => a.kind === b.kind && (a.kind === "example" ? a.id === b.id : a.file === b.file);

/** Puts an entry first: {kind: "example", id, title, category} or {kind: "file", title, file, text}. */
export function addRecent(entry) {
  if (entry.kind === "file" && entry.text.length > MAX_TEXT) return;
  const rest = recentEntries().filter((e) => !same(e, entry));
  save(KEY, [entry, ...rest].slice(0, MAX_RECENT));
}

export function clearRecent() {
  save(KEY, []);
}
