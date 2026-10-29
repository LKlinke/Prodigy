// Entry point of the vendored CodeMirror bundle (static/vendor/codemirror.js).
// Re-exports the pieces the GUI uses, plus the pGCL language definition.
export { EditorState, StateField, StateEffect, Compartment, RangeSetBuilder, Prec } from "@codemirror/state";
export {
  EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection,
  dropCursor, rectangularSelection, crosshairCursor, Decoration, WidgetType, ViewPlugin, placeholder,
} from "@codemirror/view";
export { defaultKeymap, history, historyKeymap, indentWithTab, undo, redo } from "@codemirror/commands";
export {
  StreamLanguage, syntaxHighlighting, HighlightStyle, bracketMatching, indentOnInput, foldGutter,
  foldKeymap, indentUnit,
} from "@codemirror/language";
export { tags } from "@lezer/highlight";
export { linter, lintGutter, setDiagnostics, lintKeymap, forceLinting } from "@codemirror/lint";
export { searchKeymap, highlightSelectionMatches } from "@codemirror/search";
export { autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap, completeFromList } from "@codemirror/autocomplete";

import { StreamLanguage } from "@codemirror/language";

const KEYWORDS = new Set(["while", "if", "else", "skip", "abort", "observe", "loop", "query", "tick",
  "return", "infer", "sample", "fun", "not"]);
const TYPES = new Set(["nat", "bool", "real", "const", "rparam", "nparam", "dist"]);
const BUILTINS = new Set(["unif", "unif_d", "unif_c", "geometric", "poisson", "logdist", "binomial",
  "bernoulli", "iid", "sample_pgf"]);
const ATOMS = new Set(["true", "false"]);

export const PGCL_WORDS = { keywords: [...KEYWORDS], types: [...TYPES], builtins: [...BUILTINS] };

/** pGCL as accepted by the parser in prodigy/pgcl (comments: # and //, queries ?Pr ?Ex ?Opt !Print !Plot). */
export const pgcl = StreamLanguage.define({
  name: "pgcl",
  startState: () => ({ afterType: false }),
  token(stream, state) {
    if (stream.eatSpace()) return null;
    if (stream.match("//") || stream.peek() === "#") {
      stream.skipToEnd();
      return "comment";
    }
    if (stream.match(/^\?(?:Pr|Ex|Opt)\b/) || stream.match(/^!(?:Print|Plot)\b/)) return "meta";
    if (stream.match(/^\\infty/) || stream.match("∞")) return "atom";
    if (stream.match(/^\d+\.\d*(?:[eE][+-]?\d+)?/) || stream.match(/^\.\d+/) || stream.match(/^\d+/)) {
      return "number";
    }
    if (stream.match(/^[A-Za-z_]\w*/)) {
      const word = stream.current();
      if (KEYWORDS.has(word)) { state.afterType = false; return "keyword"; }
      if (TYPES.has(word)) { state.afterType = true; return "type"; }
      if (ATOMS.has(word)) return "atom";
      if (BUILTINS.has(word)) return "builtin";
      if (state.afterType) { state.afterType = false; return "def"; }
      return "variable";
    }
    if (stream.match(":=")) { state.afterType = false; return "operator"; }
    if (stream.match(/^(?:<=|>=|\|\||[<>=&+\-*\/%^:])/)) return "operator";
    if (stream.match(/^[{}()\[\]]/)) return "bracket";
    if (stream.match(/^[;,]/)) { state.afterType = false; return "punctuation"; }
    stream.next();
    return null;
  },
  languageData: {
    commentTokens: { line: "//" },
    closeBrackets: { brackets: ["(", "[", "{"] },
  },
});
