// Entry point of the vendored CodeMirror bundle (static/vendor/codemirror.js).
// Re-exports the pieces the GUI uses, plus the language definitions of pGCL and .ppy.
export { EditorState, StateField, StateEffect, Compartment, RangeSetBuilder, Prec } from "@codemirror/state";
export {
  EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection,
  dropCursor, rectangularSelection, crosshairCursor, Decoration, WidgetType, ViewPlugin, placeholder,
} from "@codemirror/view";
export { defaultKeymap, history, historyKeymap, indentWithTab, undo, redo, invertedEffects } from "@codemirror/commands";
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

const PPY_KEYWORDS = new Set(["while", "if", "else", "skip", "abort", "observe", "loop", "query", "tick",
  "return", "infer", "sample", "def", "switch", "case", "default", "const", "param", "print", "plot", "optimize",
  "and", "or", "not", "categorical"]);
const PPY_TYPES = new Set(["nat", "bool", "real", "dist"]);

export const PPY_WORDS = { keywords: [...PPY_KEYWORDS], types: [...PPY_TYPES], builtins: [...BUILTINS] };

/** The Python-like .ppy syntax (prodigy/pgcl/ppy_parser.py): comments start with '#' only, since
 * '//' is floor division; declarations are 'x, y: nat'; queries are 'query Pr[…]'. */
export const ppy = StreamLanguage.define({
  name: "ppy",
  startState: () => ({ statementStart: true, declaring: false, afterDef: false, afterQuery: false }),
  token(stream, state) {
    if (stream.eatSpace()) return null;
    if (stream.peek() === "#") {
      stream.skipToEnd();
      return "comment";
    }
    const start = state.statementStart;
    state.statementStart = false;
    if (state.afterQuery && stream.match(/^(?:Pr|Ex)\b/)) { state.afterQuery = false; return "meta"; }
    state.afterQuery = false;
    if (stream.match(/^\\infty/) || stream.match("∞")) return "atom";
    if (stream.match(/^\d+\.\d*(?:[eE][+-]?\d+)?/) || stream.match(/^\.\d+/) || stream.match(/^\d+/)) {
      return "number";
    }
    if (start && stream.match(/^[A-Za-z_]\w*(?=\s*(?:,\s*[A-Za-z_]\w*\s*)*:)/)) {
      // the names of a declaration 'x, y: nat'
      state.declaring = true;
      return "def";
    }
    if (stream.match(/^[A-Za-z_]\w*/)) {
      const word = stream.current();
      if (state.declaring && !PPY_TYPES.has(word)) return "def";
      if (state.afterDef) { state.afterDef = false; return "def"; }
      if (word === "query") { state.afterQuery = true; return "meta"; }
      if (["print", "plot", "optimize"].includes(word)) return "meta";
      if (PPY_KEYWORDS.has(word)) {
        if (word === "def" || word === "const" || word === "param") state.afterDef = true;
        return "keyword";
      }
      if (PPY_TYPES.has(word)) { state.declaring = false; return "type"; }
      if (ATOMS.has(word)) return "atom";
      if (BUILTINS.has(word)) return "builtin";
      return "variable";
    }
    if (stream.match(/^(?:\*\*|\/\/|==|<=|>=|->|&&|\|\||[<>=!+\-*\/%])/)) { state.declaring = false; return "operator"; }
    if (stream.match(":")) return "operator";
    if (stream.match(/^[{}()\[\]]/)) {
      const bracket = stream.current();
      if (bracket === "{" || bracket === "}") state.statementStart = true;
      state.declaring = false;
      return "bracket";
    }
    if (stream.match(";")) { state.statementStart = true; state.declaring = false; return "punctuation"; }
    if (stream.match(",")) return "punctuation";
    stream.next();
    return null;
  },
  languageData: {
    commentTokens: { line: "#" },
    closeBrackets: { brackets: ["(", "[", "{"] },
  },
});
