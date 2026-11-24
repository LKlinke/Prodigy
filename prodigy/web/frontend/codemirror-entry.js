// Entry point of the vendored CodeMirror bundle (static/vendor/codemirror.js).
// Re-exports the pieces the GUI uses, plus the language definition of .ppy.
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

const BUILTINS = new Set(["unif", "unif_d", "unif_c", "geometric", "poisson", "logdist", "binomial",
  "negative_binomial", "bernoulli", "iid", "sample_pgf"]);
const ATOMS = new Set(["true", "false"]);

const PPY_KEYWORDS = new Set(["while", "if", "else", "skip", "abort", "observe", "loop", "query",
  "return", "infer", "sample", "def", "switch", "case", "default", "const", "param", "print", "plot", "optimize",
  "and", "or", "not", "categorical"]);
const PPY_TYPES = new Set(["nat", "bool", "real", "dist"]);

export const PPY_WORDS = { keywords: [...PPY_KEYWORDS], types: [...PPY_TYPES], builtins: [...BUILTINS] };

/** The Python-like .ppy syntax (prodigy/pgcl/ppy_parser.py): comments are '#' to the end of the
 * line and '/* … *\/' (not nested, may span lines), since '//' is floor division; declarations are
 * 'x, y: nat'; queries are 'query Pr[…]'. */
export const ppy = StreamLanguage.define({
  name: "ppy",
  startState: () => ({ statementStart: true, declaring: false, afterDef: false, afterQuery: false, inComment: false }),
  token(stream, state) {
    if (state.inComment || stream.match("/*")) {
      // a block comment, possibly continued from an earlier line
      state.inComment = true;
      while (!stream.eol()) {
        if (stream.match("*/")) { state.inComment = false; break; }
        stream.next();
      }
      return "comment";
    }
    if (stream.eatSpace()) return null;
    if (stream.peek() === "#") {
      stream.skipToEnd();
      return "comment";
    }
    const start = state.statementStart;
    state.statementStart = false;
    if (state.afterQuery && stream.match(/^(?:Pr|Ex|Var|Cov)\b/)) { state.afterQuery = false; return "meta"; }
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
    commentTokens: { line: "#", block: { open: "/*", close: "*/" } },
    closeBrackets: { brackets: ["(", "[", "{"] },
  },
});
