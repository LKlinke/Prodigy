// The program editor: CodeMirror 6 with pGCL highlighting, diagnostics from the server and
// inline annotations (the strategy above each loop, the result after each query).
import {
  EditorState, StateField, StateEffect, Compartment, EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, drawSelection, Decoration, WidgetType, defaultKeymap, history,
  historyKeymap, indentWithTab, syntaxHighlighting, HighlightStyle, bracketMatching, indentOnInput,
  indentUnit, tags, lintGutter, setDiagnostics, lintKeymap, searchKeymap, highlightSelectionMatches,
  autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap, pgcl, PGCL_WORDS, placeholder,
} from "../vendor/codemirror.js";

// The editor's look, in the app's colour tokens. As a CodeMirror theme it takes precedence over
// CodeMirror's base theme (plain CSS rules in app.css would lose to its scoped selectors).
const appTheme = EditorView.theme({
  "&": { height: "100%", fontSize: "13px", color: "var(--text)", backgroundColor: "transparent" },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "1.6" },
  ".cm-content": { padding: "12px 0 48px", caretColor: "var(--accent)" },
  ".cm-gutters": { backgroundColor: "var(--surface)", color: "var(--text-3)", border: "none" },
  ".cm-lineNumbers .cm-gutterElement": { padding: "0 8px 0 12px", minWidth: "32px" },
  ".cm-activeLine": { backgroundColor: "var(--active-line)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--text)" },
  ".cm-selectionBackground, &.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground":
    { backgroundColor: "var(--selection)" },
  ".cm-cursor, .cm-dropCursor": { borderLeft: "2px solid var(--accent)" },
  ".cm-matchingBracket, &.cm-focused .cm-matchingBracket": { backgroundColor: "var(--fill-hover)", outline: "none" },
  ".cm-selectionMatch": { backgroundColor: "var(--fill-hover)" },
  ".cm-searchMatch": { backgroundColor: "rgba(255, 204, 0, 0.35)", outline: "none" },
  ".cm-searchMatch.cm-searchMatch-selected": { backgroundColor: "rgba(255, 159, 10, 0.55)" },
  ".cm-tooltip": {
    backgroundColor: "var(--surface)", color: "var(--text)", border: "1px solid var(--separator-strong)",
    borderRadius: "8px", boxShadow: "var(--shadow)", overflow: "hidden",
  },
  ".cm-tooltip-autocomplete ul li[aria-selected]": { backgroundColor: "var(--accent-fill)", color: "var(--on-accent)" },
  ".cm-tooltip-lint": { padding: "0" },
  ".cm-diagnostic": { padding: "6px 10px", fontFamily: "var(--font-ui)", fontSize: "var(--text-callout)", borderLeftWidth: "3px" },
  ".cm-diagnostic-error": { borderLeftColor: "var(--error)" },
  ".cm-diagnostic-warning": { borderLeftColor: "var(--unknown)" },
  ".cm-diagnostic-info": { borderLeftColor: "var(--text-3)" },
  ".cm-lintRange-error": { backgroundImage: "none", textDecoration: "wavy underline var(--error)", textUnderlineOffset: "3px" },
  ".cm-lintRange-warning": { backgroundImage: "none", textDecoration: "wavy underline var(--unknown)", textUnderlineOffset: "3px" },
  ".cm-lintRange-info": { backgroundImage: "none" },
  ".cm-gutter-lint": { width: "14px" },
  ".cm-lint-marker": { width: "10px", height: "10px" },
  ".cm-panels": { backgroundColor: "var(--surface)", color: "var(--text)" },
  ".cm-panels.cm-panels-top": { borderBottom: "1px solid var(--separator)" },
  ".cm-panels.cm-panels-bottom": { borderTop: "1px solid var(--separator)" },
  ".cm-textfield": {
    backgroundColor: "var(--surface)", color: "var(--text)", border: "1px solid var(--separator-strong)",
    borderRadius: "var(--radius-control)",
  },
  ".cm-button": { backgroundImage: "none", backgroundColor: "var(--fill)", color: "var(--text)", border: "none", borderRadius: "var(--radius-control)" },
  ".cm-placeholder": { color: "var(--text-2)", fontFamily: "var(--font-ui)" },
});

// CodeMirror picks its own defaults for anything the theme does not set from this flag; it follows
// the system appearance like the rest of the app.
const darkScheme = window.matchMedia?.("(prefers-color-scheme: dark)");

const highlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: "var(--text)", fontWeight: "600" },
  { tag: tags.typeName, color: "var(--code-type)" },
  { tag: [tags.number, tags.atom, tags.bool], color: "var(--code-number)" },
  { tag: tags.standard(tags.variableName), color: "var(--code-builtin)" },
  { tag: tags.meta, color: "var(--text)", fontWeight: "700" },
  { tag: tags.comment, color: "var(--text-2)", fontStyle: "italic" },
  { tag: [tags.operator, tags.bracket, tags.punctuation], color: "var(--text-2)" },
  { tag: tags.definition(tags.variableName), color: "var(--text)", fontWeight: "500" },
]);

// ---- annotations -----------------------------------------------------------------------

const setAnnotations = StateEffect.define();

class LoopWidget extends WidgetType {
  constructor(spec) { super(); this.spec = spec; }
  eq(other) { return JSON.stringify(other.spec) === JSON.stringify(this.spec); }
  toDOM() {
    const wrap = document.createElement("div");
    wrap.className = "cm-loop-annotation";
    const button = document.createElement("button");
    button.type = "button";
    button.title = "Choose how this loop is analysed";
    button.textContent = `Loop ${this.spec.id} · ${this.spec.strategy}`;
    if (this.spec.outcome) {
      const outcome = document.createElement("span");
      outcome.className = "outcome";
      outcome.dataset.kind = this.spec.kind;
      outcome.textContent = this.spec.outcome;
      button.append(" · ", outcome);
    }
    button.addEventListener("mousedown", (event) => event.preventDefault());
    button.addEventListener("click", () => this.spec.onClick?.(this.spec.id));
    wrap.append(button);
    return wrap;
  }
  ignoreEvent() { return true; }
}

class QueryWidget extends WidgetType {
  constructor(spec) { super(); this.spec = spec; }
  eq(other) { return JSON.stringify(other.spec) === JSON.stringify(this.spec); }
  toDOM() {
    const span = document.createElement("span");
    span.className = `cm-query-annotation${this.spec.stale ? " cm-annotation-stale" : ""}`;
    span.dataset.kind = this.spec.kind;
    const glyph = document.createElement("span");
    glyph.className = "glyph";
    glyph.textContent = this.spec.glyph;
    glyph.setAttribute("aria-hidden", "true");
    span.append(glyph, this.spec.text);
    span.title = this.spec.title || "";
    return span;
  }
  ignoreEvent() { return true; }
}

const annotationField = StateField.define({
  create: () => Decoration.none,
  update(decorations, transaction) {
    let result = decorations.map(transaction.changes);
    for (const effect of transaction.effects) {
      if (effect.is(setAnnotations)) result = effect.value;
    }
    return result;
  },
  provide: (field) => EditorView.decorations.from(field),
});

function buildAnnotations(state, { loops = [], queries = [] }) {
  const ranges = [];
  for (const loop of loops) {
    if (!loop.line || loop.line > state.doc.lines) continue;
    const line = state.doc.line(loop.line);
    ranges.push(Decoration.widget({ widget: new LoopWidget(loop), block: true, side: -1 }).range(line.from));
  }
  for (const query of queries) {
    if (!query.line || query.line > state.doc.lines) continue;
    const line = state.doc.line(query.line);
    ranges.push(Decoration.widget({ widget: new QueryWidget(query), side: 1 }).range(line.to));
  }
  return Decoration.set(ranges, true);
}

// ---- completion ------------------------------------------------------------------------

function completions(context) {
  const word = context.matchBefore(/[\w?!]+/);
  if (!word || (word.from === word.to && !context.explicit)) return null;
  const declared = new Set();
  const text = context.state.doc.toString();
  for (const match of text.matchAll(/\b(?:nat|bool|real|rparam|nparam|const|dist)\s+([A-Za-z_]\w*)/g)) declared.add(match[1]);
  const options = [
    ...PGCL_WORDS.keywords.map((label) => ({ label, type: "keyword" })),
    ...PGCL_WORDS.types.map((label) => ({ label, type: "type" })),
    ...PGCL_WORDS.builtins.map((label) => ({ label, type: "function", apply: `${label}(` })),
    ...["?Pr[", "?Ex[", "!Print", "!Plot["].map((label) => ({ label, type: "keyword" })),
    ...[...declared].map((label) => ({ label, type: "variable" })),
  ];
  return { from: word.from, options, validFor: /^[\w?!]*$/ };
}

// ---- public API ---------------------------------------------------------------------------

export function createEditor(parent, { doc = "", onChange, onRun, onStop } = {}) {
  const shortcuts = keymap.of([
    { key: "Mod-Enter", run: () => { onRun?.(); return true; } },
    { key: "Mod-.", run: () => { onStop?.(); return true; } },
  ]);
  const appearance = new Compartment();
  const extensions = [
    shortcuts,
    appTheme,
    appearance.of(EditorView.darkTheme.of(Boolean(darkScheme?.matches))),
    lineNumbers(),
    highlightActiveLineGutter(),
    history(),
    drawSelection(),
    indentOnInput(),
    bracketMatching(),
    closeBrackets(),
    autocompletion({ override: [completions], activateOnTyping: true }),
    highlightActiveLine(),
    highlightSelectionMatches(),
    indentUnit.of("    "),
    keymap.of([...closeBracketsKeymap, ...defaultKeymap, ...searchKeymap, ...historyKeymap,
      ...completionKeymap, ...lintKeymap, indentWithTab]),
    pgcl,
    syntaxHighlighting(highlightStyle),
    lintGutter(),
    annotationField,
    placeholder("Write a pGCL program, or choose Open… or Examples."),
    EditorView.contentAttributes.of({ "aria-label": "pGCL program", spellcheck: "false", autocorrect: "off", autocapitalize: "off" }),
    EditorView.updateListener.of((update) => {
      if (update.docChanged) onChange?.(update.state.doc.toString());
    }),
  ];
  const view = new EditorView({ state: EditorState.create({ doc, extensions }), parent });
  darkScheme?.addEventListener?.("change", (event) => {
    view.dispatch({ effects: appearance.reconfigure(EditorView.darkTheme.of(event.matches)) });
  });

  return {
    view,
    getSource: () => view.state.doc.toString(),
    /** Replaces the program as one undoable change. */
    setSource(text) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text }, selection: { anchor: 0 },
        scrollIntoView: true });
    },
    /** Starts over with this program: a fresh state without undo history. */
    load(text) {
      view.setState(EditorState.create({ doc: text, extensions }));
      view.dispatch({ effects: appearance.reconfigure(EditorView.darkTheme.of(Boolean(darkScheme?.matches))) });
      onChange?.(text);
    },
    setDiagnostics(diagnostics) {
      const doc = view.state.doc;
      const list = [];
      for (const d of diagnostics) {
        const lineNumber = Math.min(Math.max(1, d.line || 1), doc.lines);
        const line = doc.line(lineNumber);
        const from = Math.min(line.to, line.from + Math.max(0, (d.column || 1) - 1));
        let to = from;
        if (d.end_line && d.end_column) {
          const endLine = doc.line(Math.min(Math.max(1, d.end_line), doc.lines));
          to = Math.min(endLine.to, endLine.from + Math.max(0, d.end_column - 1));
        }
        if (to <= from) to = Math.min(line.to, from + 1);
        list.push({ from, to: Math.max(to, from), severity: d.severity, message: d.hint ? `${d.message}\n${d.hint}` : d.message });
      }
      view.dispatch(setDiagnostics(view.state, list));
    },
    setAnnotations(spec) {
      view.dispatch({ effects: setAnnotations.of(buildAnnotations(view.state, spec)) });
    },
    revealLine(lineNumber) {
      const doc = view.state.doc;
      const line = doc.line(Math.min(Math.max(1, lineNumber), doc.lines));
      view.dispatch({ selection: { anchor: line.from }, effects: EditorView.scrollIntoView(line.from, { y: "center" }) });
      view.focus();
    },
    focus: () => view.focus(),
  };
}
