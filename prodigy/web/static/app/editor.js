// The program editor: CodeMirror 6 with pGCL and .ppy highlighting, diagnostics from the server
// and inline annotations (the strategy above each loop, the result after each query).
//
// The syntax ("pgcl" or "ppy") is part of the editor state, so that undoing a conversion to .ppy
// also brings the old syntax back (invertedEffects). A change can carry a companion change of state
// outside the editor ({before, after}, e.g. the setup's invariant programs), which undo and redo
// apply together with the text.
import {
  EditorState, StateField, StateEffect, Compartment, EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, drawSelection, Decoration, WidgetType, defaultKeymap, history,
  historyKeymap, indentWithTab, syntaxHighlighting, HighlightStyle, bracketMatching, indentOnInput,
  indentUnit, tags, lintGutter, setDiagnostics, lintKeymap, searchKeymap, highlightSelectionMatches,
  autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap, pgcl, PGCL_WORDS, ppy, PPY_WORDS,
  placeholder, invertedEffects,
} from "../vendor/codemirror.js";

export const SYNTAXES = {
  pgcl: { label: "pGCL", extension: ".pgcl", language: pgcl, words: PGCL_WORDS,
    queries: ["?Pr[", "?Ex[", "?Var[", "?Cov[", "!Print", "!Plot["],
    declaration: /\b(?:nat|bool|real|rparam|nparam|const|dist)\s+([A-Za-z_]\w*)/g },
  ppy: { label: ".ppy", extension: ".ppy", language: ppy, words: PPY_WORDS,
    queries: ["query Pr[", "query Ex[", "query Var[", "query Cov["],
    declaration: /(?:^|[;{]\s*|\b(?:const|param|def)\s+)([A-Za-z_]\w*)(?=\s*[:=(])/gm },
};

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
  // room left of the code for the breakpoint (●) and current-step (▶) markers
  ".cm-line": { position: "relative", paddingLeft: "20px" },
  ".cm-step-line": { backgroundColor: "var(--step-line)" },
  ".cm-step-range": { outline: "1px solid var(--step-outline)", borderRadius: "2px" },
  ".cm-step-first::before": {
    content: '"▶"', position: "absolute", left: "10px", top: "0", color: "var(--accent)", fontSize: "9px",
    lineHeight: "inherit",
  },
  ".cm-breakpoint::after": {
    content: '"●"', position: "absolute", left: "1px", top: "0", color: "var(--breakpoint)", fontSize: "10px",
    lineHeight: "inherit",
  },
  ".cm-lineNumbers .cm-gutterElement:hover": { color: "var(--text)", cursor: "pointer" },
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

// ---- analysis timeline: current step and breakpoints ----------------------------------------------

const setStep = StateEffect.define();

/** The instruction of the step at the analysis card's timeline position: its lines get a background, its text a frame. */
const stepField = StateField.define({
  create: () => Decoration.none,
  update(decorations, transaction) {
    let result = decorations.map(transaction.changes);
    for (const effect of transaction.effects) {
      if (effect.is(setStep)) result = effect.value;
    }
    return result;
  },
  provide: (field) => EditorView.decorations.from(field),
});

function buildStep(state, step) {
  if (!step?.line || step.line > state.doc.lines) return Decoration.none;
  const doc = state.doc;
  const first = doc.line(step.line);
  const endLine = doc.line(Math.min(Math.max(step.end_line || step.line, step.line), doc.lines));
  const from = Math.min(first.to, first.from + Math.max(0, (step.column || 1) - 1));
  let to = step.end_column ? Math.min(endLine.to, endLine.from + Math.max(0, step.end_column - 1)) : endLine.to;
  if (to <= from) to = Math.min(first.to, from + 1);
  const ranges = [Decoration.line({ class: "cm-step-line cm-step-first", attributes: { "data-step": step.label || "" } }).range(first.from)];
  for (let n = step.line + 1; n <= endLine.number; n += 1) {
    ranges.push(Decoration.line({ class: "cm-step-line" }).range(doc.line(n).from));
  }
  if (to > from) ranges.push(Decoration.mark({ class: "cm-step-range" }).range(from, to));
  return Decoration.set(ranges, true);
}

const toggleBreakpointEffect = StateEffect.define();
const setBreakpointsEffect = StateEffect.define();

/** Lines with a breakpoint (as line decorations, so they move with edits). */
const breakpointField = StateField.define({
  create: () => Decoration.none,
  update(decorations, transaction) {
    let result = decorations.map(transaction.changes);
    for (const effect of transaction.effects) {
      if (effect.is(setBreakpointsEffect)) result = effect.value;
      if (effect.is(toggleBreakpointEffect)) {
        const position = effect.value;
        let found = false;
        result.between(position, position, () => { found = true; });
        result = found
          ? result.update({ filter: (from) => from !== position })
          : result.update({ add: [breakpointMark.range(position)], sort: true });
      }
    }
    return result;
  },
  provide: (field) => EditorView.decorations.from(field),
});

const breakpointMark = Decoration.line({ class: "cm-breakpoint", attributes: { "aria-label": "Breakpoint" } });

function breakpointLines(state) {
  const lines = [];
  const iter = state.field(breakpointField).iter();
  while (iter.value) {
    lines.push(state.doc.lineAt(iter.from).number);
    iter.next();
  }
  return [...new Set(lines)].sort((a, b) => a - b);
}

// ---- completion ------------------------------------------------------------------------

// ---- syntax ---------------------------------------------------------------------------

const setSyntax = StateEffect.define();

const syntaxField = StateField.define({
  create: () => "pgcl",
  update(value, transaction) {
    for (const effect of transaction.effects) {
      if (effect.is(setSyntax)) value = effect.value;
    }
    return value;
  },
});

// undo and redo of a change that switched the syntax switch it back
const invertSyntax = invertedEffects.of((transaction) => {
  const before = transaction.startState.field(syntaxField);
  return transaction.effects.some((e) => e.is(setSyntax)) ? [setSyntax.of(before)] : [];
});

// a companion change {before, after}: applied with its transaction, swapped on undo and redo
const companion = StateEffect.define();

const invertCompanion = invertedEffects.of((transaction) => transaction.effects
  .filter((e) => e.is(companion))
  .map((e) => companion.of({ before: e.value.after, after: e.value.before })));

function completions(context) {
  const word = context.matchBefore(/[\w?!]+/);
  if (!word || (word.from === word.to && !context.explicit)) return null;
  const syntax = SYNTAXES[context.state.field(syntaxField)];
  const declared = new Set();
  const text = context.state.doc.toString();
  const keywords = new Set(syntax.words.keywords.concat(syntax.words.types));
  for (const match of text.matchAll(syntax.declaration)) if (!keywords.has(match[1])) declared.add(match[1]);
  const options = [
    ...syntax.words.keywords.map((label) => ({ label, type: "keyword" })),
    ...syntax.words.types.map((label) => ({ label, type: "type" })),
    ...syntax.words.builtins.map((label) => ({ label, type: "function", apply: `${label}(` })),
    ...syntax.queries.map((label) => ({ label, type: "keyword" })),
    ...[...declared].map((label) => ({ label, type: "variable" })),
  ];
  return { from: word.from, options, validFor: /^[\w?!]*$/ };
}

// ---- public API ---------------------------------------------------------------------------

function languageFor(syntax) {
  const spec = SYNTAXES[syntax];
  return [
    spec.language,
    placeholder(`Write a ${spec.label} program, or choose Open… or Examples.`),
    EditorView.contentAttributes.of({ "aria-label": `${spec.label} program` }),
  ];
}

/** onSyntaxChange(syntax) is called whenever the syntax changes, also by undo and redo;
 * onCompanion({before, after}) whenever a change made with setSource(…, {companion}) is applied,
 * undone (before and after swapped) or redone. */
export function createEditor(parent, { doc = "", syntax = "pgcl", onChange, onSyntaxChange, onCompanion, onRun, onStop,
  onBreakpointsChange, stepKeys = {} } = {}) {
  const toggleAt = (view, position) => {
    view.dispatch({ effects: toggleBreakpointEffect.of(view.state.doc.lineAt(position).from) });
    return true;
  };
  const shortcuts = keymap.of([
    { key: "Mod-Enter", run: () => { onRun?.(); return true; } },
    { key: "Mod-.", run: () => { onStop?.(); return true; } },
    { key: "F9", run: (view) => toggleAt(view, view.state.selection.main.head) },
    ...Object.entries(stepKeys).map(([key, run]) => ({ key, run: () => run() !== false })),
  ]);
  const appearance = new Compartment();
  const language = new Compartment();
  const extensions = [
    shortcuts,
    appTheme,
    appearance.of(EditorView.darkTheme.of(Boolean(darkScheme?.matches))),
    lineNumbers({
      domEventHandlers: {
        // a click on a line number sets or clears a breakpoint on that line (F9 at the cursor does the same)
        mousedown: (view, line) => toggleAt(view, line.from),
      },
    }),
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
    invertSyntax,
    invertCompanion,
    syntaxHighlighting(highlightStyle),
    lintGutter(),
    annotationField,
    stepField,
    breakpointField,
    EditorView.contentAttributes.of({ spellcheck: "false", autocorrect: "off", autocapitalize: "off" }),
    EditorView.updateListener.of((update) => {
      // before onSyntaxChange, so that what it renders already shows the companion state
      for (const transaction of update.transactions) {
        for (const effect of transaction.effects) if (effect.is(companion)) onCompanion?.(effect.value);
      }
      const before = update.startState.field(syntaxField);
      const after = update.state.field(syntaxField);
      if (before !== after) {
        // the language follows the syntax (also on undo); reconfiguring needs its own transaction
        queueMicrotask(() => view.dispatch({ effects: language.reconfigure(languageFor(view.state.field(syntaxField))) }));
        onSyntaxChange?.(after);
      }
      if (update.docChanged) onChange?.(update.state.doc.toString());
      const marksBefore = update.startState.field(breakpointField);
      if (marksBefore !== update.state.field(breakpointField) || (update.docChanged && marksBefore.size)) {
        onBreakpointsChange?.(breakpointLines(update.state));
      }
    }),
  ];
  const createState = (text, initialSyntax) => EditorState.create({
    doc: text,
    extensions: [extensions, syntaxField.init(() => initialSyntax), language.of(languageFor(initialSyntax))],
  });
  const view = new EditorView({ state: createState(doc, syntax), parent });
  darkScheme?.addEventListener?.("change", (event) => {
    view.dispatch({ effects: appearance.reconfigure(EditorView.darkTheme.of(event.matches)) });
  });

  return {
    view,
    getSource: () => view.state.doc.toString(),
    getSyntax: () => view.state.field(syntaxField),
    /** Replaces the program (and its syntax, if given) as one undoable change; a companion
     * {before, after} is undone and redone with it (see onCompanion). */
    setSource(text, syntax = null, { companion: change = null } = {}) {
      const effects = syntax && syntax !== view.state.field(syntaxField) ? [setSyntax.of(syntax)] : [];
      if (change) effects.push(companion.of(change));
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text }, selection: { anchor: 0 },
        effects, scrollIntoView: true });
    },
    /** Starts over with this program: a fresh state without undo history. */
    load(text, syntax = view.state.field(syntaxField)) {
      const before = view.state.field(syntaxField);
      view.setState(createState(text, syntax));
      view.dispatch({ effects: appearance.reconfigure(EditorView.darkTheme.of(Boolean(darkScheme?.matches))) });
      if (before !== syntax) onSyntaxChange?.(syntax);
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
    /** Highlights the instruction of a step ({line, column, end_line, end_column, label}); null clears it. */
    showStep(step, { scroll = true } = {}) {
      const effects = [setStep.of(buildStep(view.state, step))];
      if (step?.line && scroll && step.line <= view.state.doc.lines) {
        effects.push(EditorView.scrollIntoView(view.state.doc.line(step.line).from, { y: "nearest", yMargin: 48 }));
      }
      view.dispatch({ effects });
    },
    getBreakpoints: () => breakpointLines(view.state),
    toggleBreakpoint(lineNumber) {
      if (lineNumber >= 1 && lineNumber <= view.state.doc.lines) toggleAt(view, view.state.doc.line(lineNumber).from);
    },
    setBreakpoints(lines) {
      const doc = view.state.doc;
      const marks = [...new Set(lines)].filter((n) => n >= 1 && n <= doc.lines).sort((a, b) => a - b)
        .map((n) => breakpointMark.range(doc.line(n).from));
      view.dispatch({ effects: setBreakpointsEffect.of(Decoration.set(marks, true)) });
    },
    focus: () => view.focus(),
  };
}
