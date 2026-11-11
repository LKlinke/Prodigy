# Prodigy web GUI and analysis API

`prodigy web` starts a local web server and opens the GUI in the browser. You write a pGCL
program, choose how each `while` loop is analysed, and get the output distribution and the
answers to the queries — each with what is *guaranteed* about it.

```sh
uv sync
uv run prodigy web                    # http://localhost:8080
uv run prodigy --engine ginac web     # GiNaC as default engine (usually much faster)
uv run prodigy web --port 9000 --time-limit 300 --no-browser
```

The GUI needs no Node.js or internet connection; the editor (CodeMirror 6) and KaTeX are
vendored under `prodigy/web/static/vendor/`.

Prodigy is tested with Python 3.11 and 3.14.

## What the GUI shows

* **Program** (left): editor with highlighting for both syntaxes, syntax errors (and
  type-checker warnings) as you type. Above each loop a pill shows its strategy and, after a
  run, what it certified. After each query (`query Pr[…]`, `?Pr[…]`) the value appears inline.
  The chip next to the title shows the syntax of the program; see [Two syntaxes](#two-syntaxes).
* **Setup** (right): initial distribution, one strategy per loop with its parameters,
  analysis options (engine, conditioning, simplification, number of coefficients, time
  limit, EVT synthesis strategy, solver, recording of intermediate distributions). The
  synthesis strategy *from heuristics* combines a template heuristic and a positivity
  heuristic chosen below it (on the command line: `--strategy from_heuristics
  --template-heuristic … --pos-heuristic …`).
* **Result** (right): three cards and a quiet line, in this order (#288):
  * **Answer** (emphasised by an edge in the colour of the guarantee): the guarantee once, as a
    badge with its glyph and word and one sentence (for a lower bound with the residual mass, for
    an upper bound with the side condition of Park's lemma), timing and engine on the right. Then
    the queries as large rows: a short exact value large with its decimal small (`5/16 ≈ 0.3125`),
    otherwise the decimal large with the exact form small; the source line links to the editor. A
    query names its own kind of bound only where it differs from the overall guarantee; the
    interval of a lower bound (`∈ [0.5, 0.5 + 3 × 10⁻⁸]`) and the hatched interval bar are values
    and stay. *Other expected values* lists E[v] for the variables that no `Ex[v]` query covers.
  * **Output distribution**: one figure with one control row: the variable (`x | y | … | x and
    y`; the pair switches the same figure to the joint view, see [Joint distribution of two
    variables](#joint-distribution-of-two-variables)), the view (*Chart | Table* for one variable,
    *Heatmap | 3D | Table* for a pair) and the scale *Linear | Log*. Under the figure one **Mass**
    line for whatever it shows: the total mass, the mass shown (the values of the chart, or the
    window), the mass beyond, and for a lower bound the hatched "?" mass not yet accounted for,
    each labelled, with a grey bar from 0 to 1. Support, parameters and the probability of a
    violated observation are listed above the controls. *Generating function* is a disclosure with
    the closed form and its copy buttons. A program with nested inference gets the block
    *Inferred distributions* in this card: per value of each `dist` variable its `infer(…)`, the
    normalised generating function, the expected value, the normalising constant Z and the
    coefficients; the `dist` variables are not part of the output distribution. In the **Steps**
    tab, every distribution lists the values of the `dist` variables next to it. A loop that
    assigns a `dist` variable is marked in the editor as not implemented yet.
  * **How it was computed**: one row per loop with its line, strategy, verdict in words (`✓
    exact`, `≥ lower bound, at most … not accounted for`, `≤ upper bound`, `! no guarantee`) and
    time; a failed loop keeps its badge (*Failed*, *Refuted*, *Not proven*) and its message. The
    certificates are a disclosure. *Step Through N Steps* opens the Steps tab when the steps were
    recorded.
  * **Details**: not a card but a disclosure line under the cards with the warnings, the console
    output and *Result as JSON*. If there are warnings, a notice above the cards says so and opens
    the line.

  The chart has a **Linear / Log** switch for the probability axis; the browser remembers the
  choice (the heatmap and the 3D view share it). On the log scale the axis is labelled in decades
  (`1`, `10⁻³`, …) and marked *log*, it spans at most 12 decades (smaller probabilities get the
  minimal bar), and a probability 0, which has no place on a log axis, is drawn as a dashed
  outline on the axis. The Steps tab does not animate log charts. The choice of variable and view
  is kept when the program is analysed again.
* **Steps** (right): a debugger for the analysis, see [Stepping through an analysis](#stepping-through-an-analysis).
* **Examples**: the programs in `pgfexamples/` (in `.ppy`), with a suggested setup derived from
  their first line (e.g. `check_equality` examples come with their invariant). A `.pgcl` example
  in a folder given with `prodigy web --examples` is converted to `.ppy` on the fly (with its
  invariants).
* **New, Open…, Save** (toolbar): in Chrome and Edge, Open and Save work directly on `.ppy` and
  `.pgcl` files on disk, and the file stays linked across reloads; Save As… (⇧⌘S) picks another file.
  Safari and Firefox cannot write to a folder: Open reads the chosen file, Save downloads the
  program. The program is also kept in the browser, together with the setup (the strategy of each
  loop), so a reload never loses either; replacing a program with unsaved changes (New, Open…, an example) asks first.

Keyboard: ⌘/Ctrl + Return analyses, ⌘/Ctrl + . (or Esc) stops, ⌘/Ctrl + O opens,
⌘/Ctrl + S saves; in the stepper F10 / ⇧F10 go to the next / previous step, F5 / ⇧F5 to the next /
previous breakpoint, and F9 in the editor sets a breakpoint. Every analysis runs in its own process with a time limit and can be stopped
at any time.

The GUI follows the system appearance (light or dark, macOS: System Settings → Appearance);
following Apple's guidelines it has no switch of its own. Firefox can override the appearance
for websites (Settings → General → Website appearance).

## Joint distribution of two variables

Choosing the pair (`x and y`, or *Two variables* with more than two) in the control row of the
card *Output distribution* shows Pr[x = i, y = j] for two chosen variables;
the other variables are marginalised. The coefficients are exact: the server computes them from
the output distribution of the finished analysis it keeps (`POST /api/jobs/<id>/joint`, a short
job of its own, see `prodigy/analysis/joint_grid.py`), the browser only draws them.

* **Pair**: the variables on the horizontal and the vertical axis (⇄ swaps them).
* **Window**: the values *from*–*to* of each axis, at most 50 per axis and up to 1000; *Show
  window* (or Return) computes them. A finite distribution starts with its support (the status
  line names it), an infinite one with the values 0–15.
* **Heatmap**: one cell per pair of values, coloured on the viridis scale (perceptually uniform,
  readable with colour-vision deficiencies) by probability; probability 0 is an empty grey cell,
  a value that depends on parameters is hatched. *Linear / Log* switches the colour scale (the same
  setting as the bar charts; the log scale spans at most 12 decades). Values never depend on colour
  alone: the line below the heatmap names the selected cell (`x = 1, y = 2: at least 0.0277778
  (1/36); at most 0.031684` for a lower bound), large cells print their value, and the table lists
  all of them. Keyboard: Tab focuses the heatmap, the arrow keys move between cells (↑ is the next
  larger y), Home/End go to the first/last column, Page Up/Page Down to the top/bottom row; the
  pointer selects the cell under it.
* **3D**: a bar per cell (plain canvas, no library), height and colour by probability. Pointing at
  a bar (or at the floor of a cell with probability 0) shows a tooltip with the values of both
  variables and the probability, formatted as in the heatmap's line; the bar gets an outline and the
  line below the view repeats the text. Keyboard: Tab focuses the view and shows the tooltip of the
  selected bar; the arrow keys, Home/End and Page Up/Page Down move between bars as in the heatmap
  (the heatmap and the 3D view share the selected cell). Drag or Shift + ←/→ rotate, Shift + ↑/↓
  tilt, pinch (⌘/Ctrl + wheel) or +/− zoom, 0 resets; the buttons above the view do the same.
* **Table**: the window as a table, rows y (largest on top) and columns x; hovering a value shows
  its exact form.
* **Colour key and mass**: two separate, labelled blocks below the heatmap and the 3D view.
  *Colour: probability of one cell* is the viridis key of a single cell's probability (linear from
  0, or logarithmic; the heading names the range). The card's *Mass* line, set apart by a rule,
  adds up the cells: the total mass, the mass in the window, the mass outside it, and for a lower
  bound the hatched "?" mass not yet accounted for, which may still end up in any cell, outside the
  window, or never terminate. For a lower bound every cell is a lower bound and gets *at most* =
  value + residual mass (the line below the view and the tooltip say *at least* and *at most*); for
  an upper bound every cell is an upper bound.

Nothing in the joint view is animated, so it needs no special case for reduced motion; it follows
light and dark mode (the 3D view redraws when the appearance changes).

## Stepping through an analysis

The **Steps** tab steps through the analysis instruction by instruction, to understand a result,
to find where an analysis gets slow or fails, or for teaching. It shows the steps of the last
analysis that recorded them: *Record steps and analyze* (or the option *Record each step* in
Setup) records them; the whole trace is recorded within the time limit of the analysis and
navigated afterwards, so stepping back is as cheap as stepping forward.

* **Navigation**: *Next* and *Back* (F10, ⇧F10, or ← → in the panel), first and last step,
  *Breakpoint ⟹* / *⟸ Breakpoint* to the next or previous step on a line with a breakpoint (F5,
  ⇧F5), *Jump to line…*, and the list of all steps (↑ ↓ in the list). A click on a line number in the
  editor, or F9 at the cursor, sets or clears a breakpoint (●).
* **Editor**: the instruction of the current step is highlighted and marked with ▶.
* **Steps**: every top-level instruction is a step, and so is every instruction inside the
  branches of `if`, `switch` and probabilistic choices; those show the *sub-distribution* of their
  branch (the part where the guard holds, or weighted by the probability of the branch), not
  normalised. Instructions inside loop bodies, function bodies and the checks of invariants are
  no steps. If observations were violated, a last step conditions on them.
* **Distribution before and after** a step, with the result's components: closed form, mass,
  violated observations, the size of the closed form, and the marginal charts with the hatched
  bar “?” for the mass not yet accounted for. Each distribution carries the guarantee that holds
  at that point, of the same kinds as the result: exact, lower bound with residual mass, upper
  bound, or none. A step whose instruction failed has no distribution after it.
* **Loops**: the strategy, its messages and the checks of its certificate in words (fixed point,
  non-negativity, finite mass, sub-invariance …, each *proven*, *failed*, *not established* or a
  note); a failed check says why, e.g. “Φ(ν) − ν has the negative coefficient −1/2 at x = 0, c = 0”.
  For unrolling strategies (*Unroll N iterations*, *Iterate until mass captured*, *Approximate
  expected visiting times*) **Kleene iterates** animates the iterations: the part that has left
  the loop grows towards the output of the loop, the hatched “?” bar and the bar below the chart
  show the mass still inside shrinking. *Play* (Space) runs through the iterates, the slider picks
  one, and the speed menu (0.5×, 1×, 2×) sets the pace. At most 40 iterates are kept per loop
  (`step_iterations`), spread evenly, plus the last. An iterate whose marginal was not recorded
  (see below) has no bars: they fade out at their last height, and a dashed frame says
  “x: not recorded” with the reason; the slider, the status line and the screen-reader label say
  the same. Not recorded means unknown, never probability 0.
* **Transitions**: two consecutive steps (or iterates) are keyframes, and the view moves between
  them with ease-in/ease-out in about a third of a second: bars and the “?” bar grow or shrink,
  bars that appear or disappear fade, the axis rescales, the closed form cross-fades inside its
  fixed box, and the mass, the size and the watched queries count to their new value. A counted
  number is only a transition: it is shown muted with “≈”, screen readers get the exact value,
  and the exact value replaces it at the end. A new step during a transition continues from what
  is on screen. With *Reduce motion* (macOS: System Settings → Accessibility → Display) only the
  end state is shown.
* **Queries at this step**: a query such as `Pr[x == 3]` or `Ex[x]` (in the program's syntax) is
  evaluated after every top-level step, as if it stood right after that instruction, with the same
  sound intervals as the program's queries; adding or removing one analyses again.
* **Diagnosis**: the step where the closed form grows most and the step where most of the time
  goes (time without nested steps).

Per distribution the trace keeps the closed form up to 3000 characters, its mass and the first 10
coefficients of the marginals of up to 6 variables; larger closed forms keep no marginals, and
after 20 seconds spent on marginals only closed forms are kept. Each distribution names the
variables without a marginal and why (`marginals_not_recorded`), and the Steps tab shows that
reason instead of a chart. The command line records the same
trace with `prodigy analyze --steps` (see the CLI documentation); the JSON keys are `steps` and
`trace` (see `prodigy/analysis/trace.py`).

## Two syntaxes

The GUI reads and writes both syntaxes: the Python-like `.ppy` and the original pGCL
(`.pgcl`). The file extension decides when a file is opened; **New** asks which one to use
(Return chooses `.ppy`, the syntax for new programs); examples open as `.ppy`. Highlighting, comments (`#` and `/* … */` in `.ppy`;
`#` and `//` in pGCL), completion, diagnostics and the loop pills follow the syntax.

**Convert to .ppy** (next to the syntax chip of a pGCL program) converts the program with the
same converter as `prodigy convert`. Names that are keywords in `.ppy` are renamed (`switch`
becomes `switch_`); the notice lists the renamings. Invariant programs in the setup are
converted as well. The conversion is one undoable change: ⌘Z restores the pGCL program, its
syntax and the setup's invariant programs (an invariant edited after the conversion stays as it is). A converted file is not written over the `.pgcl` file: Save asks for a new `.ppy` file.

Everything typed into the setup is read in the program's syntax: an invariant program of
*Check invariant* is a loop-free program in that syntax. Generating functions (the initial
distribution, EVT invariants and sub-invariants, candidates of *Suggest*) accept powers as `^`
and `**` in both syntaxes; a candidate of the language model is shown with the program's power
operator.

## Guarantees

Every loop strategy reports what is known about its result, and the session combines this
for the whole program:

| Kind | Meaning | Produced by |
| --- | --- | --- |
| exact | equal to the semantics | loop-free code; unrolling/iteration that reaches a fixed point or leaves no mass in the loop; an EVT fixed point with non-negative coefficients **and finite mass** |
| lower | coefficient-wise below the semantics; at most the *residual mass* r is unaccounted for | unroll N, iterate until mass captured, EVT iteration |
| upper | coefficient-wise above the semantics (Park's lemma) | invariant check (exact if the loop terminates almost surely), EVT invariants with non-negative coefficients whose equation holds for the series (see below) |
| unknown | no guaranteed direction | mixing lower and upper approximations, unchecked side conditions |

For a lower bound, every probability query becomes an interval
μ_N(φ) ≤ μ(φ) ≤ μ_N(φ) + r **without any termination assumption**: the unaccounted mass may
terminate with any value, violate an observation, or never terminate. When conditioning on
observations, the residual is rescaled together with the distribution (division by
1 − known violation mass), and the interval stays sound. Expectations of monotone,
non-negative expressions get the lower bound only.

EVT invariants (fixed points I = G + ⟦body⟧(⟨I⟩_guard)) are only upper bounds in general:
an invariant with infinite mass can be a *spurious* fixed point. The GUI checks
non-negativity with the configured positivity heuristic and decides finiteness of the mass
for rational closed forms (Pringsheim: no pole of I(t,…,t) in (0, 1]); only then is the
result reported as exact. Example: for `while (x > 0) { {x := x-1} [1/3] {x := x+1} }`
from x = 1, invariant synthesis finds a fixed point with infinite mass and Prodigy's value
Pr[x = 0] = 1 is reported as an upper bound — the true value is 1/2. Unrolling 25
iterations gives Pr[x = 0] ∈ [0.4956, 1].

Even the upper bound needs a side condition: the fixed-point equation is checked on the closed
form of I, Park's lemma needs it for the power series. They differ when the series diverges
where the body evaluates I, e.g. when it marginalises a variable (`x := 5` substitutes x = 1):
`1/(1 - 2x)` is −1 at x = 1, its series is ∞. So an EVT invariant with infinite (or undecided)
mass gives an upper bound only if its series converges on the open unit polydisc (no pole of
I(t,…,t) in (0, 1)) or the loop is *formal*: its guards compare variables with literals, its
probabilities are constants, and it only assigns `x := x + c`, `x := x - c` and `x := x + y`.
Otherwise the result is unknown (issue #276). The walk above is formal, so its upper bound
stands. The certificate names the reason as `series_identity`.

An invariant that fails its check gets a verdict. *Refuted* means it is provably wrong: a
counterexample, or Φ(I) − I is a nonzero rational function after exact cancellation.
*Not proven* means the check could neither prove nor refute it, e.g. for non-rational
closed forms such as `sin(c)^2 + cos(c)^2 - 1`, which SymPy simplifies to 0 but SymEngine and
GiNaC cannot, or for a template whose parameters the solver could not determine; such an
invariant may still be correct. The result JSON carries the verdict as `error.verdict`.

*Synthesise EVT invariant* (`evt_synthesis`) enumerates templates N/D with unknown
coefficients and solves the fixed-point equation for them. The search is bounded by two
parameters: *Maximal template degree* (`max_degree`, default 4: N has at most this degree in
each variable, D at most N's degrees in total) and *Time limit* (`time_limit`, default 60 s;
it also interrupts a solver call, since a single template can make SymPy's Gröbner basis run
for minutes); a third one, *Time limit per template* (`template_time_limit`, default 5 s), skips a
template whose equations are not solved in time, so that one expensive template does not use up the
whole time limit (skipped templates are counted in the message and in
`error.details.templates_skipped`). The denominator D has the constant term 1, which loses no
invariant: a generating function N/D in lowest terms has a denominator with a non-zero constant term.
Many loops have no rational invariant at all, e.g. those with algebraic or
non-D-finite generating functions in `pgfexamples/inference/loopy/hard_distributions`. If no
invariant is found within the bounds, the analysis ends with status `error`, verdict *unknown*
(`error.type` `SynthesisBoundReached`) and a message such as "No rational EVT invariant up to
degree 4 found (21 templates tried)"; `error.details` says which bound was reached
(`reason`: `degree` or `time_limit`). The synthesis strategy `inf_poly` enumerates
polynomials of every degree and is only stopped by the time limit. Example:
`prodigy analyze prog.ppy --loop '0=evt_synthesis:max_degree=2,time_limit=20'`. The command
`prodigy invariant_synthesis` has the options `--max-degree`, `--time-limit` and
`--template-time-limit`.

Two strategies give lower bounds without a termination assumption (package
`prodigy.analysis.lowerbounds`, see the README):

* **EVT sub-invariant** checks a generating function ν with finite mass and ν ≤ Φ(ν); then
  [¬guard]·ν lies below the output (lower induction), and at most |Φ(ν)| − |ν| mass is
  missing. If that is 0, the result is exact.
* **EVT sandwich** synthesises a finite-mass sub-invariant and a super-invariant (Park) by
  linear programming and checks both exactly. The ?Pr and ?Ex queries directly after the
  loop get both sides from the two certificates — expectations included, which otherwise
  only have a lower bound. For example, the random walk with a step counter in
  `pgfexamples/lower_bounds/sandwich/rw_counter_23.ppy` gives E[c] ∈ [2.99948, 3.00815].
  Queries after other instructions fall back to the residual mass. Needs a rational input
  distribution and no `observe`.

## Candidates from a language model

The strategy *EVT invariant from a language model* (`ai_invariant`) asks a language model for an
EVT invariant or a finite-mass EVT sub-invariant (parameter *Kind of candidate*). Prodigy checks
every candidate exactly, like one typed in by hand, so the guarantee never depends on the model.
A rejected candidate goes back to the model with the reason and the full details of the check
(e.g. Φ(I) − I, or the negative coefficient of Φ(ν) − ν and the state where it occurs). The
model is asked up to *Attempts* times in total.

Larger sub-invariants give tighter lower bounds. After a proven sub-invariant, Prodigy tells the
model the mass of its lower bound and the mass that is not accounted for, and asks for a larger
one while attempts are left. Each new candidate is checked exactly as well; Prodigy keeps the
proven sub-invariant whose lower bound has the most mass and stops early once one is exact. A
failed request after a proven sub-invariant keeps that sub-invariant.

The certificate names the model and the chosen candidate and lists the other attempts with
Prodigy's verdict in words — *Proven*, *Proven, not larger*, *Refuted*, *Not proven*, *No
candidate* — e.g. "Φ(ν) − ν has the negative coefficient -1/4 at x = 0, c = 1, so ν ≤ Φ(ν)
fails there". *Details of the check* shows what the model got. With the chosen candidate, the
check can be repeated without the model (strategies `evt_invariant`, `evt_subinvariant`).

The API key is read from the environment of `prodigy` and never reaches the browser:

```sh
export ANTHROPIC_API_KEY="sk-ant-..."         # Claude API
export PRODIGY_AI_MODEL=claude-opus-5-5       # optional, default claude-sonnet-5-5
uv run prodigy --engine ginac web

# or an OpenAI-compatible API, e.g. a local model served by Ollama:
export PRODIGY_AI_BASE_URL=http://localhost:11434/v1 PRODIGY_AI_MODEL=<model>
```

`ANTHROPIC_WORKSPACE_ID` is needed for a Claude API key that is not scoped to a workspace,
`OPENAI_API_KEY` for OpenAI's API; `PRODIGY_AI_PROVIDER` (`anthropic` or `openai`) chooses if
both are configured. Answers are untrusted input: a candidate may only contain numbers, the
program's variables and parameters, `+ - * / ^` (or `**`), parentheses, `sqrt` and `exp`, and
must parse as an expression of the program's syntax, before any computer algebra system sees it.

### The Suggest button

For teaching, a human in the loop is often better than the automatic strategy: the field *EVT
invariant I* of *Check EVT invariant* and the field *EVT sub-invariant nu* of *EVT
sub-invariant* have a **Suggest** button. It analyses the program up to the loop (earlier loops
with their strategies from Setup), asks the model once for a candidate for this loop and fills it
into the field. The candidate is only a proposal: look at it, edit it if needed, and press
Analyze; Prodigy's exact check decides, as for a candidate typed in by hand. *Restore previous*
puts back what was in the field before, *Answer of …* shows the model's full answer, and the
button turns into **Stop** while the request runs. The request runs in its own process like an
analysis, with the server's time limit (`--time-limit`), so the API key stays on the server.
If no model is configured, the button is disabled and the reason is shown below the field.

## Non-interactive analysis API

The while handler no longer asks questions itself. An `AnalysisSession`
(`prodigy/analysis/session.py`) carries one decision per loop; loops are identified by
their index in textual order (`prodigy outline FILE` lists them).

```python
from prodigy.pgcl import parse_pgcl
from prodigy.analysis.session import AnalysisSession

session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy",
                          decisions={"0": {"strategy": "unroll", "params": {"iterations": 50}}},
                          capture_output=True)
result = session.run().to_json()      # schema "prodigy/analysis-result@1"
```

`result["syntax"]` is the syntax of the program; query expressions, loop guards and the
instructions of `steps` are written in it (`x == 1` in `.ppy`, `x = 1` in pGCL).

`result["bounds"]["kind"]` (`exact`, `lower`, `upper`, `unknown`) is the guarantee of the
result. It holds only for `result["status"] == "ok"`; a result that failed or stopped
(`error`, `needs_decision`, `timeout`, `cancelled`) has kind `unknown` or no `bounds` at all.
A distribution, mass or query value with `nan` or ∞ in it is never `ok`: the result has status
`error` with `error.type` `NonFiniteResult` (the value ∞ of an expectation query is allowed).

From the command line:

```sh
prodigy analyze prog.ppy --loop 0=unroll:iterations=50 --json
prodigy analyze prog.ppy --loop '*=evt_synthesis'           # same strategy for all loops
prodigy analyze prog.ppy --decisions decisions.json --output result.json
prodigy loop_strategies                                     # strategies and parameters
prodigy main prog.ppy --loop 0=unroll:iterations=10         # interactive, with presets
```

`prodigy main` without `--loop` behaves exactly as before (same menu, same prompts).

### Adding a loop strategy

```python
from prodigy.analysis.loop_strategies import Bound, LoopOutcome, LoopStrategy, ParamSpec, register_loop_strategy

def run(task):            # task.instruction, task.distribution, task.param("n"), task.progress(i, n), ...
    ...
    return LoopOutcome(distribution, error_prob, kind=Bound.LOWER, residual=r,
                       certificate={"type": "my_rule", ...}, messages=["..."])

register_loop_strategy(LoopStrategy(
    name="my_rule", title="My rule", menu_text="My rule (CLI menu text)", soundness=Bound.LOWER,
    description="One or two sentences for the GUI.", run=run,
    params=(ParamSpec("n", "int", "Iterations", prompt="Iterations: ", default=10, minimum=0),)))
```

The strategy then appears in the CLI menu (after the six historic entries), in
`prodigy analyze`, and in the GUI's strategy menu with a form for its parameters.

## HTTP API (used by the GUI)

| Endpoint | |
| --- | --- |
| `GET /api/meta` | version, engines, loop strategies with parameter specs, synthesis strategies and heuristics, limits |
| `POST /api/check` `{source, syntax}` | diagnostics `{severity, message, line, column, end_line, end_column}` and outline (variables, loops with ids and lines, queries) |
| `POST /api/convert` `{source}` | the pGCL program in `.ppy`: `{source, renamings: [{from, to}], messages}`; 422 with `line`/`column` if it is not valid pGCL |
| `POST /api/analyze` | starts an analysis (`source, syntax, input_distribution, engine, normalize, simplification, decisions, coefficients, record_steps, watches, step_iterations, synthesis_strategy, template_heuristic, positivity_heuristic, solver, time_limit`; the two heuristics only apply to `synthesis_strategy` `from_heuristics`; `record_steps` records the step trace, `watches` (at most 8 queries such as `Pr[x == 3]`, in the program's syntax) are evaluated after every top-level step, `step_iterations` (1–200, default 40) bounds the iterates kept per loop) and returns a job id |
| `GET /api/jobs/<id>?log_offset=n` | state (`running`, `done`, `timeout`, `cancelled`, `crashed`), new console output, progress per loop, result |
| `DELETE /api/jobs/<id>` | cancels the analysis |
| `POST /api/jobs/<id>/joint` `{x, y, x_window, y_window}` | starts a job for the exact coefficient grid of two variables of the finished analysis `<id>`; the windows `[from, to]` are optional (default: the support of a finite distribution, else 0–15) with at most 50 values per axis (`limits.max_joint_axis` in `GET /api/meta`) and values up to 1000. The distribution, the engine and the guarantee come from the result the server keeps, never from the request. The job's result is `{status: "ok", joint: {x, y, x_values, y_values, cells, bound, mass, shown_mass, outside_mass, residual_mass, finite_support, support}}`: `cells[r][c]` is `{prob, p}` (exact and float; for a lower bound also `upper`, `upper_p`) for `y = y_values[r]`, `x = x_values[c]`. 400 for an unknown variable, the same variable twice or a window beyond the limits, 404 for an unknown or pruned analysis (analyze again), 409 while it runs |
| `POST /api/suggest` | asks the language model for one candidate for a loop: an analysis request without `coefficients` and `record_steps`, plus `loop` (id) and `form` (`evt_invariant` or `evt_subinvariant`), `time_limit` optional (default: the server's); returns a job id. The job's result is `{status: "ok", suggestion: {candidate, param, answer, model, provider, entering, loop, form}}` or an error (the model's answer in `error.suggestion` if it held no admissible candidate). `GET /api/meta` says under `ai` whether a model is configured. |
| `GET /api/examples`, `GET /api/examples/<path>?syntax=ppy` | example library; source in the syntax of the example file (`syntax`; a `.pgcl` example is converted to `.ppy` with `syntax=ppy`, with `renamings`), `file_name` and suggested setup |

Every request with a program takes `syntax` (`pgcl` or `ppy`) or `file_name` (its extension
decides); without either the program is pGCL.

## Security and hosting

The server binds to localhost. API requests must be JSON and come from the page itself
(Host and Origin are checked), which keeps other web sites from driving a local server.

Prodigy parses distributions and invariants with SymPy, whose parser evaluates Python code.
The server admits a generating function from the setup only if it consists of numbers, names,
`+ - * / ^ **`, parentheses and the functions `sqrt exp log sin cos tan`, and parses as an
expression of the program's syntax. Still, **do not expose the server to untrusted users without a sandbox** (container without
network and file system access, resource limits). For a hosted teaching instance:
`--host 0.0.0.0 --allowed-host prodigy.example.org --max-jobs 4 --time-limit 60
--memory-limit 2048` behind a reverse proxy, inside such a container.

## Design notes

The interface follows Apple's Human Interface Guidelines as far as they apply to a web
app (foundations: accessibility, colour, typography, layout, writing).

* Colours are tokens by role (`prodigy/web/static/app/app.css`), with light, dark and
  increased-contrast variants and no app-specific appearance switch. Text contrast on the
  window and content surfaces: text 16.8/15.5:1 (light) and 15.6/12.8:1 (dark), secondary
  text 5.1/4.7:1 and 6.6/5.4:1; status colours 4.95–6.8:1 (light) and 5.0–8.9:1 (dark).
* One colour, one meaning: green = exact, purple = bound, orange = no guarantee, red =
  error, blue = interaction. Status is never conveyed by colour alone (glyph ✓ ≥ ≤ ! and a
  word).
* The signature element is the *sandwich*: what is certain is drawn solid, what is only
  bounded is hatched — in the interval bar of each query and as the "?" bar of the
  residual mass in the marginal charts and the joint view.
* Sequential data (the heatmap and the 3D view of a joint distribution) use viridis, a
  perceptually uniform scale that stays ordered for the common colour-vision deficiencies;
  numbers printed on a cell are black or white, whichever has the higher contrast.
* System font for interface text (13 px body, macOS scale), monospace for code and exact
  values, KaTeX for closed forms. Controls are at least 28 px high.
* Long expressions (closed forms, invariants, certificates, printed query values) sit in
  bounded boxes (`components/formula-box.js`): the content scrolls inside its box, never the
  page, and the box takes the keyboard focus only while there is something to scroll. A box whose
  content is taller than the box offers *Show more* / *Show less* (a button with `aria-expanded`);
  the enlarged box is still bounded. In the stepper, the instruction of the step and its closed
  form sit in boxes of fixed height, the notes below the instruction have two lines reserved, and
  the status line of the Kleene iterates reserves the height of its longest line, so stepping and
  the animation do not move the views below. The console output shown while an analysis runs has a fixed height too.

## Reuse in VS Code

`prodigy/web/static/app/components/` (result view, stepper, charts, joint view, formatting) depends only on
the result JSON, the CSS variables of `result-view.css` and `stepper.css` and optionally KaTeX,
so a VS Code webview can import it unchanged and map the variables to `--vscode-*` theme colours.
The stepper (`renderStepper`) reaches the editor only through callbacks (`onIndex`,
`onSelectLine`, …); `prodigy analyze --steps --json` gives an extension the same trace. The joint
view appears when `renderResult` gets a `loadJoint(payload, signal)` callback that returns a grid.

## Updating the vendored libraries

See `prodigy/web/frontend/README.md` (`npm install && npm run build`).
