# Prodigy web GUI and analysis API

`prodigy web` starts a local web server and opens the GUI in the browser. You write a `.ppy`
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
  The chip next to the title shows the syntax of the program; see [The `.pgcl` syntax](#the-pgcl-syntax).
* **Setup** (right): initial distribution (refused with a message unless it is proven to be a
  distribution, see the language reference), one strategy per loop with its parameters (see
  [Choosing a loop strategy](#choosing-a-loop-strategy)),
  analysis options (conditioning, simplification of the shown result, number of coefficients, time
  limit, EVT synthesis strategy, solver, recording of intermediate distributions).
  *Simplify the shown result* (on by default; CLI: on unless `--no-simplification`) does not change
  the analysis (it always uses the same normal form: factored closed forms, expanded polynomials). Only the output distribution that is shown is rewritten: a bounded search tries cheap exact rewrites in the order of their cost
  (`together`, `cancel`, `factor`; GCD-based ones only below a size and degree threshold, never
  `simplify`), checks each to be equal to the computed form, stops after 2 s and shows the
  shortest form (by `count_ops`), or the computed one with the reason (“already short”, “too large”,
  “time limit reached”). The *Generating function* disclosure then switches between *Simplified* and
  *As computed*; coefficients, mass and expectations always come from the computed form (#404). The
  synthesis strategy *from heuristics* combines a template heuristic and a positivity
  heuristic chosen below it (on the command line: `--strategy from_heuristics
  --template-heuristic … --pos-heuristic …`).
* **Result** (right): two cards and a quiet line, in this order (#288, #311):
  * **Answer** (emphasised by an edge in the colour of the guarantee): the guarantee once, as a
    badge with its glyph and word and one sentence (for a lower bound with the residual mass, for
    an upper bound with the side condition of Park's lemma), timing and engine on the right. Then
    the queries as large rows: a short exact value large with its decimal small (`5/16 ≈ 0.3125`),
    otherwise the decimal large with the exact form small; the source line links to the editor. A
    query names its own kind of bound only where it differs from the overall guarantee; the
    interval of a lower bound (`∈ [0.5, 0.5 + 3 × 10⁻⁸]`) and the hatched interval bar are values
    and stay. *Other expected values* lists E[v] for the variables that no `Ex[v]` query covers.
  * **Analysis**: one card with the lenses *Distribution | Size | Time | Checks* and one timeline
    that all lenses share; see [The analysis card](#the-analysis-card). At its default position,
    the final result, the Distribution lens is the output distribution and the Checks lens lists
    every loop with its verdict and certificates.
  * **Details**: not a card but a disclosure line under the cards with the warnings, the console
    output and *Result as JSON*. If there are warnings, a notice above the cards says so and opens
    the line.

  A result belongs to the program and setup it was computed for (#310). Once either differs, a
  notice above the cards says so (*This result is for an earlier version of the program*, *…of the
  setup* or *…of the program and the setup*) with an **Analyze Again** button; the cards are
  dimmed, the guarantee badge on the Result tab and the values in the editor disappear. Changes
  made while an analysis runs are kept: the progress view notes that the run uses the earlier
  version, and its result arrives marked as outdated. Changing everything back (e.g. with undo)
  makes the result current again.

  While the coefficients of the output distribution are computed, the progress view draws them as
  they arrive (#405): one chart per variable under *Distribution so far*, with the guarantee badge.
  The values done are bars; the values still being computed are hatched and labelled
  *computing…*, and the line above each chart says *computing… 33 of 200*, *waiting* or *done*. A
  rational generating function gives its coefficients one at a time (the recursion of its
  denominator); other closed forms (`exp`, `log`, algebraic) are expanded with doubling orders
  4, 8, 16, …, each a complete prefix. The coefficients come in batches about every 100 ms and are
  exactly those of the result; when the result arrives, its chart starts from the streamed one.
  Stopping the analysis stops the streaming.

  Next to the view switch, the **View options** toggle (#452) shows or hides an inspector for all
  plots of a result (bar charts, heatmap, 3D view). It never covers the plot, so every change shows
  at once next to it: when the figure has at least 1000 px of width, the inspector is a 260 px
  column to the right of the plot; in a narrower figure it is a section between the toolbar and the
  plot, which it pushes down (its groups side by side as far as they fit). The button looks
  pressed while the inspector is open; Escape inside the inspector closes it and returns focus to
  the button. The browser remembers the choices and whether the inspector is open per viewer, not
  per analysis. It has three groups:

  * *Scale*: a slider from linear to logarithmic. Every position is the Box-Cox scale
    t(p) = (p^λ − 1)/λ, normalised between a floor and the largest value: λ = 1 is linear (from 0),
    λ = ½ the square root, λ → 0 the logarithm (from the smallest value, at most 12 decades below
    the largest). The floor moves continuously from 0 to that smallest value as λ goes from 1 to 0,
    so every position is a monotone scale with ticks at their true places. The slider snaps to
    *Linear*, *√ Root* and *Log* (buttons below it jump there) and names the scale as text, e.g.
    *Square root (λ = 0.50)*, also for screen readers. On a scale that is not linear the axis is
    marked (*log*, *√*, *λ 0.30*), smaller probabilities get the minimal bar, and a probability 0
    is a dashed outline on the axis. Only linear charts are animated.
  * *Colour*: Viridis (default), Cividis (for colour-vision deficiencies) or Greyscale (for print;
    there hatching and dashed edges carry every distinction, an empty heatmap cell has a dashed
    edge). The marginal bars take a tone of the map with at least 4.5:1 against the background.
  * *Intervals*: *Glass* (a translucent part with dashed edges from the lower to the upper end) or
    *Outline* (the dashed edges alone; the default when the system asks for reduced transparency
    or increased contrast), and *Show unaccounted mass (?)*: the hatched "?" bar of the marginal
    chart and the "?" tower beside the 3D grid, on the same scale as the bars.

  The upper ends of single values are drawn only when at least one value has an upper end of its
  own (`upper_source: "certificate"`, #451). While every upper end is the lower end plus the same
  residual mass, drawing it at every bar would repeat one number and hide the distribution: then
  only the "?" bar shows the residual, and a value's interval appears on hover (bar charts) or for
  the selected bar (3D view). The lens, the choice of variable and view are
  kept when the program is analysed again; the timeline starts at the final result.
* **Examples**: the programs in `pgfexamples/` (in `.ppy`), with a suggested setup derived from
  their first line (e.g. `check_equality` examples come with their invariant and open in Compare
  mode with both programs, see [Comparing two programs](#comparing-two-programs)). `.pgcl` files in a
  folder given with `prodigy web --examples` are not listed. The sheet lists the eight most recently opened programs (examples and local files)
  first, reachable with ↑ ↓ and Return; "Clear recent" empties the list. It is kept in the browser's
  `localStorage` only. A file is remembered with its text (up to 60 000 characters), not as a link to
  the file on disk.
* **Copy LaTeX**: the generating function of the result and of each `dist` value has a "LaTeX"
  button that copies the LaTeX KaTeX renders; the button reads "Copied" afterwards.
* **New, Open…, Save** (toolbar): in Chrome and Edge, Open and Save work directly on `.ppy` and
  `.pgcl` files on disk, and the file stays linked across reloads; Save As… (⇧⌘S) picks another file.
  Safari and Firefox cannot write to a folder: Open reads the chosen file, Save downloads the
  program. The program is also kept in the browser, together with the setup (the strategy of each
  loop), so a reload never loses either; replacing a program with unsaved changes (New, Open…, an example) asks first.
* **Engine** (toolbar, pop-up button before Analyze, #308): SymPy or GiNaC for the next run (SymEngine was removed in
  v0.6.0, #416; a setup kept in the browser that names it falls back to the default engine); GiNaC is usually much faster, SymPy is the reference implementation. An engine the server
  cannot use (GiNaC without `prodigy-ppl-ginac`) is listed as *not installed* and cannot be chosen;
  the tooltip says how to install it. The choice is part of the setup kept in the browser, stays
  across New, Open… and reloads, and changing it marks the shown result as out of date. The
  default is the server's `--engine`. In a narrow window the label is dropped (the pop-up keeps
  the name "Engine" for assistive technology) and the toolbar wraps.

Keyboard: ⌘/Ctrl + Return analyses (checks, in Compare mode), ⌥⌘1 / ⌥⌘2 choose the mode, ⌘/Ctrl + . (or Esc) stops, ⌘/Ctrl + O opens,
⌘/Ctrl + S saves; F10 / ⇧F10 move the analysis card's timeline to the next / previous position,
F5 / ⇧F5 to the next / previous breakpoint, and F9 in the editor sets a breakpoint. Every analysis runs in its own process with a time limit and can be stopped
at any time.
A message in the toolbar (an error, a notice) that does not fit is cut off with an ellipsis; a *Show full
message* button next to it opens the whole text (Esc closes it), so it is reachable by keyboard and
screen reader, and on a phone the message wraps.

What counts against the server's limits: analyses and checks, which you start, take one of the
`--max-jobs` slots (default 2). The computations the GUI starts in the background, the grids of the
joint view and the suggestions of a language model, have `--max-background-jobs` slots of their own
(default 2) and never take a slot of an analysis. Moving the timeline or changing the pair of the joint view
stops the grid that is no longer wanted, a new grid replaces the oldest running one, and a background job
that no page asks about any more (closed or reloaded) is stopped after 20 s. Each job is still bound by the
time limit and by `--memory-limit`. If an analysis is refused because the slots are taken (for example by
another browser tab), the toolbar says what is running ("An analysis of a.ppy is running") and offers
*Cancel it* / *Cancel them*; the button is focused so that Return cancels, and the message carries an icon
and text, not just a colour.

The GUI follows the system appearance (light or dark, macOS: System Settings → Appearance);
following Apple's guidelines it has no switch of its own. Firefox can override the appearance
for websites (Settings → General → Website appearance).

## Comparing two programs

The window has two modes, chosen with the segmented control **Analyze | Compare** in the toolbar
(#154; ⌥⌘1 / ⌥⌘2, Ctrl+Alt+1 / 2 elsewhere; ← → inside the control). The mode is part of the URL:
`/` (or `/analyze`) is Analyze, `/compare` is Compare, so a reload, a bookmark and the browser's
back button keep it. While a job runs the mode stays; stop it first.

* **Analyze** is the single-program GUI described above.
* **Compare** answers one question: are two programs, **A** and **B**, equivalent, i.e. do they have
  the same output distribution for every input? It never analyses a program on an input; there is
  no input to choose and no output distribution of either program. It has its own layout:
  * two editors, side by side in a wide editor pane, and in a narrow one with tabs *Program A |
    Program B* above them (← → move between the tabs); each has its syntax chip, diagnostics,
    warnings and the strategy pills above its loops;
  * its own Setup, with only what the check needs: **Loops of program A** and **Loops of program B**
    (each loop of either program has its own strategy; a section appears only for a program that has
    loops) and the time limit of the check (the engine is the toolbar's);
  * its own run button, **Check** (⌘↩), and its own result: the verdict (below).

**What the modes share and what each keeps.** Compare is a mode of its own; nothing set up for an
analysis reaches a comparison, and back:

| State | Shared by both modes | Analyze only | Compare only |
| --- | --- | --- | --- |
| Program A (the first editor, its syntax, the open file) | ✓ | | |
| Program B and the strategies of its loops | | | ✓ |
| Strategies of program A's loops | | its own | its own |
| Input | | the initial distribution | none: the check covers every input |
| Engine (toolbar), options, time limit | | its own | its own engine and time limit (no other options) |
| Last result | | its own | its own |

Each setup is kept in the browser across reloads. The strategies of program A's loops follow the
loops of the shared program in both modes when loops are inserted or removed (#277). Opening a
program (Open…, New) replaces program A in either mode and clears the loop strategies of both modes
and Analyze's input.

**Loops in Compare.** An equivalence verdict needs proven results, and the check runs every program on
all inputs at once, where unrolling or a synthesis gives no verdict. So a loop nobody chose a strategy
for gets **Check invariant**: with the other program as its invariant when its program is that one loop
and the other program is loop-free, otherwise with an invariant program still to write (Check then
says so and puts the cursor into the field). The card says that this is Compare's default; a default
follows the programs (it is recomputed when they change) until you change the strategy or the field.
The pill above the loop says *Check invariant: program B* or *Check invariant · write it in Setup*.
**Use program B** (on a loop of program A) and **Use program A** (on a loop of program B) fill in the
other program. "Compare this loop" takes only the loop-free program the loop continues with along,
never the loop's Analyze strategy.

In the example
library, the examples marked *compare* (`check_equality` in `pgfexamples/equivalence/` and
`template_parameter_synthesis/`) open in Compare mode with both programs, and a loop of program
A then uses program B as its invariant, as `prodigy check_equality` does; every other example
opens in Analyze mode. A loop of program A whose invariant is program B follows program B while
you edit it; **Use program B** next to an invariant field of program A fills it in.

### A loop against a loop-free program

Checking a loop against a loop-free program you write (*Check invariant*, the strategy
`invariant`: I ≡ if (guard) { body; I }, Park's lemma) is a comparison of two programs, so it lives
in Compare mode (#399); Analyze does not offer it in its strategy menu.

* **Compare this loop** (a button on each loop card of Analyze's Setup) opens Compare with the
  loop as program A and the loop-free program as program B: B is the loop's invariant program if it
  has one, otherwise B starts with the declarations, and the cursor waits where its instructions go.
  The loop keeps its strategy as the strategy of program A's loop; choose *Check invariant* and
  **Use program B** for the old Park check.
* **Which program A.** If the loop is all the program does (apart from queries), program A is the
  program itself. Otherwise program A is the program's declarations and the loop alone: the code
  before the loop (initialisations, enclosing `if`s) is left out, because a comparison holds for
  every input and so covers whatever that code produces; the code after the loop does not concern
  the loop. Loops inside functions cannot be compared. The initial distribution of Analyze's Setup
  belongs to the whole program and stays in Analyze; the comparison has no input.
* **Back to Analyze** (the mode switch, or the continue flow below) brings back the program the
  loop came from, with its strategies and initial distribution, as long as program A is still that
  loop (⌘Z in the editor undoes it). While program A is one loop of the program, Save is refused
  so that the file never gets only the loop.
* **Continue in Analyze with B as the loop's result.** When the verdict is *Equivalent* or *One
  direction is proven* with Program A ⪯ Program B, the verdict card offers this button. It goes
  back to Analyze, sets the loop's strategy and analyses the program, which continues after the loop
  with B's result:

  | Verdict | Strategy of the loop in Analyze | Guarantee |
  | --- | --- | --- |
  | Equivalent | the strategy that proved program A exact in the comparison (the strategy `invariant` alone gives only an upper bound) | exact, if that strategy proves it again on the program's input; the result says so |
  | Program A ⪯ Program B | Check invariant with program B, checked again (Park's lemma) | upper bound, exact if the loop terminates almost surely |

  Analyze always proves the loop's result itself; the verdict is not taken on trust. The button is
  not offered when program A is not exactly one loop, program B has loops, the verdict holds only
  for parameter values (put them into B first), or only Program B ⪯ Program A is proven (B is then
  a lower bound of the loop, which this strategy cannot give; unroll the loop for lower bounds).
* **Saved setups and the library.** A loop whose strategy is still `invariant` (saved in the
  browser, an example marked *compare*, or the continue flow) keeps it and runs in Analyze: its card
  says what it is, shows program B, and offers **Open in Compare** and **Choose a strategy** instead
  of the strategy menu. On the command line `--loop invariant` is unchanged.

The check runs as a job like an analysis (own process, time limit, Stop). It has two parts:

1. **The check for every input.** Both programs run on one generating function that stands for
   every input at once (the second-order check of `prodigy check_equality`), and the solver
   compares the results; for programs with parameters it looks for parameter values that make
   them equal (parameter synthesis, see [Check invariant and parameter synthesis](#choosing-a-loop-strategy));
   the verdict card lists the values found.
2. **The witness search**, unless the check proved the computed distributions equal: both programs
   run on one input, the solver's counterexample (or the input on which an invariant was refuted, or
   every variable 0), first with the strategies of the Setup, then with every loop unrolled 30 times
   (a lower bound with its residual mass is still sound evidence). It has at most 20 s (a third of the
   time limit if that is shorter); a search that runs out of time leaves the verdict open and says so
   (witnesses on logarithmic closed forms can take minutes, #348). It is not an analysis: its only
   output is the witness.

The **verdict** at the top has a glyph and a word:

| Verdict | When |
| --- | --- |
| `=` **Equivalent** | the check proved equality, and both programs were analysed exactly in it (with parameters: for the parameter values shown) |
| `≠` **Not equivalent** | only with a **witness**: an input and an output state whose probabilities provably differ, i.e. the sound intervals of the two programs (exact value, lower bound plus residual mass, or Park upper bound) do not overlap |
| `?` **Not decided** | everything else, with the reason: the check could not decide, it is *not implemented* for these programs (shown as such, never as "not equivalent"), the programs declare different variables, only one direction is proven, or the witness search ran out of time |

*One direction is proven* happens when a loop was handled by an invariant: Park's lemma gives an
upper bound, so equality of the computed distributions proves only *Program A ⪯ Program B*
(coefficient-wise, for every input); they are equal if the loop terminates almost surely, which
Prodigy does not check. Likewise an unrolled loop (a lower bound) gives the other direction. A
raw "the generating functions differ" of the solver is not a verdict either: it may come from an
approximation, so without a witness the result stays *Not decided*. *How it was checked* lists
what the check found (and the difference of the generating functions if it could not decide).

The **witness** is shown compactly on the verdict card: the input, the output state, and the
probability (or interval) of that state in program A and in program B with their guarantees. It is
the evidence for *Not equivalent*, not an analysis; to see a program's output distribution, analyse
it in Analyze.

The result of a check is `prodigy/equivalence-result@1` with the key `equivalence`
(see [HTTP API](#http-api-used-by-the-gui)).

## Joint distribution of two variables

Choosing the pair (`x and y`, or *Two variables* with more than two) in the control row of the
Distribution lens shows Pr[x = i, y = j] for two chosen variables;
the other variables are marginalised. The coefficients are exact: the server computes them from
the distribution of the finished analysis it keeps (`POST /api/jobs/<id>/joint`, a short job of
its own, see `prodigy/analysis/joint_grid.py`), the browser only draws them. At a position of the
timeline other than the final result, the grid is that of the recorded step or iterate (the
request names `step` and `iterate`; a closed form the trace kept only in part has no grid). The
grid grows from its corner by total degree while it is computed; the cells not computed yet are
hatched, and the status line says *Computing… n of m cells*.

* **Pair**: the variables on the horizontal and the vertical axis (⇄ swaps them).
* **Window**: the values *from*–*to* of each axis, at most 50 per axis and up to 1000; *Show
  window* (or Return) computes them. A finite distribution starts with its support (the status
  line names it), an infinite one with the values 0–15.
* **Heatmap**: one cell per pair of values, coloured by probability on the colour map and scale of
  the *View options* (viridis by default, see the bar charts above); probability 0 is an empty grey
  cell, a value that depends on parameters is hatched. Both ends of the colour key are labelled. Values never depend on colour
  alone: the line below the heatmap names the selected cell with its probability as an interval
  (`x = 1, y = 2: [1/36, 13/360] ([0.0277777, 0.0361112])` for a lower bound, see *Probabilities as
  intervals* below), large cells print their value, and the table lists all of them. Keyboard: Tab focuses the heatmap, the arrow keys move between cells (↑ is the next
  larger y), Home/End go to the first/last column, Page Up/Page Down to the top/bottom row; the
  pointer selects the cell under it.
* **3D**: a bar per cell (plain canvas, no library), height and colour by probability; the view fits the bars, so low bars leave no empty space.
  For a lower bound the selected bar carries its interval as a glass or outlined box up to the
  upper end, and a hatched tower labelled "?" beside the grid is the mass not yet accounted for; a
  key below the view says both in words, and pointing at the tower shows its value. Pointing at
  a bar (or at the floor of a cell with probability 0) shows a tooltip with the values of both
  variables and the probability, formatted as in the heatmap's line; the bar gets an outline and the
  line below the view repeats the text. Keyboard: Tab focuses the view and shows the tooltip of the
  selected bar; the arrow keys, Home/End and Page Up/Page Down move between bars as in the heatmap
  (the heatmap and the 3D view share the selected cell). Drag or Shift + ←/→ rotate, Shift + ↑/↓
  tilt, pinch (⌘/Ctrl + wheel) or +/− zoom, 0 resets; the buttons above the view do the same.
* **Table**: the window as a table, rows y (largest on top) and columns x; hovering a value shows
  its exact form.
* **Colour key and mass**: two separate, labelled blocks below the heatmap and the 3D view.
  *Colour: probability of one cell* is the key of a single cell's probability on the chosen colour
  map (the heading names the scale and the range). The card's *Mass* line, set apart by a rule,
  adds up the cells: the total mass, the mass in the window, the mass outside it, and for a lower
  bound the hatched "?" mass not yet accounted for, which may still end up in any cell, outside the
  window, or never terminate. For a lower bound every cell is the lower end of an interval whose
  upper end adds the residual mass (at most 1); for an upper bound every cell is the upper end of
  [0, value]. The colour shows the computed value (the lower or upper end; the key says which), and
  a note below the view states the guarantee in words.

**Probabilities as intervals** (#405). Bar charts, their tooltips and tables, the heatmap's line,
the 3D view's tooltip and the grid table show each probability by its guarantee:

* exact: the value, `1/4 (0.25)`;
* lower bound with residual mass r: `[p, min(1, p + r)]`, e.g. `[1/4, 3/8] ([0.25, 0.375])`;
* upper bound: `[0, p]`;
* no guarantee: `unknown (computed 0.25, no guarantee)`.

Exact rationals are shown with their decimals; the decimals of an interval are rounded outward (the
lower end down, the upper end up, 6 significant digits), so the printed interval still contains the
true value. A bar of a lower bound is solid up to the lower end; its part up to the upper end is drawn as
glass or outline (see *View options*); a bar of an upper bound is hatched. The guarantee is also stated in words below each chart and table
(*Exact probabilities.*, *Bounds: …*), and in the streamed charts while the analysis runs.

Moving along the timeline or playing it redraws the joint view for each position (#289): the pair,
the view and a window chosen with *Show window* stay (without one, each position gets its default
window), and the colour scale keeps the largest probability seen for this pair and window, so that
the frames can be compared. The joint view itself is not animated, so it needs no special case for
reduced motion; it follows light and dark mode (the 3D view redraws when the appearance changes).

## The analysis card

The **Analysis** card of the Result tab (#311) shows how the result came about. It has four
**lenses**, *Distribution | Size | Time | Checks*, all with the same layout: a header that says where
the timeline stands (`after step 9 of 14 · line 4`, `in step 5 of 14 · loop 0, iteration 3 of 25`,
or `final result`) with the instruction of that step, the lens's controls, its figure, one fact line,
the timeline, and the details as disclosures. The answer card above always shows the final result,
whatever the position.

* **Timeline**: one position per recorded step; a loop is a shaded segment, and its recorded iterates
  are the positions inside it before the loop's exit; the last position is the final result, which
  is also where the timeline starts. Previous / next (← →, or F10 / ⇧F10 anywhere), *Play* / *Pause*
  (Space on the slider; it never starts by itself), the slider (click or drag; Home / End: start and
  final result, Page Up / Down: ten positions), *Final Result*, and F5 / ⇧F5 to the next / previous
  step on a line with a breakpoint (a click on a line number in the editor, or F9, sets one; the
  timeline marks it with a dot). The editor highlights the instruction of the position and marks it
  with ▶. While the timeline moves, the parts of the card above it keep their largest height, so its
  buttons stay under the pointer. Moving between positions tweens the bars of a chart in about a third
  of a second; with *Reduce motion* (macOS: System Settings → Accessibility → Display) the views jump.
* **Recording**: the timeline needs the steps of the analysis. A result without them offers
  *Record Steps* (the option *Record each step* in Setup, then analyse again) in place of the
  timeline; the whole trace is recorded within the time limit of the analysis and navigated
  afterwards, so going back is as cheap as going forward.
* **Steps**: every top-level instruction is a step, and so is every instruction inside the
  branches of `if`, `switch` and probabilistic choices; those show the *sub-distribution* of their
  branch (the part where the guard holds, or weighted by the probability of the branch), not
  normalised. Instructions inside loop bodies, function bodies and the checks of invariants are
  no steps. If observations were violated, a last step conditions on them.
* **Distribution**: the distribution after the step (before it, if the step failed) with the
  controls of the output distribution: the variable or a pair (heatmap, 3D, table, see
  [Joint distribution of two variables](#joint-distribution-of-two-variables)), chart or table, and
  the scale. Each position carries the guarantee that holds there (exact, lower bound with residual
  mass, upper bound, or none), as a badge with a sentence. The fact line is the **Mass** line; at
  an iterate it splits the mass into what has left the loop and the hatched “?” still inside it,
  which shrinks from iterate to iterate. A marginal that the trace did not record is shown as a
  dashed frame “x: not recorded” with the reason, never as probability 0. Disclosures: the
  *Generating function* (KaTeX on every engine, with copy buttons), *Facts* (support, parameters,
  violated observations; at a step: mass and size), the values of the `dist` variables (at the end
  of the program: *Inferred distributions* with their `infer(…)`, normalised generating function,
  expected value, normalising constant Z and coefficients; at a step: their closed forms, also with
  KaTeX), and **Queries after every step**: a query such as `Pr[x == 3]` or `Ex[x]` (in the
  program's syntax) is evaluated after every top-level step, as if it stood right after that
  instruction, with the same sound intervals as the program's queries; adding or removing one
  analyses again (recording the steps).
* **Size**: the length of the closed form in characters (the same measure on every engine) per
  position as a line chart, linear or log, with the loop segments shaded and the current position
  marked with its value; a click on the chart moves the timeline there. Fact line: at this position,
  the largest with its step and line, the final result. Disclosures: where the closed form grows most
  (*Show this step*) and the *Table of all positions* (line, instruction, guarantee, size, time; a
  button per row moves the timeline), which is also the accessible alternative of both charts.
* **Time**: the time of each step without its nested steps as bars, so they add up; a loop is one
  step, labelled with its time, and its iterates have no bars of their own. Fact line: at this
  position, the slowest step, the total analysis time, and the time spent recording the steps,
  which the steps do not count. Disclosures: where most of the time goes and the table.
* **Checks**: at a loop (its iterates or its exit), the loop's strategy, verdict and messages, the
  checks of its certificate in words (fixed point, non-negativity, finite mass, sub-invariance …,
  each *proven*, *failed*, *not established* or a note, with a glyph), the certificate as a
  disclosure and, at an iterate, how much has left the loop. At the final result: every loop with
  its line, strategy, verdict in words (`✓ exact`, `≥ lower bound, at most … not accounted for`,
  `≤ upper bound`, `! no guarantee`) and time; a failed loop keeps its badge (*Failed*, *Refuted*,
  *Not proven*) and its message. At another step: *no loop here*, and a button per loop to go to it.

Per distribution the trace keeps the closed form up to 3000 characters, its mass and the first 10
coefficients of the marginals of up to 6 variables; larger closed forms keep no marginals, and
after 20 seconds spent on marginals only closed forms are kept. Each distribution names the
variables without a marginal and why (`marginals_not_recorded`). At most 40 iterates are kept per
loop (`step_iterations`), spread evenly, plus the last. The command line records the same trace
with `prodigy analyze --steps` (see the CLI documentation); the JSON keys are `steps` and `trace`
(see `prodigy/analysis/trace.py`).

## Choosing a loop strategy

Each loop card has a pop-up button **Strategy** with one section per goal (#324). Below it, one
sentence says what you get, what you must provide and roughly what it costs; the chip in the card's
corner names the outcome. *How to choose a strategy* (above the loop cards) lists all of them.

| Group | Strategies | What you get |
| --- | --- | --- |
| Automatic | Automatic (`auto`, the default) | exact if Prodigy finds an EVT invariant within a few seconds, otherwise a lower bound from unrolling N iterations |
| Exact result | Check EVT invariant, Synthesise EVT invariant, EVT invariant from a language model (and Check invariant, in Compare mode) | exact, or an upper bound if a side condition fails (finite mass, almost-sure termination); no result without a proof |
| Bounds | Unroll N iterations, Iterate until mass captured, Check EVT sub-invariant, EVT sandwich | lower bounds (every query an interval) without a termination assumption; the sandwich also upper bounds |
| Explore | Expected visiting times H_N (`evt_iterate`) | the iterates of the expected visiting times, to guess an EVT invariant; the result is only a lower bound |

**Check invariant and parameter synthesis.** *Check invariant* (`invariant`) checks a concrete
loop-free program I (I ≡ if (guard) { body; I } for all inputs) and continues with it. Since it
needs a second program, the GUI offers it in Compare mode only (see
[A loop against a loop-free program](#a-loop-against-a-loop-free-program)); Analyze runs it for
loops that already have it. Finding parameter values instead (an I with parameters the program
does not have, as in `pgfexamples/template_parameter_synthesis/`, or a check that fixes the
program's parameters to values) is a comparison of the loop with I, so it belongs to Compare mode
(#154). In Analyze mode such an invariant ends with *Parameter synthesis is a comparison* and the
button **Open in Compare**: the loop becomes program A (the program itself if the loop is all it
does, else its declarations and the loop; undo restores yours), with I as the strategy of that loop,
and I becomes program B; **Check** then reports the parameter values on the verdict card. In
Compare mode the strategy accepts such invariants for the loops of either program. On the command
line `--loop invariant` keeps working with a parametric I and prints a hint to use
`prodigy check_equality`. All other strategies are analysis steps and are the same in both modes.

Parameters that must be given (an invariant, a sub-invariant) are shown right away; those with a
default sit behind the disclosure **Parameters**, whose summary shows their values
(`Iterations N = 25`). *About this strategy* has the longer description. The names, the `--loop`
syntax and the CLI menu numbers stay as they were; the CLI menu, `prodigy loop_strategies` (grouped
the same way) and the GUI use the same titles and sentences, which live in the registry
(`LoopStrategy.group`, `summary`, `outcome`).

**The default.** A loop without a decision gets *Automatic* (`auto`): it runs *Synthesise EVT
invariant* with templates up to degree 2 and a time limit of 5 s (2 s per template), and keeps its
result only if it is certified **exact**. Otherwise (no invariant, an invariant with infinite mass
that only gives an upper bound, an observe statement before the loop, or any failure) it unrolls
N = 25 iterations (50 on the command line), which gives a sound lower bound with residual mass.
Both parts are the unchanged strategies with their own checks, so *Automatic* is exactly as sound
as they are; the certificate says which one was used and why (`automatic`), and *Checks* shows it.
Why this default: unrolling alone never gives an exact answer for a loop that does not terminate
within N steps, while many textbook loops (geometric loops, sequential and nested loops) have small
rational invariants that the synthesis finds in about a second; the time limit keeps the cost of a
failed search small. An upper bound alone is not taken, since the lower bound of unrolling is the
safer default answer.

### Exploring expected visiting times

*Expected visiting times H_N* (`evt_iterate`) iterates the EVT operator
Ψ_G(H) = G + ⟦body⟧(⟨H⟩_guard) from 0. Its iterates are **not output distributions**: the
coefficient of a state in H_N is the expected number of times the loop head is reached in that
state within N iterations; their limit is the EVT invariant that *Check EVT invariant* checks. The
result view shows a card **Expected visiting times** (labelled *exploration*) above the analysis
card: a slider over N (up to 12 iterates, spread evenly), the closed form of H_N, and a table of the
coefficients of the first 30 states (rows) for each N (columns), shaded by value; the value is
always written in the cell, and the column of the chosen N is outlined. Below it, a **candidate**:
a rational generating function guessed from the coefficients (per variable, SymPy's rational guess
over the slices of the other variables, after extrapolating the last three iterates with Aitken's
Δ² process), or, without one, the closed form of the chosen H_N. **Use as EVT invariant
candidate** switches the loop to *Check EVT invariant*, fills in the candidate and focuses the field
in Setup; Analyze then checks it exactly. The guess is only a proposal and never part of a result.
For the loops of `pgfexamples/invariant_synthesis` `geometric`, `geometric_counter` and
`thirds_geometric`, the guess after 20 iterations is proven exact; for random walks it usually finds
none, since their coefficients converge slowly.

The loop still contributes a sound **lower bound**: the terminating part of H_N, the mass that left
the loop within N iterations (as before; the answer card says *lower bound*, and the exploration
card says that this is not the program's distribution). The command line keeps the strategy for
scripts; `prodigy analyze --json` has the exploration in the loop's certificate (`exploration`:
`iterates`, `states`, `guess`).

## The `.pgcl` syntax

Programs are written in `.ppy`. The `.pgcl` syntax was removed in v0.6.0, together with
*Convert to .ppy*: the server answers a `.pgcl` program with the error "The .pgcl syntax was
removed in v0.6.0. Convert the program to .ppy with Prodigy v0.5.x (`prodigy convert
program.pgcl`, or Convert to .ppy in its GUI).", shown in the editor. (The editor still offers
a pGCL mode for New and Open; it goes in a later step of #433.)

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
| lower | coefficient-wise below the semantics; at most the *residual mass* r is unaccounted for | unroll N, iterate until mass captured, expected visiting times H_N, automatic (if no exact invariant was found) |
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
non-negativity with the configured positivity heuristic (three-valued: non-negative, a negative
coefficient, or undecided; program parameters are real coefficients of unknown sign, never series
variables, so a coefficient such as 1/(1 − p) stays undecided) and decides finiteness of the mass
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

An invariant may also be an exponential polynomial `A·exp(c·k) + B` (polynomials A and B), which
arises from a choice probability such as `1/(k + 1)` in a counter k (issue #323): such a
probability acts on the generating function as an integral operator, and the fixed-point check
needs only derivatives and evaluation at 0. Its non-negativity is decided coefficient-wise, and
its series converges everywhere, so with non-negative coefficients the result is exact. The same
holds for logarithmic invariants `P + Q·log(1 - c·k)` (exact with finite mass, otherwise an upper
bound) and for products of such factors over disjoint variables. See
"Probabilities rational in a counter" in `docs/source/language-design.md`.

An invariant that fails its check gets a verdict. *Refuted* means it is provably wrong: a
counterexample, or Φ(I) − I is a nonzero rational function after exact cancellation.
*Not proven* means the check could neither prove nor refute it, e.g. for non-rational
closed forms such as `sin(c)^2 + cos(c)^2 - 1`, which SymPy simplifies to 0 but
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
The fixed-point equation is linear in the coefficients of N, so these are solved by linear algebra and
only those of D need a non-linear solver; templates with few unknowns come first, those of D counted
eight times. Templates whose numerator would have more than 500 coefficients (with many variables,
e.g. degree 2 and more in seven variables) are left out, and the message names the degree reached. Every solution
is a proposal that the exact fixed-point check decides. Fixed points whose coefficients the positivity
heuristic cannot show to be non-negative are not results: if no certified invariant is found, they are
listed as candidates in the loop's messages and in `error.details.candidates`, e.g. for a program
parameter p the candidate (1 + b·p)/(1 − (1 − p)·x), which may be checked by other means.
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
invariant I* of *Check EVT invariant* and the field *EVT sub-invariant nu* of *Check EVT
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
A distribution, mass or error probability with `nan` or ∞ in it is never `ok`: the result has status
`error` with `error.type` `NonFiniteResult`. A query whose value is undefined, or that cannot be
computed, loses its guarantee and gets a warning, but the analysis and its distribution stay (#316).
The value ∞ of an expectation or variance query is a value, with the result's guarantee: under an
upper bound it reads `Ex[x] ≤ ∞`.

From the command line:

```sh
prodigy analyze prog.ppy --loop 0=unroll:iterations=50 --json
prodigy analyze prog.ppy --loop '*=auto'                    # same strategy for all loops
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
    name="my_rule", title="My rule", group="bounds", outcome="lower bound", soundness=Bound.LOWER,
    summary="A lower bound; you provide n; the cost grows with n.",   # CLI menu, loop_strategies, GUI
    description="A few sentences for help texts.", run=run,
    params=(ParamSpec("n", "int", "Iterations", prompt="Iterations: ", default=10, minimum=0),)))
```

The strategy then appears in the CLI menu (after the historic entries), in `prodigy analyze`, and
in the GUI's strategy pop-up in the section of its `group` (`automatic`, `exact`, `bounds`,
`explore`, or `other`, shown as *More*) with a form for its parameters.

## HTTP API (used by the GUI)

| Endpoint | |
| --- | --- |
| `GET /`, `GET /analyze`, `GET /compare` | the page; the path selects the mode (Analyze or Compare), which the page keeps in the URL |
| `GET /api/meta` | version, engines (`name, label, available, reason, default`; `reason` says why an unavailable engine cannot be used), loop strategies with parameter specs, synthesis strategies and heuristics, limits |
| `POST /api/check` `{source, syntax}` | diagnostics `{severity, message, line, column, end_line, end_column}` and outline (variables, loops with ids and lines, queries); with `input_distribution` (and `engine`) also `input_distribution: {status: ok | refused | skipped | timeout, message}`, shown at the Setup field (the check runs in a process that is killed after 8 s) |
| `POST /api/analyze` | starts an analysis (`source, syntax, input_distribution, engine, normalize, simplification, decisions, coefficients, record_steps, watches, step_iterations, synthesis_strategy, template_heuristic, positivity_heuristic, solver, time_limit`; the two heuristics only apply to `synthesis_strategy` `from_heuristics`; `record_steps` records the step trace, `watches` (at most 8 queries such as `Pr[x == 3]`, in the program's syntax) are evaluated after every top-level step, `step_iterations` (1–200, default 40) bounds the iterates kept per loop) and returns a job id |
| `GET /api/jobs` | the running jobs: `{jobs, background}`, each `{id, kind, description, elapsed, auxiliary}`; `jobs` are analyses and checks, `background` grids and suggestions |
| `GET /api/jobs/<id>?log_offset=n` | state (`running`, `done`, `timeout`, `cancelled`, `crashed`), new console output, progress per loop, result; while it runs, `stream`: the coefficients computed so far (`variables`, `limit`, `bound`, `residual_mass`, `marginals: {x: {rows, done}}`, or for a grid job `grid` with `cells[r][c]` null while pending; `prodigy/web/jobs.py: merge_stream`) |
| `DELETE /api/jobs/<id>` | cancels the analysis |
| `POST /api/jobs/<id>/joint` `{x, y, x_window, y_window}` | starts a job for the exact coefficient grid of two variables of the finished analysis `<id>` (not of a comparison, which has no distributions); the windows `[from, to]` are optional (default: the support of a finite distribution, else 0–15) with at most 50 values per axis (`limits.max_joint_axis` in `GET /api/meta`) and values up to 1000. The distribution, the engine and the guarantee come from the result the server keeps, never from the request. The job's result is `{status: "ok", joint: {x, y, x_values, y_values, cells, bound, mass, shown_mass, outside_mass, residual_mass, finite_support, support}}`: `cells[r][c]` is `{prob, p}` (exact and float; for a lower bound also `upper`, `upper_p`) for `y = y_values[r]`, `x = x_values[c]`. 400 for an unknown variable, the same variable twice or a window beyond the limits, 404 for an unknown or pruned analysis (analyze again), 409 while it runs |
| `POST /api/equivalence` | checks whether program A and program B are equivalent for every input: `source`, `syntax`, `decisions`, `engine`, `time_limit` and the EVT synthesis options of program A, plus `other_source`, `other_syntax` (default: `syntax`) and `other_decisions` (the strategies of B's loops, keyed by B's own loop ids; B never takes A's decisions) for program B. There is no initial distribution: a request with `input_distribution` is refused (400). Returns a job id; the job streams no coefficients. Its result is `{schema: "prodigy/equivalence-result@1", prodigy_version, status, error, syntax, loops, other_loops, log, timing, equivalence}`, where `loops` and `other_loops` are the loop records of programs A and B in the check (as in an analysis result) and `equivalence` is `{verdict: equivalent \| not_equivalent \| unknown, reason, relation: null \| first_below_second \| second_below_first, constraints: [{param: value}], witness: null \| {input, state, first, second}, witness_search: null \| timeout, check: {answer: equal \| different \| undecided \| failed, first, second, constraints, counterexample, difference, error, time}}`. `input` and `state` of the witness are variable values (every variable of the input; the output state as the engine lists it); `first`/`second` of the witness are `{lower, lower_float, upper, upper_float, kind}`; `check.first`/`check.second` are the guarantees (`exact`, `lower`, `upper`, `unknown`) of the two analyses inside the check. The rules are in `prodigy/analysis/equivalence/verdict.py`. Earlier versions returned the analysis result of program A on one input, with `equivalence.input` and `equivalence.other` (program B's analysis result); both are gone, and a comparison has no grids. |
| `POST /api/suggest` | asks the language model for one candidate for a loop: an analysis request without `coefficients` and `record_steps`, plus `loop` (id) and `form` (`evt_invariant` or `evt_subinvariant`), `time_limit` optional (default: the server's); returns a job id. The job's result is `{status: "ok", suggestion: {candidate, param, answer, model, provider, entering, loop, form}}` or an error (the model's answer in `error.suggestion` if it held no admissible candidate). `GET /api/meta` says under `ai` whether a model is configured. |
| `GET /api/examples`, `GET /api/examples/<path>` | example library (`.ppy` files); source, `syntax` (`ppy`; the parameter `syntax=pgcl` gives 400), `file_name` and suggested setup; a `check_equality` example also has `other: {id, source, syntax, file_name}`, its second program |

Programs are `.ppy`. A request with `syntax` `pgcl` or a `file_name` ending in `.pgcl` is
refused with the removal message (`POST /api/check` reports it as an error diagnostic).

## Security and hosting

The server binds to localhost. API requests must be JSON and come from the page itself
(Host and Origin are checked), which keeps other web sites from driving a local server.

Prodigy parses distributions and invariants with SymPy, whose parser evaluates Python code.
The server admits a generating function from the setup only if it consists of numbers, names,
`+ - * / ^ **`, parentheses and the functions `sqrt exp log sin cos tan`, and parses as an
expression of the program's syntax. Still, **do not expose the server to untrusted users without a sandbox** (container without
network and file system access, resource limits). For a hosted teaching instance:
`--host 0.0.0.0 --allowed-host prodigy.example.org --max-jobs 4 --max-background-jobs 4 --time-limit 60
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
  perceptually uniform scale that stays ordered for the common colour-vision deficiencies, or
  cividis or greyscale from the *View options*; numbers printed on a cell are black or white,
  whichever has the higher contrast (at least 4.5:1 on every colour of the three maps).
* System font for interface text (13 px body, macOS scale), monospace for code and exact
  values, KaTeX for closed forms. Controls are at least 28 px high.
* Long expressions (closed forms, invariants, certificates, printed query values) sit in
  bounded boxes (`components/formula-box.js`): the content scrolls inside its box, never the
  page, and the box takes the keyboard focus only while there is something to scroll. A box whose
  content is taller than the box offers *Show more* / *Show less* (a button with `aria-expanded`);
  the enlarged box is still bounded. The console output shown while an analysis runs has a fixed
  height.

## Reuse in VS Code

`prodigy/web/static/app/components/` (result view, analysis card, timeline, charts, joint view,
formatting) depends only on the result JSON, the CSS variables of `result-view.css` and
`analysis-card.css` and optionally KaTeX, so a VS Code webview can import it unchanged and map the
variables to `--vscode-*` theme colours. The analysis card reaches the editor only through callbacks
(`onPosition` with the step to highlight, `onSelectLine`, `onRecord`, …); `prodigy analyze --steps
--json` gives an extension the same trace. The joint view appears when `renderResult` gets a
`loadJoint(payload, signal, onPartial)` callback that returns a grid (and passes the cells computed so
far to `onPartial`).

## Updating the vendored libraries

See `prodigy/web/frontend/README.md` (`npm install && npm run build`).
