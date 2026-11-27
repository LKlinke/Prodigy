# Architecture

This page states the **target** structure of Prodigy that the maintenance pass of v0.7.0 (#221, #493) works
towards, and how far the code is from it today. The dependency rules are checked by `tests/test_layering.py`;
every rule that the code still breaks is an expected failure there, naming the item below (`T1` …) and the work
package that removes it. Package paths are the current ones; `prodigy/pgcl` became `prodigy/language` with #446.

## Goals

The code must stay workable for a person reading it without an assistant:

- **Small modules.** At most about 800 lines per Python module (pylint `max-module-lines`, enforced when
  `too-many-lines` is switched on); six modules are larger today, none of them in `prodigy/distribution` (the SymPy engine `pgf/sympy/engine.py` went from 1 454 to 794 lines in P4b–c).
- **Every package says what it owns** in its docstring, and what it does not own.
- **Shallow, one-way dependencies.** Layers import downwards only; no import cycles, also not through imports
  inside functions; dependencies are passed explicitly, not fetched from a global.
- **One place per piece of logic.** A case split, a parser of some text, an arithmetic: each exists once.

Good design is not given up for these numbers: a module that is long because its subject is one coherent rule
(with a docstring that explains it) is a justified local exception.

## Layers

```text
 front ends   prodigy.cli            prodigy.web (Flask + ES modules)      prodigy.plotting (renders plot data)
                   |                          |                                  |
                   +------ front-end API: session, result, config, loop registry, exceptions ------+
                                              |
 analysis     session          AnalysisSession: runs a program text, decisions, trace, assembles the result
                   |
              semantics        analyzer + instructionhandler/* (one handler per construct)
                   |           loops/* (unrolling, invariants/exact, invariants/bounds, registry)
                   |           nested inference, equivalence (compare mode), independence, optimization
                   |
              vocabulary       Bound (exact/lower/upper/unknown), the loop-strategy API (LoopDecision,
                   |           ParamSpec, LoopTask, LoopOutcome), ForwardAnalysisConfig (engine choice)
 ----------------- | ------------------------------------------------------------------------------------------
 distribution prodigy.distribution
                   |           Distribution: the semantic interface (representation-neutral)
                   |           PGF: shared logic on generating functions, proof rules on closed forms
                   |           engines SymPy, GiNaC (hooks);  kernel: FLINT, exact rational arithmetic
 language     prodigy.language
                   |           parser, AST, type checker, warnings, outline (loops, queries, positions),
                   |           printer (spelling), structured diagnostics
 util         prodigy.util     helpers without Prodigy knowledge

 simulator    prodigy.simulator   sampling semantics beside the analysis; imports only the language
```

Every arrow points down. Inside the analysis the three sub-layers follow the same rule: the vocabulary imports
no handler, strategy or session; the semantics imports no session; nothing imports a front end.

| Layer | Owns | Does not own |
|---|---|---|
| util | helpers | anything about programs or distributions |
| language | the `.ppy` text: parsing, AST, types, warnings, the *outline* of a program (its loops with ids, positions, depth and function; its queries), spelling an AST, positions and structured diagnostics (line, column, message) | any CAS; proof rules; GUI wording |
| distribution | the meaning of a program state: the `Distribution` interface, the generating-function logic and its proof rules, JSON display values | which engine is chosen; loops, strategies, results |
| analysis vocabulary | the soundness vocabulary `Bound`, the interface between the loop handler and the strategies, the configuration and with it the engine choice | running anything |
| analysis semantics | the forward semantics of each construct, the proof rules for loops, nested inference, program comparison | sessions, JSON, text output |
| analysis session | running an analysis on a program text: decisions per loop, the trace, the result `prodigy/analysis-result@1` | presentation |
| front ends | options, files, HTTP, jobs, text and graphical output | analysis internals beyond the front-end API |

## Interfaces between the layers

**Types that cross.** Language → everything: `Program`, `Expr`, `Instr`, `Type`, and the outline (`language/outline.py`:
`LoopInfo`, `QueryInfo`, `ProgramOutline`). Distribution → analysis: `Distribution` and its factory, and the proof-rule
functions with three-valued verdicts (`True`, `False` with a witness, `None`). Analysis → front ends:
`AnalysisSession`, `ForwardAnalysisConfig` with its option enums, the loop registry, the result object and its
JSON (typed with #379/P9), the value helpers (`values.py`), the checks before a run (`checks.py`), the trace,
the comparison verdict, the exceptions, and `tools.py`: the re-exports of what the CLI's commands and the GUI's
endpoints need beyond the session (invariant synthesis and verification, sandwich bounds, independence, the
joint grid, language-model candidates). A module moved inside the analysis changes an import there, not a
front end.

**Where text is parsed.** Program text is parsed once by the language layer, where it enters (front end or
`AnalysisSession`). Positions of loops and queries come from the parser (`language/outline.py`), not from regular
expressions over the source. Expressions from users or language models are parsed by the
language parser or an allowlist before SymPy sees them, and only once (watches: three times today, review §2.3).
Engine text is written by `language/engine_text.py` and turned into SymPy by one public function of the
distribution layer, `parse_engine_text` (#479); `str()` of the AST is `.ppy` and never reaches an engine.

**Where text is spelled.** An AST by the language printer; a distribution by `distribution.display` (and
`formula-box.js` in the GUI); diagnostics are built from the language's structured diagnostics, the front end
adds wording and envelope.

**Where the engine is chosen.** Once, in `ForwardAnalysisConfig.factory`. Nothing above imports an engine
module or `prodigy_ginac`; the engines' availability is a function of the configuration.

**Dependencies are passed, not fetched.** Every instruction handler is a function
`handle(frame, instruction, state) -> ProgramState`, dispatched by the table `analyzer.HANDLERS`. The immutable
`Frame` carries the `AnalysisContext` (the session's services: spelling, loop decisions, trace, the registry of
`dist` values), the program, the configuration and the semantics for nested instructions; `ProgramState` is the
distribution and the violation mass. Only the entry points (`compute_discrete_distribution`, which also takes
`session=`, and the positional `compute_semantics`) look up the active session with `boundary_context()`.

## Distributions: interface, generating functions, engines, kernel

**`Distribution` is the semantic interface** and nothing else: the operations of the semantics (`filter`,
`update`, `marginal`, probabilities, expectations, `normalize`, `approximate`, the comparisons
`decide_equal`/`decide_le`). It does not mention generating functions, so another representation could
implement it. It lives alone in `distribution.py`, with `CommonDistributionsFactory`; the value types are in
`values.py`, the errors in `errors.py`, the term helpers in `terms.py`.

**`PGF(Distribution)` holds all logic on generating functions**: the case analysis of each operation, the
decisions how and when to simplify, and the proof rules on closed forms (`equality`, `ordering`, `convergence`,
`roots_of_unity`, `multisection`, … in the subpackage `pgf/rules/`). This is where most of the mathematics of the
analysis goes, so it is in one place, as template methods, each with the class of inputs and the justification
in its docstring. We have no second representation and none is planned; the split is cheap (it is mostly
moving code that is already separate in substance) and it makes the name honest: a reader sees which code
depends on generating functions. `PGF` is built from layers, one module each: `pgf/hooks.py` (the abstract
engine hooks), `pgf/base.py` (names, comparisons, marginals, convergence proofs), `pgf/filtering.py`,
`pgf/updating.py` and `pgf/pgf.py`. Engines: `SympyPGF` (`pgf/sympy/`) and `GinacPGF` (`pgf/ginac/`); each is
its own factory (decided: the factories are class- and staticmethods, `SympyPGF.geometric(…)`), so there are
no separate factory classes. `GeneratingFunction`, `FPS` and `ProdigyPGF` stay one release as deprecated
aliases of the engine classes, and the old module paths (`distribution.generating_function`,
`distribution.equality`, …) stay as re-export modules without logic until the last PR of v0.7.0 (P4a, #511).

**Case analysis and engine hooks** (rule for #488, review §2.1). Case analysis lives in `PGF`: which operand of
`x // n`, `x % n`, `x ^ n` is fixed, support splits, their order, zero mass, normalising `const op var` in
`filter`, the table of sampling PGFs, and what ends with "not implemented". An engine hook is a CAS operation
on an already planned case (substitute, expand to an order, compose i.i.d. copies, compute on a finite part,
simplify, compare to zero). A hook does not branch on the program, does not choose between cases and does not
raise "not implemented" by itself: it returns a result or says the CAS cannot do it, and `PGF` decides.

**Closed forms are SymPy expressions.** The AST is the syntax of programs; closed forms need algebra (normal
forms, cancellation, series, roots), and building that on the AST would mean writing a CAS. There is exactly one
translation AST expression → SymPy and spelling back goes through `spell`/`display`.

**The kernel** (`distribution/kernel/`, #417) is FLINT through `python-flint`: exact arithmetic on rational
functions over ℚ (`RationalGF`). It is not an engine and has no case analysis; `PGF` calls it as a hook when a
form is rational, and `coefficient_stream` already uses it for the coefficients of result tables. It cannot do
`exp`, `log` or `sqrt` symbolically; those stay with SymPy (or GiNaC). Measured on the operations
(`docs/source/kernel.md`) it is one to two orders of magnitude faster than GiNaC's engine and two to four than
SymPy. **Decision point** (#417 step 4): after the kernel is a hook of `PGF`, measure how much of the example
library is not rational and how fast SymPy plus kernel is there. If that suffices, GiNaC becomes optional and
frozen and the bindings repository can be retired later; otherwise GiNaC stays and the kernel serves the
rational paths only. Until then no feature is built on the kernel that GiNaC cannot also serve. The goal is two
arithmetics (SymPy for general closed forms, FLINT for the rational hot path) instead of today's four (SymPy,
GiNaC, the kernel, and the `Fraction` dictionaries of `product_form`/`egcalc`).

**The simulator** produces estimates with confidence intervals by running paths, not a distribution; `filter`
and `update` have no meaning there. What it shares with `Distribution` is the reading side (probability,
expectation, marginal), which the fuzzer compares. If that comparison grows, this reading side becomes a small
protocol; nothing more is abstracted.

## Instruction handlers

One handler per construct is kept: each handler is one rule of the forward semantics, so the code follows
`docs/source/semantics.md`. What changes is how they are called: an explicit `AnalysisContext` instead of six
positional arguments and the hidden session, the state as the pair (distribution, error mass), and a dispatch
table from instruction type to handler instead of an `isinstance` chain. The while handler only hands the loop
to the registry; the strategies have left `while_handler.py` (T13a: from 1 294 lines to 30).

## Loops: goals for the user, proof rules in the code

The user asks for a **goal**: an exact result, a lower bound, an upper bound, or both. The code is organised by
**proof rule**, because one rule serves several goals: Park induction gives upper bounds and checks exact
invariants, lower induction gives lower bounds, unrolling gives lower bounds and exact results when no mass is left
inside the loop or an iteration reaches a fixed point. The registry connects the two: each strategy declares which `Bound`s it can deliver and under which
conditions, and a goal is served by trying the applicable rules from the cheapest to the strongest.

```text
analysis/loops/
    api.py              LoopDecision, ParamSpec, LoopTask, LoopOutcome (the vocabulary layer)
    registry.py         registration, the strategies' goals, the order in which they are tried
    unrolling.py        fixed and threshold unrolling (acceleration, Newton iteration: later, here)
    invariants/exact/   EVT invariants: checking given ones, synthesis (today evtinvariants/)
    invariants/bounds/  sub- and superinvariants by LP, sandwich bounds (today lowerbounds/)
    witness.py          witnesses that refute "Φ(I) = I" or "Φ(I) ≤ I"
```

`lowerbounds` is misnamed: it certifies sandwich bounds (an EVT sub-invariant from below, a Park
superinvariant from above, synthesised by linear programming over eventually geometric templates, with exact
rational certificates); unrolling is only its validation baseline (`explicit.py`). It has two structural
problems, solved in three steps:

1. **No own IR.** The tuple IR (`ir.py`) exists only because the calculus covers a fragment of the language. It
   is replaced by the typed AST plus a fragment check that names the unsupported construct, and affine guards
   come from `linear_form` in `distribution/pgf/rules/linear_guards.py` (review §2.6). mypy then checks this code. (v0.7.0, P5b.)
2. **No own arithmetic.** `product_form`'s polynomials (dictionaries of `Fraction`) become kernel polynomials;
   only what is specific stays: eventually geometric coefficients, the exact positivity check, masses, moments.
   (#417, after the kernel is a hook of `PGF`.)
3. **No second semantics.** `Φ` on a basis element `b / D` is `update`/`filter`/choice on a rational generating
   function, which the instruction handlers on a kernel-backed `PGF` compute as well; the numerator is read off
   as `D · Φ(b / D)`, a polynomial for the supported fragment (checked, not assumed). Measure first: the LP calls
   `Φ` for many basis elements; if the general path is much slower, `egcalc` stays as a thin specialisation on
   kernel polynomials. (#417.)

**Equivalence** is two things today: the compare mode for two programs, and the check of an invariant written as
a loop-free program (`Φ(I) = I`, superinvariants `Φ(I) ≤ I`), which the while handler reaches by importing the
compare mode. The invariant check belongs to `loops/invariants`; the witness search both need moves to
`loops/witness.py`; `equivalence/` keeps the compare mode only.

## Other analysis packages

- **independence** (a master's thesis): sound syntactic under-approximation of pairwise stochastic
  independence; used only by `prodigy independent_vars`, not by the analysis. It stays as it is and is
  documented. Using it to enable operations (e.g. `E[x·y]` as a product for independent `x`, `y`) is a feature
  whose soundness must be shown first (follow-up).
- **optimization**: the `optimize(expr, x, p)` statement and a 62-line `GFOptimizer`, SymPy only, one test. It
  is marked experimental. Parameter fitting such as maximum likelihood is a feature (follow-up).
- **plotting** is a front end: the analysis returns plot data (the `plot` statement stays in the language and
  its data goes into the result), the GUI draws it, and the CLI renders it with matplotlib. A batch mode
  `prodigy plot` for figures of many programs is a feature (follow-up).

## Current state and target

Measured on `main` at `0bba880` with an `ast` walk over `prodigy` (all imports, also inside functions) and
`pylint --enable=cyclic-import`. Work packages are those of the review in #221 (§13) plus **P5a**.

| # | Current state | Target | Removed by |
|---|---|---|---|
| T1 | Removed: a cycle of 17 modules (`session`, `result`, `trace`, `upper_track`, `nested_inference`, `analyzer`, all instruction handlers, `loop_strategies`, `equivalence.verdict`, `evt_exploration`, `series_identity`, `web.diagnostics`) through handlers importing `session`, the registry importing `while_handler` and `result` running the analysis. Now the vocabulary layer (`Bound` and the strategy API in `loops/api.py`, values in `values.py`, `AnalysisContext` in `context.py`) imports nothing above it; handlers reached the run with `current_context()` (since T11 through their `Frame`); `loops/builtin.py` registers the built-in strategies; the session runs the analysis and `result.py` only assembles | no module cycle (ast walk and pylint) | P5a (#497) |
| T2 | Removed: `analysis.result` imported `web.diagnostics`; the structured diagnostics are in `language/diagnostics.py`, the first failure of a program in `analysis/checks.py` | structured diagnostics in the language layer | P5a (#497) |
| T3 | Removed: the type checker imported a proof rule (`distribution.finite_measure`) to refute sampling PGFs; the refutation is `analysis.checks.refute_sampling_pgfs`, run after the type check on every entry path (CLI, session, GUI) | the PGF refutation is a check of the analysis, run before the session | P5a (#497) |
| T4 | Removed: no cycle `lowerbounds.problem` ↔ `lowerbounds.subinvariant` was left (neither the ast walk nor pylint finds one); the private `_contains` shared by both is the public `contains` of `lowerbounds/ir.py` | shared types in one module | P5a (#497) |
| T5 | Removed: `while_handler` used `SympyPGF.zero()`; it uses `config.factory.zero()` (the zero test of the unrolling is still never true for programs with variables, see the PR) | `factory.zero()` / `is_zero_dist()` | P5a (#497) |
| T6 | Removed: `result.sym` used the engine's private `_parse_to_sympy`, and `str()` of `Expr`, `Instr` and `Program` was undocumented pGCL-like text that the engines parsed back. Now `str()` of every node is `.ppy` (a printer installed from outside the AST); expressions reach the engines only as documented engine text (`language/engine_text.py`, `language-design.md`), read by the one parser `distribution/engine_text.py::parse_engine_text`; loop queries and loops are looked up by structure, not by text | one engine text format, one printer and one parser | P5a (#497, #479) |
| T7 | Removed: `query_handler` imported `plotting` (matplotlib in the analysis) to draw in the interactive CLI. Now a `plot` statement always records its data in the result; the GUI draws it and the CLI renders it after the run with `prodigy.plotting` (data that cannot be drawn, such as the joint plot of an infinite support, is reported instead of a figure) | plot data in the result, rendering in the front ends | P7 (#514) |
| T8 | Removed: front ends imported 18 analysis modules outside the front-end API, and `web/app.py` imported `prodigy_ginac` to see whether GiNaC is installed. Now what the CLI's commands and the GUI's endpoints need beyond the session comes from `analysis/tools.py` (re-exports with `__all__`, no logic), and the GUI asks `importlib.util.find_spec` | the front-end API, exported with `__all__` | P3, P7 (#514) |
| T9 | Removed: the case analyses of the updates and of filters on constants were duplicated in both engines, in different orders. Now `iid` and the factories use one table `pgf/sampling.py` and the hook `_compose_iid` (P4b); `*`, `/`, `%`, `**` and floor division one case analysis `PGF._binary_by_cases` with the hooks `_update_scaled` and `_termwise` (P4c); comparisons with constants one normalisation `PGFFiltering._filter_constant` with the hook `_filter_relation` (`=`, `<=` and a natural bound), and the case split of probabilities over finite supports lives in `PGFFiltering._filter_by_cases` (P4d) | case analysis in `PGF`, CAS hooks per engine | P4b–d (#488) |
| T10 | Removed: `distribution.py` (1 943 lines) mixed the interface, the shared GF logic and value types, and the proof rules lay flat beside the engines. Now the interface is `distribution.py`, the GF logic the layers of `pgf/`, the rules `pgf/rules/`, the engines `pgf/sympy/` and `pgf/ginac/` | `Distribution` interface, `PGF` with `pgf/rules/`, engines `SympyPGF`/`GinacPGF` | P4a (#511) |
| T11 | Instruction handlers share a 6-argument positional signature (48 of pylint's `too-many-*arguments`) and fetch the session globally | `AnalysisContext`, dispatch table | P5b |
| T12 | `session.py` holds decision providers, records and the session (the regular expressions over the source are gone: the outline with `LoopInfo` and positions from the parser is `language/outline.py`; `LoopDecision` and `DecisionRequired` are in `loops/api.py`) | session module(s) below 800 lines | P5a (outline, API, done), P5b (split) |
| T13 | Loop proof rules spread over `while_handler.py` (registry + all strategies), `evtinvariants/`, `lowerbounds/`; the invariant check reaches into the compare mode | `analysis/loops/` as above; strategies declare their goals | P5b |
| T14 | `lowerbounds` has its own tuple IR and, with `product_form`, its own `Fraction` arithmetic and statement semantics | typed AST with a fragment check (P5b); kernel polynomials and the shared semantics (#417, measured) | P5b, #417 |
| T15 | `syntax` threaded through 30 files although only `.ppy` exists | rejected once at the boundary | P2 |
| T16 | `Dict[str, Any]` for result, trace and requests; two `JsonValue` aliases | one JSON alias, `TypedDict`s | P9 (#379) |
| T17 | `tests/test_layering.py` checked a package `prodigy.ppy` that does not exist, so the language layer was unchecked | checked (fixed with #493) | #493 |

### Order of the milestone

1. **#493** (this page and the layering tests).
2. **P1** dead code; **#446** rename (alone, ~195 files); **P2** `syntax` removal and watch validation;
   **P3** typing basics (`__all__` makes the interfaces explicit). Serial: they touch every package.
3. **P5a layer cuts** (T1–T6, the outline and strategy API of T12): small in lines but crossing the analysis.
   It comes before the package PRs so that they split modules along the target, not along today's cycles.
4. In parallel, no shared files: **P4a–d** (`distribution`: first the `Distribution`/`PGF` split T10, then the
   three case-analysis PRs T9), **P5b** (`analysis`: handlers T11, `loops/` T13, the IR of T14, the module splits
   of T12, review A2–A12), **P6** (language checker), **P7** (web, CLI, plotting: T7, T8), **P8** (tests).
5. **P9** JSON types after P5b and P7 (same files).
6. Last PR: switch the structure checks of `pylintrc` on, with justified local exceptions only.

#417 (kernel as a hook, then T14 steps 2–3 and the GiNaC decision) is not part of v0.7.0; it starts after P4,
whose `PGF` hooks it plugs into.

Each package PR leaves its own package clean for the checks it is responsible for (measured with
`pylint --disable=all --enable=<check> prodigy/<package>`), so that the last PR only flips the switches.

| Check (disabled in `pylintrc`) | Hits today | Made clean by |
|---|---|---|
| `cyclic-import` | 0 (19 before P5a, all inside T1) | P5a (#497) |
| `no-member` | 2 | P3 |
| `duplicate-code` | 5 blocks in `prodigy` | P4 (engines; `rational_parts`: done), P2 (watch validation), P5b (three-valued `all`), P6 (name walk) |
| `too-many-arguments`, `too-many-positional-arguments` | 75, 72 | P5b (handler context, loops), P7 (CLI), rest in P4/P6 |
| `too-many-branches`, `-return-statements`, `-statements`, `-nested-blocks` | 33, 50, 8, 1 | P4 (`distribution`: clean, with justified local exceptions for the proof rules), P5b (`lowerbounds`, `evtinvariants`), P6 (`check.py`: done, now the `check/` package) |
| `too-many-lines` (with `max-module-lines = 800`), `too-many-locals`, `-instance-attributes`, `-public-methods` | 6 modules over 800 lines; 94, 20, 2 | P4c (`pgf/sympy/engine.py`: done), P5b (`while_handler`, `session`, `nested_inference`, `result`, `trace`), P6 (`check.py`: done, `ppy_parser.py`), P7 (`cli.py`, `web/requests.py`: done with #514, now the `cli/` package and `web/compare.py`) |

### Follow-ups (features, not v0.7.0)

Goal-first loop choice in GUI and CLI (the registry of P5b makes it possible); acceleration and Newton iteration
as loop rules; a CLI batch mode `prodigy plot`; using proven independence to enable operations (soundness
first); parameter fitting with `optimize`. (The frontend's large ES modules were split in P7, review W5:
`main.js` by concern and mode, `analysis-card.js` by lens, `joint-view.js` into cells and 3D bars.)
