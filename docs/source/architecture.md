# Architecture

This page states the **target** structure of Prodigy that the maintenance pass of v0.7.0 (#221, #493) works
towards, and how far the code is from it today. The dependency rules are checked by `tests/test_layering.py`;
every rule that the code still breaks is an expected failure there, naming the work package that removes it.
Package paths are the current ones; `prodigy/pgcl` becomes `prodigy/language` with #446.

## Layers

```text
 front ends     prodigy.cli            prodigy.web (Flask + ES modules)        prodigy.plotting
                     |                          |                                    |
                     +-------- front-end API of the analysis (session, result) -----+
                                                |
 analysis       session / orchestration   prodigy.analysis.session, trace, upper_track, nested_inference driver
                     |
                semantics                 analyzer, instructionhandler/*, loop strategies (registry + groups),
                     |                    evtinvariants, lowerbounds, equivalence, independence, optimization
                     |
                analysis core             config (engine choice), current-session context, value types
                     |                    (LoopDecision, LoopInfo, Bound), exceptions
 ------------------- | ---------------------------------------------------------------------------------------
 distribution   prodigy.distribution      Distribution (shared case analysis), proof rules on closed forms,
                     |                    display values; engines as hooks: SymPy, GiNaC, FLINT kernel
 language       prodigy.pgcl              parser, AST, type checker, language warnings, printer (spelling),
                     |                    structured diagnostics (line, column, message)
 util           prodigy.util              logging, small helpers (no Prodigy imports)

 simulator      prodigy.simulator         sampling semantics beside the analysis; imports only the language
```

Every arrow points down: a layer imports only layers below it, never above, also not lazily inside a function
or under `TYPE_CHECKING`. Inside the analysis the three sub-layers follow the same rule. There are no import
cycles, neither at module level nor through function-local imports (pylint's `cyclic-import` counts both).

| Layer | Owns | Does not own |
|---|---|---|
| util | helpers without Prodigy knowledge | anything about programs or distributions |
| language | the `.ppy` text: parsing, the AST, types, warnings, spelling an AST in program syntax, positions and structured diagnostics | any CAS; proof rules; GUI wording |
| distribution | the meaning of a program state as a generating function: `Distribution` with its case analysis, the proof rules (`equality`, `finite_measure`, `convergence`, …), JSON display values | which engine is used; loops, strategies, results |
| analysis core | `ForwardAnalysisConfig` (and with it the engine choice), the ambient session context, value types shared by handlers and strategies | running anything |
| analysis semantics | the forward analysis of each instruction, loop strategies and the proofs behind them (invariants, lower bounds, nested inference, equivalence) | JSON shapes, text output, the GUI |
| session | running an analysis on a program text, the trace, assembling the result `prodigy/analysis-result@1` | presentation |
| front ends | user interaction: options, files, HTTP, jobs, text and graphical output | analysis internals beyond the front-end API |

## Interfaces between the layers

**Types that cross.** Language → everything: `Program`, `Expr`, `Instr` (the AST) and `Type`. Distribution →
analysis: `Distribution` and `CommonDistributionsFactory` only, plus the proof-rule functions with their
three-valued verdicts (`True`/`False` with a witness/`None`). Analysis → front ends: `AnalysisSession`,
`ForwardAnalysisConfig` with its option enums, the strategy registry (`loop_strategies`), the result object and
its JSON (`result.to_json()`, typed with #379/P9), the trace, the equivalence verdict, and the exceptions. A
front end needs nothing else; what it needs today beyond that (CLI: `lowerbounds.*`, `evtinvariants.*`,
`program_info`) is a missing entry in that API.

**Where text is parsed.** Program text is parsed once, by the language layer (`parse_pgcl` →
`parse_program` with #446), at the boundary where it enters: the front end or `AnalysisSession`. Expressions
from users or language models (watches, invariants, parameters, candidates) are parsed by the language parser
or an allowlist before SymPy sees them; inside the analysis no one parses text again (today watches are parsed
three times, review §2.3). Numbers and closed forms from the engines are turned into SymPy by one public
function of the distribution layer, not by a private engine helper (`result.sym` uses
`generating_function._parse_to_sympy` today; engine-neutral text is #479).

**Where text is spelled.** An AST is spelled by the language printer (`ppy_printer.spell`);
`session.spell` only adds the session's program. A distribution is spelled for people by `distribution.display`
(and the frontend's `formula-box.js`). Diagnostics for the GUI and the CLI are built from the language's
structured diagnostics; the web layer only adds its wording and JSON envelope.

**Where the engine is chosen.** Once, in the analysis core: `ForwardAnalysisConfig.factory` maps the
`Engine` option to a `CommonDistributionsFactory`. Everything above uses that factory and the `Distribution`
interface; no module of the analysis or a front end imports an engine module (`generating_function`,
`fast_generating_function`, `kernel`) or `prodigy_ginac`. Availability checks of an engine are a function of the
analysis core, not an import in the front end.

## Case analysis and engine hooks (rule for #488, review §2.1)

The mathematics is shared; only CAS operations differ between engines.

- **Case analysis lives in `Distribution`**: which operand of `x // n`, `x % n`, `x ^ n` is fixed, splitting
  the support into a finite and an infinite part, the order of these cases, zero mass, normalising
  `const op var` to `var op' const` in `filter`, the table of sampling PGFs, and the decision what ends with
  "not implemented". These are template methods; their docstring states the class of inputs and why each case is
  sound.
- **An engine hook is a CAS operation** on an already planned case: substitute, expand a series to a given
  order, compose a PGF with `count` i.i.d. copies, compute on a finite part, simplify, compare to zero. A hook
  does not branch on the shape of the program, does not choose between cases, and does not raise
  "not implemented" on its own: it returns a result or signals "the CAS cannot do this", and the shared logic
  decides.
- A case split found in two engines is moved to `Distribution`, not fixed twice. The FLINT kernel (#417) plugs
  in as a third set of hooks.

## Current state and target

Measured on `main` at `0bba880` with an `ast` walk over `prodigy` (all imports, also inside functions) and
`pylint --enable=cyclic-import`. Work packages are those of the review in #221 (§13).

| # | Current state | Target | Removed by |
|---|---|---|---|
| T1 | A cycle of 17 modules (imports at run time, also inside functions): `session`, `result`, `trace`, `upper_track`, `nested_inference`, `analyzer`, all instruction handlers, `loop_strategies`, `equivalence.verdict`, `evt_exploration`, `series_identity`, and `web.diagnostics`. Handlers import `session` for `current_session` and the value types; `loop_strategies` imports `while_handler` to register the built-ins; `result` runs the analysis (`compute_discrete_distribution`) and nested inference | analysis core module for the session context and value types; registry without handler imports (built-ins registered by the package that owns them); the result is assembled from data, it does not run analysis | **P5a** (new: layer cuts) |
| T2 | `analysis.result` imports `web.diagnostics` (type errors and language warnings with positions) | structured diagnostics in the language layer; `web.diagnostics` keeps wording and JSON | P5a |
| T3 | The type checker (`pgcl/check.py`) imports a proof rule (`distribution.finite_measure`) to refute sampling PGFs (a function-local import: no cycle between modules, but the two packages depend on each other) | language layer without CAS or proof rules; the PGF refutation becomes a check of the analysis run before the session (the sample handler already checks every executed sampling) | P5a |
| T4 | A second cycle `lowerbounds.problem` ↔ `lowerbounds.subinvariant` | shared types in one module | P5a |
| T5 | `while_handler` uses `SympyPGF.zero()` (engine knowledge in the semantics) | `factory.zero()` / `Distribution.is_zero()` | P5a |
| T6 | `result.sym` uses the engine's private `_parse_to_sympy` | public parser of engine text in `distribution` | #479 (P5a if not done before) |
| T7 | `query_handler` imports `plotting` (matplotlib in the analysis) | plotting is a front end; the analysis returns data | P7 |
| T8 | Front ends import 18 analysis modules outside the front-end API (CLI: 15 modules, among them `analyzer`, `lowerbounds.*`, `evtinvariants.*`, `program_info`, `independence`; web: `analyzer`, `nested_inference`, `ai_candidates`, `joint_grid`, `input_distribution`, the option enums of the heuristics and solvers; `equivalence.verdict` instead of the package); `web/app.py` imports `prodigy_ginac` to test availability | the front-end API above, exported with `__all__` (P3); internals behind it | P3 (`__all__`), P7 |
| T9 | Case analysis of `_update_iid`, `_update_division`/`_modulo`/`_power` and `_filter_constant_condition` duplicated in both engines, in different orders | shared planning in `Distribution`, CAS hooks per engine | P4a–c (#488) |
| T10 | `syntax` threaded through 30 files although only `.ppy` exists | rejected once at the boundary | P2 |
| T11 | All instruction handlers share a positional 6-argument signature (`instruction, prog_info, distribution, error_prob, config, analyzer`; 48 of pylint's `too-many-*arguments`) | one handler context (program info, config, analyzer) beside the instruction and the two distributions | P5b |
| T12 | `Dict[str, Any]` for the result, trace and requests; two `JsonValue` aliases | one JSON alias, `TypedDict`s for the documented shapes | P9 (#379) |
| T13 | `tests/test_layering.py` checked a package `prodigy.ppy` that does not exist, so the language layer was not checked | the language is checked (fixed in this PR) | #493 |

### Order of the milestone

1. **#493** (this page and the layering tests).
2. **P1** dead code; **#446** rename (alone, touches ~195 files); **P2** `syntax` removal and watch
   validation; **P3** typing basics (`__all__` makes the front-end API explicit). Serial: they touch every package.
3. **P5a layer cuts** (T1–T6): the structural change of the milestone, small in lines but crossing the
   analysis; it comes before the package PRs so that they split modules along the target, not along today's
   cycles. Re-enables `cyclic-import`; the layering xfails turn into passes.
4. In parallel, no shared files: **P4a–c** (`distribution`, #488), **P5b** (`analysis`: strategy modules,
   handler context, review A2–A12), **P6** (language checker), **P7** (web, CLI, plotting; T7, T8),
   **P8** (tests).
5. **P9** JSON types after P5b and P7 (same files).
6. Last PR: switch the remaining structure checks of `pylintrc` on with justified local exceptions only.

Each package PR leaves its own package clean for the checks it is responsible for (measured with
`pylint --disable=all --enable=<check> prodigy/<package>`), so that the last PR only flips the switches.

| Check (disabled in `pylintrc`) | Hits today | Re-enabled by |
|---|---|---|
| `cyclic-import` | 19 cycles reported by pylint, all inside T1 and T4 | P5a |
| `no-member` | 2 | P3 |
| `duplicate-code` | 5 blocks in `prodigy` | P4 (engines), P2 (watch validation), P5b (three-valued `all`), P6 (name walk) |
| `too-many-arguments`, `too-many-positional-arguments` | 75, 72 | P5b (handler context, lower bounds), P7 (CLI), rest in P4/P6 |
| `too-many-branches`, `-return-statements`, `-statements`, `-nested-blocks` | 33, 50, 8, 1 | P4 (`distribution`: 11, 19), P5b (`lowerbounds`, `evtinvariants`), P6 (`check.py`) |
| `too-many-locals`, `-instance-attributes`, `-lines`, `-public-methods` | 94, 20, 7, 2 | the package PRs (module splits of P5b, D2 of P4); `session`/`nested_inference` splits in P5b |
