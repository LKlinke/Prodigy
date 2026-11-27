# Development

Notes for contributors on the code layout, dependencies and releases. The working rules for
changes (branches, pull requests, tests, principles) are in `CLAUDE.md` at the repository root.

## Layout

### `prodigy/language` — the language

Parsers, AST, type checker and compiler. The package was called `prodigy/pgcl` until v0.7.0 (#446);
`prodigy.pgcl` still imports, with a `DeprecationWarning`, until v0.8.0.

- `ppy_parser.py`, `ppy_printer.py`: the `.ppy` syntax (the default, `language.md`).
- `parser.py`: `parse_program`, the entry point that reads a `.ppy` program (the `.pgcl` syntax and its
  parser, taken over from probably, were removed in v0.6.0, #433; see `prodigy/language/NOTICE`).
- `check.py`: the type checker; `checks.py`, `operations.py`: further checks and AST
  operations.
- `scope.py`: block scope, lowered to assignments and marginalisation.
- `positions.py`: source positions for messages and the editor.

### `prodigy/distribution` — distributions and engines

- `distribution.py`: the `Distribution` interface (representation-neutral) and
  `CommonDistributionsFactory`; `values.py`, `errors.py`, `terms.py` hold its value types and errors.
- `pgf/`: the logic on generating functions, `PGF`, in layers (`hooks.py` the engine hooks, `base.py`,
  `filtering.py`, `updating.py`, `pgf.py`): case splits, filters, marginals, the iteration contract;
  `sampling.py` is the table of the distributions of the language (their generating functions and the
  ranges of their parameters), from which the factories and `iid` build them on every engine.
- Engines: SymPy (`pgf/sympy/engine.py`, `SympyPGF`, the default; beside it `expressions.py`, `terms.py` the
  iteration over terms, `settings.py`), GiNaC via the bindings (`pgf/ginac/engine.py`,
  `GinacPGF`); each engine class is also its factory and holds only the CAS hooks of `pgf/hooks.py`. SymEngine was removed in
  v0.6.0 (#416). The old paths `generating_function.py`, `fast_generating_function.py`, … only re-export.
- Proof rules on closed forms, in `pgf/rules/`: `finite_measure.py`, `product_form.py`, `coefficient_signs.py`,
  `series_transforms.py` (monomial substitution, shift), `dominant_pole.py` (univariate rational
  functions with a unique dominant simple pole, Arb bounds), `convergence.py` (convergence at 1),
  `ordering.py` (`<=` on distributions), `linear_guards.py` (linear forms and guards), `hadamard.py` (the Hadamard product of rational
  functions), `value_at_one.py`
  (the value of a closed form at 1), ….
- `prodigy/distribution` and `prodigy/language` import nothing from `prodigy/analysis`
  (`tests/test_layering.py`).

### `prodigy/analysis` — the forward analysis

- The vocabulary layer imports nothing above it (`architecture.md`): `loops/api.py` (`Bound`, the
  strategy API and registry, `LoopDecision`), `values.py` (values, intervals and coefficient tables
  of a result), `context.py` (`AnalysisContext`, and the `Frame` and `ProgramState` that every handler receives).
- `analyzer.py` dispatches instructions to the `handle` functions of `instructionhandler/` (`HANDLERS`).
- `session.py`: `AnalysisSession`, one decision per loop; it runs an analysis
  (`compute_discrete_distribution`, `AnalysisSession.run`). `decisions.py` holds the decision providers
  (terminal, fixed), `loop_records.py` the loops of the program and what each produced,
  `output_capture.py` the capture of the printed output (also of native code).
- `loop_strategies.py`: the strategy registry for the front ends. The registry is `loops/registry.py`
  (with `strategies_for(goal)`), the vocabulary `loops/api.py` (`LoopStrategy`, `Delivers`, …). The built-in
  strategies are defined in `loops/` (`unrolling.py`, `exploration.py`, `invariants/park.py`,
  `invariants/exact/strategies.py`, `invariants/bounds/strategies.py`, `ai.py`, `auto.py`), registered by
  `loops/builtin.py` and appear in the CLI menu, `prodigy analyze --loop` and the GUI automatically; the while
  handler only runs the chosen one.
- `checks.py`: the checks before a run (type check, refuted sampling PGFs).
- `result.py`: the JSON result `prodigy/analysis-result@1`, assembled from what the session recorded.
- `upper_track.py`: a loop's proven upper bound (the EVT sandwich's super-invariant), carried through
  the rest of the program; it tightens the upper ends of coefficients and queries (#451).
- `equivalence/`: the equivalence check of two programs and its three-valued verdict.
- `evtinvariants/`: EVT invariants, their synthesis and the positivity heuristics.
- `lowerbounds/`: occupation sub-invariants and the sandwich (`evt_subinvariant`, `evt_sandwich`).
- `ai_candidates.py`: invariant candidates from a language model (`ai_invariant`).
- `nested_inference/`: `dist` values, `infer`, `sample` (`language-design.md`): sites and their encoding,
  functions, `Pr[d …]`/`Ex[d]`, loops, the values in the result.
- `trace.py`: the step trace (`prodigy analyze --steps`, the GUI's analysis timeline).

### `prodigy/web` — the GUI

A Flask app (`gui.md`). Each analysis runs in its own process (`jobs.py`); analyses and checks
share `--max-jobs`, background jobs (joint grids, suggestions) have `--max-background-jobs`. The
frontend is plain ES modules without a build step. CodeMirror and KaTeX are vendored under
`static/vendor` and rebuilt via `prodigy/web/frontend`. Shared components are in
`static/app/components`: every KaTeX formula goes through `formula-box.js` (bounded size), every
animation through `motion.js`, and states and numbers are written by `format.js`. `static/app/main.js`
only boots the window; the app's modules share `app-state.js` (the state, the editors) and are split by
concern: `checking.js`, `panels.js` (inspector, setup panel, annotations), `compare-mode.js` (the modes
and program B), `result-panel.js` and `run.js` (jobs); no module imports `main.js`, the entry point.
The page loads scripts and styles from `/assets/<token>/…` (`assets.py`): the token changes with the
version and with any file under `static`, so an upgrade or an edit gives every module a new URL. Modules
therefore import each other with relative paths only (`tests/web/test_assets.py`).

### Examples and tests

- `pgfexamples/`: example programs in `.ppy`, also the GUI's library and test inputs; the wheel
  contains them as `prodigy/pgfexamples`.
- Tests marked `slow` (over about 10 s on the CI) run on Python 3.14 in every CI run and weekly on
  3.11; `make test-fast` skips them (`testing.md`).

## Dependencies

- Change `pyproject.toml`, run `uv lock`, commit both files.
- The GiNaC engine comes from `prodigy-ppl-ginac` (module `prodigy_ginac`, wheels with GiNaC
  bundled). A marker installs it only where wheels exist, and the extra `ginac` installs it
  everywhere, so the engine is imported lazily (`ForwardAnalysisConfig.factory`). Right after a new
  release of the bindings, `uv lock` may still see the old index:
  `uv lock --refresh-package prodigy-ppl-ginac`.
- Dependabot opens monthly PRs in two groups; the group `linters` (pylint, astroid, mypy) usually
  needs code changes and is handled on its own.

## Releases

- A tag `v<version>` runs `.github/workflows/release.yml`: PyPI `prodigy-ppl`, the image
  `ghcr.io/lklinke/prodigy` from `Dockerfile`, and a GitHub release.
- The version comes from the latest tag `v*` (hatch-vcs); there is no version to bump and no
  relock. The Docker builds have no `.git` and get the version as build argument
  `PRODIGY_VERSION`.
- The trusted publisher on PyPI depends on the workflow file name and the environment `release`.
- `Dockerfile.artifact` builds the full environment for artifacts (GiNaC, sources, `load_env.sh`).
- Milestones are releases: a release closes its milestone; open issues move to the next release
  or back to the backlog (no milestone).
