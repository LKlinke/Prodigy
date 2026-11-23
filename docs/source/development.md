# Development

Notes for contributors on the code layout, dependencies and releases. The working rules for
changes (branches, pull requests, tests, principles) are in `CLAUDE.md` at the repository root.

## Layout

### `prodigy/pgcl` — the language

Parsers, AST, type checker and compiler. The package will be renamed to `prodigy/language` after
the `.pgcl` syntax is removed (#433, #446).

- `ppy_parser.py`, `ppy_printer.py`: the `.ppy` syntax (the default, `language.md`).
- `parser.py`: the removed `.pgcl` syntax (`grammar.rst`), left only for tests until its removal
  (#433). Its lark grammar is taken over from probably together with its history (`prodigy/pgcl/NOTICE`).
- `check.py`: the type checker; `pgcl_checks.py`, `pgcl_operations.py`: further checks and AST
  operations.
- `scope.py`: block scope, lowered to assignments and marginalisation.
- `positions.py`: source positions for messages and the editor.

### `prodigy/distribution` — distributions and engines

- `distribution.py`: the `Distribution` interface and the shared logic (case splits, filters,
  marginals, the iteration contract).
- Engines: SymPy (`generating_function.py`, the default), GiNaC via the bindings
  (`fast_generating_function.py`). SymEngine was removed in v0.6.0 (#416).
- Proof rules on closed forms: `finite_measure.py`, `product_form.py`, `coefficient_signs.py`,
  `series_transforms.py` (monomial substitution, shift), `dominant_pole.py` (univariate rational
  functions with a unique dominant simple pole, Arb bounds), `convergence.py` (convergence at 1),
  `ordering.py` (`<=` on distributions).
- `prodigy/distribution` and `prodigy/pgcl` import nothing from `prodigy/analysis`
  (`tests/test_layering.py`).

### `prodigy/analysis` — the forward analysis

- `analyzer.py` dispatches instructions to `instructionhandler/`.
- `session.py`: `AnalysisSession`, one decision per loop.
- `loop_strategies.py`: the strategy registry; strategies are registered in
  `instructionhandler/while_handler.py` and appear in the CLI menu, `prodigy analyze --loop` and
  the GUI automatically.
- `result.py`: the JSON result `prodigy/analysis-result@1`.
- `upper_track.py`: a loop's proven upper bound (the EVT sandwich's super-invariant), carried through
  the rest of the program; it tightens the upper ends of coefficients and queries (#451).
- `equivalence/`: the equivalence check of two programs and its three-valued verdict.
- `evtinvariants/`: EVT invariants, their synthesis and the positivity heuristics.
- `lowerbounds/`: occupation sub-invariants and the sandwich (`evt_subinvariant`, `evt_sandwich`).
- `ai_candidates.py`: invariant candidates from a language model (`ai_invariant`).
- `nested_inference.py`: `dist` values, `infer`, `sample` (`language-design.md`).
- `trace.py`: the step trace (`prodigy analyze --steps`, the GUI's analysis timeline).

### `prodigy/web` — the GUI

A Flask app (`gui.md`). Each analysis runs in its own process (`jobs.py`); analyses and checks
share `--max-jobs`, background jobs (joint grids, suggestions) have `--max-background-jobs`. The
frontend is plain ES modules without a build step. CodeMirror and KaTeX are vendored under
`static/vendor` and rebuilt via `prodigy/web/frontend`. Shared components are in
`static/app/components`: every KaTeX formula goes through `formula-box.js` (bounded size), every
animation through `motion.js`.

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
