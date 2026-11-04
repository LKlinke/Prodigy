# Prodigy — notes for Claude

Prodigy analyses probabilistic programs (pGCL) with probability generating functions (PGFs).
It computes output distributions exactly where it can and otherwise gives results with a stated
guarantee: exact, lower bound with residual mass, or upper bound. Maintainer: Lutz Klinkenberg.

## Working with Lutz
- Lutz writes in German; answer in German. Code, comments, docs, commit messages and PR texts are
  in English.
- One task per session, usually one GitHub issue. Open tasks are GitHub issues of
  `LKlinke/Prodigy`; refer to them in PRs (`Fixes #12`).
- Work on a branch from an up-to-date `main` and open a pull request; `main` only changes through
  PRs, and the CI must pass. PRs are merged with a merge commit, not squashed.
- Never force-push `main`, never rewrite history that is already pushed, and never change the
  author or committer of commits.
- Ask before pushing or opening a PR if Lutz has not asked for it in the session.
- Unpublished research stays out of the repository: no paper titles, venues or unpublished results
  in code, docs, commit messages or PR texts.

## Commands (uv)
```sh
uv sync                              # install exactly what uv.lock says, into .venv
uv run prodigy --engine ginac web    # GUI on http://localhost:8080 (restart after code changes)
make lint                            # pylint, must stay at 10.00/10
make mypy                            # mypy, no errors
make test                            # pytest with coverage, about 5 minutes
make docs                            # Sphinx docs into docs/build/html, warnings are errors
uv run pytest tests/analysis/test_session.py -q   # a single file
```
The CI (`.github/workflows/main.yml`) runs pylint, mypy and the tests on Python 3.11 and 3.14 with
the locked dependencies. Check like the CI before saying something passes.

Dependencies: change `pyproject.toml`, run `uv lock`, commit both files. The GiNaC engine comes from
`prodigy-ppl-ginac` (module `prodigy_ginac`, wheels with GiNaC bundled); a marker installs it only
where wheels exist, and the extra `ginac` installs it everywhere, so the GiNaC engine is imported
lazily (`ForwardAnalysisConfig.factory`). Dependabot opens monthly
PRs in two groups; the group `linters` (pylint, astroid, mypy) needs code changes and is handled
on its own.

## Layout
- `prodigy/pgcl`: parser (lark), AST, type checker, compiler. Taken over from probably with its
  history (see `prodigy/pgcl/NOTICE`); `pgcl_checks.py` and `pgcl_operations.py` are Prodigy's own.
  These modules are not type-checked yet (see `mypy.ini`).
- `prodigy/distribution`: the `Distribution` interface and its engines: SymPy
  (`generating_function.py`, default), GiNaC via prodigy_ginac (`fast_generating_function.py`), SymEngine.
- `prodigy/analysis`: the forward analysis. `analyzer.py` dispatches instructions to
  `instructionhandler/`.
  - `session.py`: `AnalysisSession` holds one decision per loop.
  - `loop_strategies.py`: the strategy registry.
  - `result.py`: the JSON result `prodigy/analysis-result@1`.
  - Loop strategies are registered in `instructionhandler/while_handler.py`; a new strategy
    appears in the CLI menu, in `prodigy analyze --loop` and in the GUI automatically.
  - `lowerbounds/`: occupation sub-invariants and the sandwich (strategies `evt_subinvariant`,
    `evt_sandwich`).
  - `ai_candidates.py`: invariant candidates from a language model (strategy `ai_invariant`).
- `prodigy/web`: the GUI, a Flask app. Each analysis runs in its own process (`jobs.py`). The
  frontend is plain ES modules without a build step; CodeMirror and KaTeX are vendored under
  `static/vendor` (rebuilt via `prodigy/web/frontend`). Docs: `docs/gui.md`.
- `pgfexamples/`: example programs, also the GUI's library and test inputs; the wheel contains them
  as `prodigy/pgfexamples`.
- Releases: a tag `v<version>` runs `.github/workflows/release.yml` (PyPI `prodigy-ppl`, image
  `ghcr.io/lklinke/prodigy` from `Dockerfile`, GitHub release). The version comes from the latest
  tag `v*` (hatch-vcs); there is no version to bump and no relock. The Docker builds have no `.git`
  and get the version as build argument `PRODIGY_VERSION`. The trusted publisher on PyPI
  depends on the file name and the environment `release`. `Dockerfile.artifact` builds the full
  environment for artifacts (GiNaC, sources, `load_env.sh`).

## Principles
- Soundness first. A result may only be reported as exact, or as a lower or upper bound, if a
  check proves it; otherwise it is `unknown`. Candidates from heuristics or language models are
  only proposals, and Prodigy's exact checks decide.
- Untrusted input: SymPy's `sympify`/`parse_expr` evaluate Python code. Validate external text
  (GUI input, model answers) with the pGCL parser or an allowlist before SymPy sees it (see
  `ai_candidates.candidate_from_answer`).
- Tests do not use the network; fake external services (see `tests/analysis/test_ai_candidates.py`).
  `tests/conftest.py` seeds random draws per test and restores the global flags of
  `GeneratingFunction` after each test.
- GUI design follows Apple's Human Interface Guidelines:
  - role-based color tokens and a contrast of at least 4.5:1;
  - dark mode follows the system;
  - status is never shown by color alone;
  - everything works from the keyboard.

## Known pitfalls
- `use_simplification`/`use_latex` of a `ForwardAnalysisConfig` only hold inside
  `config.activate()` (a context variable); `compute_discrete_distribution`, `AnalysisSession.run`
  and the CLI enter it. Code that calls the SymPy engine outside of them (e.g. handlers directly)
  gets the defaults; `tests/conftest.py` fails a test whose settings leak.
- The GiNaC bindings print to stdout at C level; the analysis captures that output.
- pytest prints a file's line only when the next file starts. In a hanging CI run, the hanging test
  is in the file after the last complete line. `faulthandler_timeout` dumps the stacks after
  10 minutes.
