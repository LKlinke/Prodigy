# Prodigy — notes for Claude

Prodigy analyses probabilistic programs (pGCL, written in `.ppy` or `.pgcl`) with probability
generating functions (PGFs).
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
  in code, docs, commit messages or PR texts. The repository and its issues are public.
- Problems found but not fixed in a PR go into a section "Found, not fixed here" of the PR text;
  each becomes an issue (with a priority label and a milestone) instead of getting lost.

## Commands (uv)
```sh
uv sync                              # install exactly what uv.lock says, into .venv
uv run prodigy --engine ginac web    # GUI on http://localhost:8080 (restart after code changes)
make lint                            # pylint, must stay at 10.00/10
make mypy                            # mypy, no errors
make test                            # pytest in parallel (xdist) with coverage, about 2 minutes
make test XDIST=                     # the same in one process
make fuzz                            # fuzzing and simulator comparison (marker `fuzz`), about 30 s
make docs                            # Sphinx docs into docs/build/html, warnings are errors
uv run pytest tests/analysis/test_session.py -q   # a single file
```
The CI (`.github/workflows/main.yml`) runs pylint, mypy, the tests on Python 3.11 and 3.14 and a
fuzzing smoke test (`make fuzz`), with the locked dependencies; a weekly workflow runs the thorough
fuzzing (`docs/source/testing.md`).

Testing during work: run `make lint`, `make mypy`, the test files that concern the change and
`make fuzz` locally. Do not run the full `make test` locally; the CI runs it. A change counts as
passing only when the CI of its PR is green.

Waiting: no command blocks for more than about 2 minutes.
- Give every run of a program or script a limit (`timeout 120 uv run prodigy …`); pytest stops each
  test after 120 s (`timeout` in `pyproject.toml`).
- Anything that may take longer runs in the background and reports when it is done; do not set a
  long tool timeout (e.g. 900 s) to wait for it.
- Do not wait for the CI in a loop (`gh pr checks --watch`, `sleep`): after pushing, end the turn;
  Lutz or the supervising session follows up on the result.
- If something runs longer than expected, stop it and find out why (it is often a bug, like #252).

Dependencies: change `pyproject.toml`, run `uv lock`, commit both files. The GiNaC engine comes from
`prodigy-ppl-ginac` (module `prodigy_ginac`, wheels with GiNaC bundled); a marker installs it only
where wheels exist, and the extra `ginac` installs it everywhere, so the GiNaC engine is imported
lazily (`ForwardAnalysisConfig.factory`). Right after a new release of the bindings, `uv lock` may
still see the old index; use `uv lock --refresh-package prodigy-ppl-ginac`. Fixes of GiNaC problems
belong in the bindings repository (`LKlinke/prodigy-ginac-bindings`), not as patches in Prodigy;
Prodigy only adds guards at its interface. Dependabot opens monthly
PRs in two groups; the group `linters` (pylint, astroid, mypy) needs code changes and is handled
on its own.

## Layout
- `prodigy/pgcl`: parsers, AST, type checker, compiler. The `.pgcl` parser (lark) is taken over
  from probably with its history (see `prodigy/pgcl/NOTICE`). Prodigy's own modules:
  - `pgcl_checks.py`, `pgcl_operations.py`;
  - `ppy_parser.py`/`ppy_printer.py`: the `.ppy` syntax;
  - `scope.py`: block scope, lowered to assignments and marginalisation;
  - `converter.py`: `prodigy convert`;
  - `positions.py`: source positions.
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
  - `nested_inference.py`: `dist` values, `infer`, `sample` (design in `docs/source/language-design.md`).
  - `trace.py`: the step trace (`prodigy analyze --steps`, the GUI's Steps tab).
- `prodigy/web`: the GUI, a Flask app. Each analysis runs in its own process (`jobs.py`). The
  frontend is plain ES modules without a build step; CodeMirror and KaTeX are vendored under
  `static/vendor` (rebuilt via `prodigy/web/frontend`). Shared components are in
  `static/app/components`; e.g. every KaTeX formula goes through `formula-box.js` (bounded size),
  and animations through `motion.js`. Docs: `docs/gui.md`.
- `pgfexamples/`: example programs in `.ppy` (converted from `.pgcl`), also the GUI's library and
  test inputs; the wheel contains them as `prodigy/pgfexamples`. Former `.pgcl` versions of a
  representative set are in `tests/pgcl/former_examples/` with their results (keeps `.pgcl` tested).
- Language: `.ppy` is the default syntax (`docs/source/language.md`); `.pgcl` is still read
  (`docs/source/grammar.rst`). Results write expressions in the program's syntax (`session.spell`).
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
- Syntax and semantics are separate: every program with a well-defined semantics is accepted by the
  parser and the type checker. What the analysis cannot compute (yet) ends with a clear "not
  implemented" message naming the construct, never with a wrong or approximate result
  (`docs/source/language-design.md`).
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
- `AnalysisSession(capture_output=True)` captures everything the analysis writes to stdout, Python
  prints and native writes to file descriptor 1 (`_native_stdout_captured`), so that
  `prodigy analyze --json` stays clean. Since prodigy-ppl-ginac 0.0.3 the bindings no longer print
  during an analysis (bindings #14); the native capture stays as a safeguard.
- SymEngine reads `e`, `E`, `I`, `pi` and `inf` as constants. Parse strings with
  `_parse_to_symengine` (it protects `e` and `inf`), never with `se.S`. Variables named `E`, `I` or
  `pi` are rejected on SymEngine (#204).
- Use `git --no-pager` in commands; a pager waits for input and hangs the shell.
- `make test` runs the tests in parallel workers (`-n auto --dist worksteal`), so the output is not
  in file order and a hanging test does not show up as an unfinished line. `faulthandler_timeout`
  dumps the stacks of a worker whose test runs longer than 10 minutes; the dump names the test.
  Tests must not depend on order or on running in one process: parameters drawn at collection
  time need a fixed seed (every worker must collect the same test ids), files go to `tmp_path`.
