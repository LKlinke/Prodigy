# Prodigy — notes for Claude

Prodigy analyses probabilistic programs (`.ppy`; `.pgcl` was removed in v0.6.0) with probability
generating functions. Results are exact or a proven lower/upper bound; otherwise `unknown`.
Details on the layout and releases: `docs/source/development.md`.

## Working on Prodigy
- Answer the user in German; code, docs, commits and PR texts are in English.
- One task per session, usually one issue of `LKlinke/Prodigy` (`Fixes #12` in the PR).
- Branch from an up-to-date `main`, open a PR, merge commit; the CI must be green. Never
  force-push or rewrite pushed history, never change authors. Ask before pushing unless the user
  said so in the session.
- The repository is public: no unpublished research (titles, venues, results) anywhere.
- Fix small findings in place: a problem in the modules the PR already touches, fixable in well
  under an hour without changing behaviour outside the PR's topic, is fixed in the same PR (its
  own commit, listed under "Also fixed"). A regression introduced by the PR is always fixed in it.
- PR texts end with "Found, not fixed here": every larger or out-of-scope problem, and every
  deviation from these rules, says whether it exists on `main`. Each item becomes an issue
  (priority and `area:`/`topic:` labels) or an item of an open collection issue.
- Before a PR counts as done, check every point of the issue's "Done when" (and items folded into
  it from other issues); name any that is left out. A removal or rename searches code, tests,
  docs, messages, the frontend and vendored bundles for leftovers and states the search in the PR.
- Milestones are releases (`v0.5.0`, …); the backlog has none. Themes are labels. Releases come
  from a tag `v*` (no version to bump); see `development.md`.

## Commands
```sh
uv sync                              # exactly what uv.lock says
uv run prodigy --engine ginac web    # GUI on http://localhost:8080 (restart after code changes)
make lint                            # pylint, must stay 10.00/10
make mypy                            # no errors
make test-fast                       # tests without the `slow` marker (> ~10 s each)
make fuzz                            # fuzzing and simulator comparison, about 30 s
make docs                            # Sphinx, warnings are errors
uv run pytest tests/analysis/test_session.py -q   # one file
```
- During work run `make lint`, `make mypy`, `make fuzz` and the tests that concern the change.
  The full `make test` is for the CI (all tests on 3.14, the fast ones on 3.11, slow ones weekly);
  a change passes only when the CI of its PR is green.
- No command blocks for more than about 2 minutes: give every program run a limit
  (`timeout 120 …`; pytest stops a test after 120 s), run longer things in the background, never
  wait for the CI in a loop — after pushing, end the turn. Anything slower than expected is
  stopped and investigated; it is often a bug.
- Dependencies: change `pyproject.toml`, run `uv lock`, commit both. After a new bindings release
  use `uv lock --refresh-package prodigy-ppl-ginac`.
- GiNaC comes from `prodigy-ppl-ginac` (wheels, imported lazily). GiNaC bugs are fixed in
  `LKlinke/prodigy-ginac-bindings`; Prodigy only adds guards at its interface.

## Layout
- `prodigy/language` (`prodigy.pgcl` until v0.7.0, #446): the `.ppy` parser and printer
  (`ppy_parser.py`, `ppy_printer.py`), type checker, AST.
- `prodigy/distribution`: the `Distribution` interface, the engines (SymPy default, GiNaC) and
  the proof rules on closed forms.
  `prodigy/analysis`: the forward analysis. `prodigy/web`: the GUI (Flask, one process per job).
- Layering: `prodigy/distribution` and `prodigy/language` import nothing from `prodigy/analysis`
  (`tests/test_layering.py`).
- Loop strategies are registered in `analysis/instructionhandler/while_handler.py`; they appear
  in the CLI, `prodigy analyze --loop` and the GUI automatically.
- GUI frontend: plain ES modules, no build step; every formula goes through
  `components/formula-box.js`, every animation through `components/motion.js`.
- `pgfexamples/` is the GUI's library and a test input; results write expressions in the
  program's syntax (`session.spell`).

## Principles
- Soundness first: a result is reported as exact or as a lower/upper bound only if a check proves
  it; otherwise it is `unknown`. Heuristics and language models only propose candidates; the
  exact checks decide. Verdicts are three-valued: "refuted" needs a witness, otherwise undecided.
- Rules for classes, never for single programs: a proof rule, heuristic or special case covers a
  well-founded class of problems with its justification in the docstring; examples are only test
  cases, and tests cover generic members of the class.
- Every program with a well-defined semantics is accepted by the parser and the type checker.
  What the analysis cannot compute ends with "not implemented" naming the construct, never with
  a wrong or approximate result (`docs/source/language-design.md`).
- Untrusted text (GUI input, model answers, CLI expressions) is validated with the parser or an
  allowlist before SymPy sees it; `sympify`/`parse_expr` evaluate Python code
  (`ai_candidates.candidate_from_answer`, `ParamSpec.parse`, `cli._expression_option`).
- Engines share their mathematics: case analysis (support splits, their order, zero mass, what
  is computed or "not implemented") lives in the shared `Distribution` logic; CAS operations are
  small engine hooks. Duplicated logic is extracted, not copied into another engine.
- Tests use no network (fake external services, see `tests/analysis/test_ai_candidates.py`);
  `tests/conftest.py` seeds random draws and restores the global flags after each test.
- GUI follows Apple's HIG: role-based colour tokens, contrast at least 4.5:1, dark mode follows
  the system, status never by colour alone, everything works from the keyboard. Use the skill
  `apple-design` for every GUI change and review it against its checklist; verify in the built-in
  browser in light and dark mode at 1400, 700 and 375 px.

## Known pitfalls
- `use_simplification`/`use_latex` of a `ForwardAnalysisConfig` hold only inside
  `config.activate()` (`compute_discrete_distribution`, `AnalysisSession.run` and the CLI enter
  it); code outside gets the defaults, and `tests/conftest.py` fails a test whose settings leak.
- `AnalysisSession(capture_output=True)` also captures native writes to file descriptor 1, so
  that `prodigy analyze --json` stays clean.
- Use `git --no-pager`; a pager hangs the shell.
- Tests run in parallel workers (`-n auto --dist worksteal`): output is not in file order, tests
  must not depend on order or one process (fixed seeds for parameters drawn at collection, files
  in `tmp_path`); a hanging worker's stack is dumped after 10 minutes and names the test.
