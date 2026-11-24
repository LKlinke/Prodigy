# Testing against independent oracles

Exact results are only as trustworthy as the engines that compute them. Besides the unit tests,
two kinds of tests compare the analysis with something that does not share its code.

## Simulation (`prodigy.simulator`)

{mod}`prodigy.simulator` is a Monte-Carlo interpreter for programs. It runs a parsed program many
times on concrete states with a seeded random number generator and shares only the parser with the
analysis, none of the engines. Its semantics is the one of the analysis: variables start at 0,
`observe` rejects a run (the results are conditioned on the runs that are not rejected), `abort`
loses the run, `query { x = e; observe(g); }` draws `x` again until `g` holds (local conditioning),
`-` is truncated subtraction, `/` must give a natural number when it is assigned,
`infer`/`sample` draw by rejection sampling. A `while` loop runs until a budget of iterations per run
is used up; such a run counts as unfinished and widens the intervals.

{func}`prodigy.simulator.compare` checks a JSON result of the analysis against a simulation: every
query `Pr[g]`, `Ex[e]`, the first rows of every marginal and the mass of the distribution. A value
reported as exact, or the interval of a lower or upper bound, must meet the confidence interval of
the simulation (exact Clopper-Pearson intervals for probabilities, normal intervals for expected values, `z = 5`).

`tests/simulation/test_examples.py` does this for every example in `pgfexamples/` that the
simulator can run (all but those with parameters or `sample_pgf`); loops are unrolled, so the
analysis gives a sound interval with the residual mass, or an exact result if the loop terminates
during unrolling.

**What this covers, and what it cannot.** The comparison is statistical, not a proof. With a fixed
seed it is reproducible, and a correct result fails with a probability of about 6·10⁻⁷ per compared
number. It finds wrong results whose error is larger than the width of the interval (about
`5 · sqrt(p(1-p)/n)` for a probability `p` from `n` runs): a wrong generating function, filter,
normalisation or substitution typically shows up at once. It does not find small errors, errors in
the tail of a distribution beyond what the runs reach, errors in parts of the parser that both
share, or errors in expected values of heavy-tailed distributions (their sample variance is not a
reliable bound).

## Fuzzing (`tests/fuzzing`)

`tests/fuzzing/programs.py` is a [Hypothesis](https://hypothesis.readthedocs.io/) strategy for small
random programs: assignments, the built-in distributions, `if`, probabilistic choice, `observe`,
`loop(n)`, bounded `while` loops, modulo and floor division, and queries. Every program can be
rendered in `.ppy` and in `.pgcl`. For each program the tests check that

- both syntaxes give the same program;
- SymPy (with the simplification of the shown result on and off) and GiNaC give equal results, decided with
  {func}`prodigy.distribution.equality.decide_equal_strings`, and the same kind of guarantee;
- no result with status `ok` contains `nan`, `zoo` or `oo`;
- the results agree with the simulation.

A result with another status (e.g. a "not implemented" error) is incomplete, not wrong, and is not
compared. Hypothesis shrinks a failing program to a small one; it becomes an issue and a regression
test in `tests/fuzzing/test_found.py`. Classes of programs with known open bugs are switched off in
`Features` (each switch names its issue), so the test suite stays green until they are fixed.

**Parametric programs** (`test_parametric_programs_agree`, #453). With `Features(parameters=True)` the
probabilities of choices, `bernoulli`, `binomial`, `geometric` and `negative_binomial` may depend on the
parameters `p` and `q` (`param p: real;`), and a `poisson` rate may be a parameter. The parametric
probabilities form a class: `p`, `1 - p`, `p/2`, `(1 - p)/2`, `(1 + p)/2` and `p*q`, which map every
parameter value in (0, 1) into (0, 1), so they are admissible for every value the test draws.
Each program is checked by `tests.fuzzing.oracles.check_parametric`:

- engine-differential: SymPy and GiNaC on the parametric program, as above (the simulator does not run
  parameters);
- instantiation: the program with random admissible values for the parameters (rationals in (0, 1)) is
  checked as a constant program, and on every engine, each number of the parametric result with the values
  substituted must not be provably different from the number of the instantiated program;
- simulator: each parametric result with the values substituted must meet the confidence intervals of a
  simulation of the instantiated program.

## Engine conformance (`tests/distribution/test_conformance.py`)

Every engine implements the same `Distribution` interface, and the engines share their case
analysis. The conformance suite tests the contract of this interface once for all
engines: each test takes the fixture `factory` (`tests/distribution/conftest.py`), which runs it on
SymPy and on GiNaC (skipped where the GiNaC wheel is missing). It covers `copy` and the setters of
variables and parameters, equality and order (including undecidable comparisons, which raise or
answer `None`, never a wrong answer), filters, marginals and queries, the updates, `iid` for every
distribution of the language, iteration (finite distributions yield their states in the same graded
order on every engine), unknown names and collisions of variables and parameters, and the error
messages, which must be the same on every engine up to the names of temporary variables. Where the
engines' operations overlap with the FLINT rational kernel (`prodigy.distribution.kernel`), the suite
checks that they agree on rational generating functions.

Results are compared as expressions, never as strings: GiNaC's order of terms changes between runs.
An expected difference between the engines is a test named `test_declared_difference_…` that states
what each engine does; nothing is skipped silently. A new engine adds one entry to
`tests/distribution/engines.py` and must pass the suite. A test of a property of every engine goes
into the suite (or takes the fixture `factory`), not into a file per engine; `tests/distribution/sympy`
keeps the tests of SymPy internals.

## Running the tests

The tests that take more than about 10 s on a CI runner (mostly equivalence checks and loop strategies)
carry the marker `slow`. `make test` runs them; `make test-fast` (`-m "not fuzz and not slow"`) leaves
them out. The CI runs everything on Python 3.14 (with coverage) and `make test-fast` on Python 3.11,
so a slow test runs once per push; `.github/workflows/fuzzing.yml` also runs them on 3.11 every week.
A test that runs longer than 10 s on the CI gets the marker; the durations are in the artifact
`testreport.xml` of the job.

The fuzzing and the comparison of the examples with the simulation carry the pytest marker `fuzz`.
`make test` leaves them out (`-m "not fuzz"`); the CI runs them in the job `fuzzing (smoke test)`,
and locally they run with

```sh
make fuzz                         # profile ci: 20 derandomized programs (8 parametric), the fast examples
PRODIGY_FUZZ=nightly make fuzz    # profile nightly: 400 programs (150 parametric), fresh seed, all examples
```

The profile `ci` checks the same programs in every run, so the CI is reproducible. The workflow
`.github/workflows/fuzzing.yml` runs the profile `nightly` every week and on demand. Both print a
falsifying program together with a `@reproduce_failure` blob.

Every generated program runs in a child process (`tests.fuzzing.oracles.isolated`) with a time
budget for all its analyses and the simulation (5 s in `ci`, 60 s in `nightly`; a typical program
needs well below a second) and a memory limit of 3 GiB (enforced on Linux). The child is killed
when the budget runs out, so even a computation in native code that never returns to the
interpreter, or one that eats all memory, cannot hang or kill the whole run. In `ci` a program over
budget or out of memory is rejected and counted (`--hypothesis-show-statistics` shows the share),
so a slow program never blocks the CI; in `nightly` it fails with the program, since a slow
analysis is worth an issue too. The comparison of an example with the simulation runs the same way
(100 s in `ci`, 900 s in `nightly`). Every test has a limit of 120 s (pytest-timeout, `[tool.pytest.ini_options]`), after which it
fails with the stacks of all threads; the weekly workflow raises it for the slow examples.

## Benchmarks (`benchmarks/`)

Performance regressions do not make a test fail, so they are measured separately. The benchmark set
`benchmarks/suite.json` is a small set of examples: loops whose unrolling used to be much slower on
one engine than on the others (#224: doubling a variable, `geometric` inside a loop, nested loops,
`observe` after a loop), loop-free programs, invariant checks and an invariant synthesis. Each case
gives the program and the strategies of its loops, and runs on every engine:

```sh
make bench                                 # all cases on SymPy and GiNaC
make bench ARGS="--engines sympy"          # one engine
make bench ARGS="--cases st_petersburg"    # one case
make bench ARGS="--write-baseline"         # replace benchmarks/baseline.json by this run
```

`benchmarks/run.py` runs each case in a fresh process (`prodigy --engine E analyze --json`, so no
cache survives from one case to the next) with a time limit of 60 s and takes the analysis time from
the JSON result, without the start of Python. It prints a table (program × engine, seconds) and, with
`--summary FILE`, appends it as Markdown to `FILE`; `--output FILE` writes the results as JSON.

Most cases run with `--coefficients 0`, so the extraction of coefficients is not measured. A case
with the key `"coefficients": N` in `suite.json` runs with `--coefficients N`; the cases named
`*_coeffs` do this with N = 100 (a loop result, a loop-free multivariate result and an unrolled nested
loop with conditioning), which adds a few seconds to the whole run. They are ordinary cases, so
`make bench`, the baseline, the CI job and `benchmarks/compare.py` cover them without further options.
When the merge base has no such case yet, the comparison lists it as `new`.

The times are compared with `benchmarks/baseline.json`. Machines differ in speed, so the baseline
also stores the time of a fixed calibration workload (SymPy and plain Python), and the baseline times
are scaled by the ratio of the calibration times of this run and of the baseline. A case is a
regression if it takes more than three times its scaled baseline (`--threshold`) and more than one
second (`--min-seconds`), or if its status differs from the baseline (e.g. a time-out). The factor is
generous on purpose: it catches the large slowdowns that matter (a factor of 10 or 100, as in #224) and
ignores the noise of shared runners.

The total time of a case (`timing.total`: analysis, result summary and coefficient tables) is compared
in the same way, so that the `*_coeffs` cases find a slowdown of the coefficient generation. A baseline
without total times is compared on the analysis time only, until it is recorded again (below).

The CI job `benchmarks (report only)` runs `make bench` with `--fail-on-regression` on Python 3.14
and writes the table to the job summary. It is not required by `build`, so a noisy runner never
blocks a pull request; a red benchmark job is a hint to look at the table. After an intended change of
the times (a faster engine, a new case), the baseline is updated with `--write-baseline` in the same
pull request.

### Comparison with `main`

A pull request can make a program slower without exceeding the (generous) baseline factor, so the CI
job also benchmarks the merge base with `main` on the same runner and compares the two runs with
`benchmarks/compare.py`:

```sh
make bench ARGS="--output $PWD/head.json"    # on the branch
git worktree add ../base main               # a second checkout (or the merge base)
(cd ../base && uv run python benchmarks/run.py --output "$OLDPWD/base.json")
make bench-compare BASE=base.json HEAD=head.json
make bench-compare BASE=base.json HEAD=head.json ARGS="--factor 3 --min-seconds 1"
```

The script prints a Markdown table (program × engine, base → head seconds and the ratio) and lists the
cases that are slower: more than `--factor` times the base time (default 2) and more than
`--min-seconds` (default 0.5), so that the noise of short runs is ignored. Cases that exist on one
side only are marked `new` or `removed`; a status that changes to something other than `ok` (e.g. a
time-out) is flagged too. The exit code is 0 unless `--fail-on-slower` is given, 2 if a file cannot be
read. In the CI the comparison is written to the job summary; its steps are `continue-on-error`, so
the comparison never fails a pull request. The base is measured with its own `benchmarks/` and its own
locked dependencies; the table is a hint to look at the case, and a single flagged case on a shared
runner may be noise (run the job again).
