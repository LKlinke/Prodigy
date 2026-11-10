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
- SymPy (with simplification on and off), GiNaC and SymEngine give equal results, decided with
  {func}`prodigy.distribution.equality.decide_equal_strings`, and the same kind of guarantee;
- no result with status `ok` contains `nan`, `zoo` or `oo`;
- the results agree with the simulation.

A result with another status (e.g. a "not implemented" error) is incomplete, not wrong, and is not
compared. Hypothesis shrinks a failing program to a small one; it becomes an issue and a regression
test in `tests/fuzzing/test_found.py`. Classes of programs with known open bugs are switched off in
`Features` (each switch names its issue), so the test suite stays green until they are fixed.

## Running the tests

The fuzzing and the comparison of the examples with the simulation carry the pytest marker `fuzz`.
`make test` leaves them out (`-m "not fuzz"`); the CI runs them in the job `fuzzing (smoke test)`,
and locally they run with

```sh
make fuzz                         # profile ci: 20 derandomized programs, the fast examples
PRODIGY_FUZZ=nightly make fuzz    # profile nightly: 400 programs with a fresh seed, all examples
```

The profile `ci` checks the same programs in every run, so the CI is reproducible. The workflow
`.github/workflows/fuzzing.yml` runs the profile `nightly` every week and on demand. Both print a
falsifying program together with a `@reproduce_failure` blob.

Every generated program has a time budget for all its analyses and the simulation (5 s in `ci`,
60 s in `nightly`; a typical program needs well below a second). In `ci` a program over budget is
rejected and counted (`--hypothesis-show-statistics` shows the share), so a slow program never
blocks the CI; in `nightly` it fails with the program, since a slow analysis is worth an issue
too. Every test has a limit of 120 s (pytest-timeout, `[tool.pytest.ini_options]`), after which it
fails with the stacks of all threads; the weekly workflow raises it for the slow examples.
