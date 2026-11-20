# The FLINT rational kernel

`prodigy.distribution.kernel` holds exact kernels below the `Distribution` logic (#417). A kernel is
not an engine: it has no case analysis (guards, support splits, zero mass), only CAS operations that
the shared `Distribution` logic or an engine may call as hooks. The analysis itself does not use it
yet (step 1 of #417: the kernel and a measurement); the coefficients of the result tables and grids do
for exponentials of polynomials (below).

## `RationalGF`

`prodigy.distribution.kernel.rational.RationalGF` is a rational function P/Q over ℚ, with P and Q
multivariate polynomials in FLINT (`fmpq_mpoly` of python-flint; FLINT's `fmpz_mpoly_q` is not
exposed by python-flint 0.9, so the pair is cancelled by a GCD in Python). Program variables and
parameters are both ring variables. Every value is reduced: gcd(P, Q) = 1 and Q has leading
coefficient 1, so the representation is unique.

- Arithmetic: `+`, `-`, `*`, `/`, integer powers, with rational constants mixed in. Polynomials skip
  the GCD; products cancel crosswise (gcd(a, d), gcd(c, b)); sums cancel only by the common factor of
  the denominators.
- `subs(name, value)`: a variable replaced by a constant, polynomial (one FLINT `compose`) or rational
  function (homogenised Horner scheme); a denominator 0 raises `ZeroDivisionError`.
- `coefficients(name)`: the Taylor coefficients in one variable by the linear recursion of the
  denominator (as `rational_series.py` does in SymPy), as functions of the other variables; a pole at
  0 raises `PoleAtZeroError`.
- `from_sympy`/`to_sympy`: a walk over the SymPy expression tree. Anything that is not a rational
  function over ℚ (`exp`, `log`, roots, floats, `pi`, `I`, symbolic exponents) raises
  `NotRepresentableError`, the caller's signal to stay with SymPy.

The tests (`tests/distribution/test_rational_kernel.py`) check every operation and the round trip
SymPy → kernel → SymPy against SymPy on random rational functions in two variables and a parameter
(hypothesis).

## Coefficients of exponentials of polynomials

`prodigy.distribution.exponential_series` (#455) computes the Taylor coefficients of finite sums of
`R·exp(g)`, with `R` rational and `g` a polynomial in the expanded variables, in the kernel: from
`F' = g'·F`, each coefficient of `exp(g − g(0))` follows from the previous `deg g` ones, and the factor
`R = P/Q` adds a finite convolution with `P` and the recursion of `Q`. Compound Poisson distributions
(`x = poisson(λ); y = iid(D, x)`, generating function `exp(λ·(x·P(y) − 1))`) belong to this class, as do
rational functions (`g = 0`). The class is closed under taking a coefficient, so a joint grid is expanded
in x and then in y without leaving the kernel; only the cells shown are translated to SymPy. The shared
`taylor_stream` (result tables) and `taylor_window` (grids) of `coefficient_stream.py` use it for both
engines; functions outside the class keep the series expansion.

## Measurement

`benchmarks/kernel.py` (not part of the test suite) runs the generating-function operations of loops
of the benchmark set (`benchmarks/suite.json`, #348) on each backend, each measurement in a fresh
process with a time limit of 60 s:

- `kernel`: `RationalGF`.
- `sympy-cas`: the same operations on SymPy expressions with `cancel` after each iteration, i.e.
  SymPy without Prodigy's case analysis.
- `sympy`, `ginac`: the engines' `Distribution` (`filter`, `update`, arithmetic), what an
  unrolled loop costs in the analysis today.

```sh
timeout 900 uv run python benchmarks/kernel.py     # all, a Markdown table (several minutes)
```

Apple M-series (arm64), Python 3.13, SymPy 1.14, python-flint 0.9, prodigy-ppl-ginac
0.0.6; times are the fastest of up to five runs, the factor is the time over the kernel's.

| workload | kernel | sympy-cas | sympy | ginac |
|---|---|---|---|---|
| unroll st_petersburg (30) | 0.3 ms | 102.5 ms (348×) | 30.4 ms (103×) | 13.1 ms (44×) |
| unroll galton_watson (15) | 0.3 ms | 294.7 ms (1051×) | 1.81 s (6437×) | 32.8 ms (117×) |
| unroll galton_watson (40) | 1.2 ms | 1.66 s (1394×) | 17.48 s (14673×) | 243.7 ms (205×) |
| unroll random_walk (25) | 0.2 ms | 73.6 ms (339×) | 538.5 ms (2481×) | 42.9 ms (198×) |
| cancel and decide equal | 1.0 ms | 47.2 ms (49×) | 103.6 ms (108×) | 3.9 ms (4×) |
| 100 coefficients | 1.0 ms | 3.5 ms (4×) | 3.6 ms (4×) | 113.7 ms (119×) |
| SymPy → kernel → SymPy | 0.2 ms | – | – | – | – |

The kernel's times include the conversion from and to SymPy where the workload starts or ends in
SymPy (cancellation, coefficients).

Share of the rational CAS in a whole analysis on SymPy (cProfile of `prodigy --engine sympy analyze`):
in `galton_watson_critical` (15 unrollings, 9.9 s under the profiler) `cancel`/`together` take 4.3 s
and the evaluation at 1 by `limit` (`_limit_at_one`, a removable singularity) 3.0 s, together about
70 %; in `log_series_coins` (18.2 s) about 60 % (cancel 7.0 s, together 2.8 s, limit 2.0 s). All
three are exact operations on rational functions that the kernel does in microseconds to
milliseconds.

## Reading the numbers

- On the operations alone the kernel is two to four orders of magnitude faster than SymPy, and one
  to two than GiNaC's engine; the gap grows with the size (Galton–Watson 15 → 40 unrollings: SymPy's
  engine ×6400 → ×14700).
- Used only as a hook inside the SymPy engine (step 2), Amdahl's law bounds the gain on a whole
  analysis by the CAS share: about 2.5–3.5× on the profiled cases, more where the CAS share is
  higher (the long unrollings of #224 and #348). A larger gain needs the `Distribution` logic to keep
  kernel objects between operations instead of SymPy expressions or strings.
- Coefficient extraction is already cheap in SymPy (`rational_series.py`); the kernel gains ×4 there.
  GiNaC's native coefficient iterator is the slow one (×119).
- The kernel covers rational generating functions only. Examples with `exp`, `log` or algebraic
  closed forms keep SymPy (or GiNaC) for those parts, so whether GiNaC can become optional (step 4)
  depends on how much of the example library is non-rational.
