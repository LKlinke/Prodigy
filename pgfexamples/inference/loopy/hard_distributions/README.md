# Hard distributions

Programs whose output distributions are hard for an analysis with generating functions: heavy
tails, infinite moments, non-termination with positive probability, and generating functions that
are not rational. The folders group them by the class of the generating function (PGF) of the
result:

| Class | Example | Distribution | Termination | Moments |
|---|---|---|---|---|
| algebraic | `algebraic/galton_watson_critical.ppy` | total progeny, Catalan numbers, `1 - sqrt(1 - x)` | AST, not PAST | `Ex = ∞` |
| algebraic | `algebraic/galton_watson_supercritical.ppy` | total progeny, `(3 - sqrt(9 - 8x))/4` | terminates with probability 1/2 | finite |
| D-finite | `d_finite/polya_urn.ppy` | Pólya urn after geometric(1/2) draws, `ln(2 - x)/(1 - x)` | AST | finite |
| D-finite | `d_finite/polya_urn_coins.ppy` | the same with fair coins only (Fast Dice Roller) | AST | finite |
| D-finite | `d_finite/yule_simon_urn.ppy` | Yule–Simon (ρ = 1), `Pr[n] = 1/((n+1)(n+2))` | AST, not PAST | `Ex = ∞` |
| D-finite | `d_finite/yule_simon_coins.ppy` | the same with fair coins only | AST, not PAST | `Ex = ∞` |
| D-finite | `d_finite/yule_simon_rho2_coins.ppy` | Yule–Simon (ρ = 2), `Pr[n] = 4/((n+1)(n+2)(n+3))` | AST | `Ex = 1`, `Ex[r^2] = ∞` |
| D-finite | `d_finite/log_series_coins.ppy` | logarithmic distribution, `ln(1 - x/2)/ln(1/2)` | AST | finite |
| not D-finite | `non_d_finite/st_petersburg.ppy` | St. Petersburg payout, `Σ 2^-(k+1) x^(2^k)` | AST | `Ex = ∞` |

Related examples elsewhere: the first-passage time of the symmetric random walk
(`lower_bounds/exact/random_walk_counter.ppy`, algebraic), upward-drifting walks that terminate with
probability 1/2 (`lower_bounds/sandwich/escape_walk.ppy`), and Poisson(1/2) from fair coins
(`inference/loopy/conditioning/poisson_from_coins.ppy`, PGF `exp((k - 1)/2)`, D-finite).

Every file states its distribution, its class and the exact values of its queries (`# expected:`)
and says what Prodigy proves today. A non-rational PGF is never the fixed point of a rational
invariant, so the exact strategies of Prodigy cannot prove these results yet; unrolling
(`unroll`, `mass_threshold`) gives sound intervals.

Two of the urn programs choose with a probability that depends on program variables
(`[red/(red + black)]`). Prodigy does not compute such choices correctly yet (issue #213); the
`*_coins.ppy` variants use fair coins only and give correct bounds.

Sources (textbook material only): N. L. Johnson, A. W. Kemp and S. Kotz, *Univariate Discrete
Distributions*, 3rd ed., Wiley 2005; T. E. Harris, *The Theory of Branching Processes*, Springer
1963; P. Flajolet and R. Sedgewick, *Analytic Combinatorics*, CUP 2009; W. Feller, *An
Introduction to Probability Theory and Its Applications*, Vol. I, 3rd ed., Wiley 1968;
J. Lumbroso, *Optimal Discrete Uniform Generation from Coin Flips, and Applications*,
arXiv:1304.1916, 2013.
