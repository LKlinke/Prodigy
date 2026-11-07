# Design: a Python-like surface syntax (`.ppy`)

Status: **implemented** ([#105](https://github.com/LKlinke/Prodigy/issues/105) design,
[#106](https://github.com/LKlinke/Prodigy/issues/106) implementation). `.ppy` is the syntax of
new programs and of the examples in `pgfexamples/`, which were converted with `prodigy convert`;
`.pgcl` is still read everywhere. The language is described in {doc}`language`; this page keeps
the reasons for the design. Parser, pretty printer and converter: `prodigy/pgcl/ppy_parser.py`,
`prodigy/pgcl/ppy_printer.py`, `prodigy/pgcl/converter.py`. The CLI (see {doc}`cli`) and the GUI
(see {doc}`gui`) read and write both syntaxes, and results write expressions in the syntax of the
program. Nested inference (`infer`, `sample`, see below) is analysed since
[#104](https://github.com/LKlinke/Prodigy/issues/104).

## Goals

- Programs that read like Python, but keep curly-brace blocks and semicolons.
- Both syntaxes produce **the same AST**. The type checker, the compiler and the analysis do not
  change, except for the new constructs listed below.
- `.pgcl` keeps working. New programs use `.ppy`. A converter (`prodigy convert`) translates `.pgcl`
  into `.ppy`. In the long run `.ppy` becomes the default; the old syntax is retired on purpose,
  with an announcement.
- The syntax can express everything. The analysis may still answer "not implemented" for cases it
  cannot compute exactly yet. Such cases end with a clear error, never with a wrong result.

## Overview

| Construct | `.pgcl` today | `.ppy` |
|---|---|---|
| Blocks | `{ … }` | `{ … }`, unchanged |
| Comments | `# …`, `// …` | `# …` to the end of the line, `/* … */` (not nested, may span lines) |
| Assignment | `x := e` | `x = e` |
| Equality | `x = e` | `x == e` |
| Boolean operators | `&`, `\|\|`, `not` | `and`, `or`, `not`; also `&&`, `\|\|`, `!` |
| Power | `x^2` | `x ** 2` |
| Division | `/` (rational) | `/` (rational), `//` (floor division, new) |
| Declaration | `nat x;` at the start | `x: nat;`, anywhere, block-scoped |
| Several variables | `nat x; nat y;` | `x, y: nat;` |
| Initial value | assignment after the declarations | `x: nat = 0;` |
| Bounds | `nat x [0, 10];` | `x: nat[0, 10];` |
| Parameters | `rparam p; nparam n;` | `param p: real; param n: nat;` |
| Constants | `const N := 10;` | `const N: nat = 10;` |
| `if` | `if (g) {…} else {…}`, `else` required | `else` optional, `else if` allowed |
| Binary choice | `{A} [p] {B}` | `{A} [p] {B}`, kept as shorthand |
| Choice with several branches | nested `{A} [p] {{B} [q] {C}}` | `switch` on a sampled variable |
| Categorical distribution | `x := 1 : 1/6 + 2 : 1/3 + 5 : 1/2` | `x = categorical({1: 1/6, 2: 1/3, 5: 1/2})` |
| Functions | `fun f := { nat a; … return e; }` | `def f(a: nat) -> nat { … return e; }` |
| Calls | `f(2)`, `f(a := 2)` | `f(2)`, `f(a = 2)` |
| Nested inference | `d := infer { f(2) }`, `x := sample { d }` | `d = infer(f, 2)`, `x = sample(d)` |
| Queries | `?Pr[…]`, `?Ex[…]`, `!Print`, `!Plot[…]`, `?Opt[…]` | `query Pr[…]`, `query Ex[…]`, `print`, `plot(…)`, `optimize(…)` |

`skip`, `abort`, `while`, `loop(n)`, `observe(g)` and the built-in distributions (`unif`,
`geometric`, `poisson`, `binomial`, `bernoulli`, `iid`, …) stay as they are. `tick(e)` and Iverson
brackets `[g]` are not part of either syntax; Prodigy's analysis never supported them. Comments start with `#` and run to the end of the line, or are enclosed in `/* … */` (not
nested; they may span lines and stand inside expressions). `//` is floor division in `.ppy`, so it
no longer starts a comment.

## Decisions in detail

### Blocks and statements

Curly braces delimit blocks, and semicolons end statements, as today. Indentation carries no
meaning. This keeps error messages, pasting into the editor and one-line expressions (invariants,
answers of language models) simple.

### Operators

`=` assigns, `==` compares. Boolean operators are accepted in both spellings: `and`/`or`/`not` and
`&&`/`||`/`!`. The canonical spelling, used by the converter and the pretty printer, is
`and`/`or`/`not`. `!` binds like `not`, as in Python weaker than the comparisons: `!x == y` means
`!(x == y)`. Comparisons do not chain; `a < b < c` is a syntax error.

`/` is rational division, as in Python. In guards, `x/2 <= 1` means `x <= 2`. An assignment to a
`nat` variable requires a natural result and is rejected otherwise; it never rounds (see the
section "Division" in the grammar page). `//` is floor division, as in Python (`7 // 2 == 3`). It
rounds the same way in guards and in assignments. As an update of a generating function it is
computed exactly via the residue classes modulo the divisor, as for `x % q`. Implemented: `e // q`
for a natural expression `e` and a positive natural constant `q`, exact if `e` has a finite marginal
or the generating function is rational in it; other cases end with an error.

### Declarations and scope

```
x, y: nat;
z: nat[0, 10] = 3;
const N: nat = 10;
param p: real;
```

- Declarations may appear anywhere. A variable is visible from its declaration to the end of the
  enclosing block.
- A declaration without a value gives 0. A declaration inside a loop body is executed in every
  iteration, so the variable starts again from its initial value.
- **At the end of its block, a variable is marginalised out of the generating function.** This
  keeps generating functions small, and helper variables of a loop body do not appear in the loop's
  invariant. Variables needed at the end of the program are declared at the outermost level.
- Every variable must be declared. Nothing is assumed implicitly. Using a variable before its
  declaration is an error.
- At the top level, declarations before the first statement declare the program's inputs, as in
  `.pgcl`: without a value they start at 0 unless an input distribution is given. A declaration
  after the first statement always starts at 0 (or at its value).
- `const`, `param` and `def` are allowed at the top level only.
- A declaration inside a block may not hide another name (a variable of the program or function,
  a constant, a parameter or a variable of an enclosing block). The same name may be declared in
  blocks that do not contain each other, with the same type. Only `nat` and `bool` variables can be
  declared in a block.
- Implementation: the compiler lowers a declaration inside a block into an assignment of its
  initial value and an assignment `x = 0` at the end of the block (`prodigy.pgcl.scope`). Setting
  `x` to 0 is the marginalisation (the generating function `G(x, …)` becomes `G(1, …)`), so the
  engines need nothing new, and outside its block the variable is always 0. Input values for it
  are ignored, and it is left out of the output distribution.

### Constants and parameters

- `const N: nat = 10;` is a known value. The compiler substitutes it everywhere: in expressions,
  `loop(N)`, bounds such as `nat[0, N]` and arguments such as `unif(1, N)`. It never appears in the
  generating function.
- `param p: real;` is an unknown symbol. Prodigy computes with it symbolically, e.g. for template
  and parameter synthesis.

### Conditionals

```
if (x > 3) {
    y = 1;
} else if (x > 1) {
    y = 2;
}
```

`else` is optional; a missing `else` means `else { skip }`. `else if` chains are allowed.

### Probabilistic choice

A choice with several branches is written as sampling a variable and switching on it:

```
c: nat = categorical({1: 1/6, 2: 1/3, 3: 1/2});
switch (c) {
    case 1: { x = 0; }
    case 2, 3: { x = 1; }
    default: { x = 2; }
}
```

- Every `case` is a block of its own. There is no fall-through. Several values are listed with
  commas. `default` is optional.
- Semantically, each `case` filters on `c == k`, and the results of the branches are added. This is
  computed exactly with the existing operations.
- `switch` works on every variable, not only on sampled ones. It takes a variable, not an
  expression, because each `case` refers to it again.
- If `c` is declared in a block and not used after the `switch`, it is marginalised out at the end
  of the block (see "Declarations and scope"), so the extra variable costs nothing afterwards.
- `{A} [p] {B}` stays as shorthand for the binary case.

### Categorical distribution

- Main form: `categorical({v1: p1, v2: p2, …})`, pairs of value and probability as in a Python
  dict.
- Short form: `categorical(p0, p1, …, pn)` for the values `0, 1, …, n`.
- The probabilities must sum to 1. For literal probabilities the parser checks this. For now the
  probabilities must be literals; probabilities with parameters, which the analysis would have to
  check, are not implemented yet.

### Functions

```
def f(a: nat, b: nat) -> nat {
    s: nat = a + b;
    return s;
}

x = f(2, 3);
y = f(2, b = 3);
```

Arguments are passed by position, optionally by name. Named arguments come after the positional
ones.

### Nested inference

Status: **implemented** ([#104](https://github.com/LKlinke/Prodigy/issues/104); the encoding is
described in `prodigy/analysis/nested_inference.py`, the language in {doc}`language`).

```
d: dist = infer(f, 10);
x = sample(d);
```

- `infer(f, args…)` takes a function and its arguments and gives the normalised output distribution
  of the call as a value of type `dist`. `sample(d)` draws independently from it.
- **Syntax and semantics are separate.** Every program whose semantics is well defined is a valid
  program: the parser and the type checker accept it. Whether the analysis can compute it is a
  separate question. If it cannot (yet), it ends with a clear "not implemented" message that names
  the construct, never with a wrong or approximate result.

#### Semantics

The semantics is defined for every valid program, whether the analysis can compute it or not.

- A value of type `dist` is a probability distribution on ℕ. With `dist` values, the state space is
  no longer countable, but it is measurable, so the semantics stays well defined.
- `infer(f, a)` in a state σ: let μ be the sub-distribution of the results of `f(a)` over the runs
  that terminate and pass every `observe`, and let Z = |μ| be its mass.
  - If Z > 0, the value is the conditional distribution μ / Z.
  - If Z = 0, the conditional distribution is undefined, and the run ends in an error state. The
    analysis reports this as an error, not as a violated observation.
- `sample(d)` draws a value from `d`, independently of all other draws. Several calls `sample(d)`
  give independent draws. `d2 = d` copies the distribution.
- `Pr[d == k]` and `Ex[d]` inside the program are expressions of type `real`. `Ex[d]` may be ∞.
  They may appear wherever a `real` expression is allowed, e.g. in guards: `if (Ex[d] > 3) { … }`.
- `dist` values may be passed to functions, `def g(d: dist) -> nat { … }`, and may be part of a
  loop's state, e.g. sequential updates `d = infer(update, d, obs)`.

#### What the analysis can compute

The semantics above is the specification. The table says which cases have an exact encoding with
generating functions. The other cases are valid programs that end with "not implemented".

| Case | Encoding | Exact |
|---|---|---|
| `d` does not depend on the state | closed-form PGF G_d(z) of the result; `x = sample(d)` marginalises `x` and multiplies by G_d(x), as `iid` does | yes, if the PGF of `f` is computable and Z > 0 is proven |
| copies and several draws from such a `d` | one factor G_d per draw | yes |
| `Pr[d == k]`, `Ex[d]` for such a `d` | a coefficient, or the derivative at 1: a number | yes |
| `d` depends on the state through arguments with finite support | case split over the values; each case as above | yes |
| `d` depends on the state, `f` has no `observe` and terminates almost surely (Z ≡ 1) | `sample(infer(f, a))` is a fresh call `f(a)` | yes, as function calls today |
| `d` depends on the state with infinite support, and Z depends on it | the joint distribution is Σₐ p(a)·μₐ/Z(a); dividing by Z(a) coefficient by coefficient is not an operation on the generating function | only special cases, e.g. Z(a) = c·rᵃ by the substitution a ↦ a/r; otherwise not implemented |
| `Pr`/`Ex` of a state-dependent `d` in guards or assignments | a filter by a non-linear predicate g(a) > c | not in general; possibly when {a : g(a) > c} is finite, cofinite or semilinear |
| `dist` values in a loop's state | a generating function cannot carry infinitely many different distributions as states | not in general; possibly when the family is indexed by a `nat` statistic, as for conjugate families, which can be encoded by the index |

The last three rows are research questions and get their own issues.

#### Decisions made in the implementation

The implementation follows the table. Where the design left a gap, it decided as follows.

- **Encoding.** A `dist` value is a *site*: the function and its arguments. The analysis numbers
  the sites it meets, and a `dist` variable is a natural number in the joint generating function,
  the number of its site (0 while it has no value). So a case split over `dist` values is a split
  over finitely many numbers, and copies and branches need nothing special. A state-dependent
  argument with infinitely many values (only allowed when Z ≡ 1) is kept in extra variables of the
  `dist` variable. At the queries and at the end, the `dist` variables are taken out of the
  distribution and reported separately.
- **Finite case splits also for `Pr`/`Ex` and loops.** Because a `dist` variable has finitely many
  values, `Pr[d == k]` and `Ex[d]` of a state-dependent `d` with finitely many values are computed
  by a case split, too; only infinitely many values are "not implemented". A loop that reads `dist`
  variables but assigns none runs once per value, with the values fixed; a loop that assigns a
  `dist` variable is "not implemented" (the last row).
- **`Pr` with comparisons.** Besides `Pr[d == k]`, `Pr[d < k]`, `<=`, `>`, `>=` (either side) are
  allowed; `k` must be a natural number, and one that depends on the state is "not implemented".
- **Numbers.** `Pr`/`Ex` are substituted as exact numbers. A comparison with an irrational value
  (e.g. e⁻¹) is decided exactly; using an irrational value as a probability or in arithmetic, and a
  value that depends on parameters, is "not implemented".
- **Z > 0 and exactness.** Z is computed exactly; Z = 0 is an `UndefinedDistributionError`, and Z
  depending on parameters is "not implemented". If the analysis of `f` is only a bound (a loop with
  a lower or upper bound), normalising it would give neither, so `infer` is "not implemented" then.
  For an argument with infinitely many values, Z ≡ 1 is proven by running `f` on the distribution
  of the arguments and checking that every argument value keeps its mass.
- **A `dist` without value.** Reading a `dist` variable before it was assigned is an
  `UndefinedDistributionError` (the semantics gives it no value).
- **Queries.** A query (`query Pr[…]`, `query Ex[…]`, `plot`) is about the program state, so a
  `dist` in it is a type error, with the hint to query a draw `y = sample(d)` instead. `Pr`/`Ex` on
  `nat` variables inside a program are a type error with the hint to wrap the code in a function
  and use `infer`.
- **`infer` inside functions.** Functions cannot call functions yet, so `infer` is only available
  in the program itself; `sample`, `Pr` and `Ex` of `dist` parameters work inside functions.
- **Not implemented: the special case Z(a) = c·rᵃ** of the sixth row. It would need a proof that Z
  has this form; it is left to the research issue.
- **Function calls.** `x = f(y)` with an argument that depends on the state ran the body on the
  marginal of the arguments and multiplied the result with the caller's distribution, so `x` came
  out independent of `y`; an `observe` in the body did not condition the program, and a call in a
  branch lost mass. Since `sample(infer(f, a))` relies on calls, calls with state-dependent
  arguments now run the body on the joint distribution, and the observations in the body
  condition the program.

#### Results

The result JSON gets a section `distributions` with the closed-form PGF of each `dist` variable,
separate from the joint generating function of the program state. The GUI shows it as a card of its
own. Implemented: one entry per variable, with one value per case if it depends on the state
(probability, `infer` text, closed form, Z, mean, coefficients; see `prodigy/analysis/result.py`);
the step trace lists the values next to each distribution.

### Queries

```
query Pr[x == 3];
query Ex[x + y];
print;
plot(x, y);
```

Queries are statements at the end of the program, introduced by keywords instead of the prefixes
`?` and `!`. No other statement may follow a query.

## Example side by side

`.pgcl`:

```
nat n;
nat m;
nat c;
rparam p;

while (0 < n) {
    { m := m + 1 } [p] { m := m + 2 }
    c := 1 : 1/2 + 2 : 1/4 + 3 : 1/4;
    if (c = 1) { n := n - 1 } else { if (c = 2) { m := 0 } else { skip } }
}
?Pr[m = 5]
```

`.ppy`:

```
n, m: nat;
param p: real;

while (n > 0) {
    {m = m + 1;} [p] {m = m + 2;}
    c: nat = categorical({1: 1/2, 2: 1/4, 3: 1/4});
    switch (c) {
        case 1: { n = n - 1; }
        case 2: { m = 0; }
    }
}
query Pr[m == 5];
```

In the `.ppy` version, `c` is declared inside the loop body, so it is marginalised out at the end of
every iteration and does not appear in the loop invariant. In the `.pgcl` version it is a variable
of the whole program.

## Still open

- The exact form of `plot` and `optimize`.
- When the old syntax is retired. The examples are converted (`.ppy` only in `pgfexamples/`); a
  representative set of their former `.pgcl` versions is kept in `tests/pgcl/former_examples/`, so
  the old syntax stays tested.

## Notes for the converter

`prodigy convert` (see {doc}`cli`) is built on the AST and the pretty printer.

- Comments are not part of the AST. The converter carries them over by their lines, so their
  position in the output is approximate (a comment at the end of a block moves behind the block).
- Names that are keywords in `.ppy` (`switch`, `case`, `default`, `def`, `param`, `print`, …) are
  renamed by appending `_` until the name is neither a keyword nor another name of the program,
  e.g. `switch` in the former `monty_hall_nested.pgcl` (`pgfexamples/inference/loop_free/conditioning/`)
  became `switch_`; it was the only renaming when the examples were converted. The converter reports every renaming. The pretty printer alone refuses such names.
