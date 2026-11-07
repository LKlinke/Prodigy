(ppy_reference)=
# Language reference (`.ppy`)

`.ppy` is Prodigy's program syntax: curly-brace blocks and semicolons, with Python's operators
and declarations. Files with the extension `.ppy` are read in it; the examples in `pgfexamples/`
are written in it. The original pGCL syntax (`.pgcl`) is still read everywhere (see
{ref}`pgcl_grammar`), and `prodigy convert` translates it into `.ppy` (see {doc}`cli`). Both
syntaxes produce the same program, so the analysis does not depend on the syntax. The reasons
for the design are in {doc}`language-design`.

```text
# geometric distribution: count the tails before the first head
c: nat;
x: nat = 1;
while (x == 1) {
    {x = 0;} [1/2] {c = c + 1;}
}
query Pr[c == 2];
query Ex[c];
```

## Programs

A program is a sequence of declarations and statements, followed by its queries. Declarations
of constants, parameters and functions (`const`, `param`, `def`) may stand anywhere at the top
level; variables may be declared anywhere (see [Declarations and scope](#declarations-and-scope)).
No statement may follow a query.

Every statement ends with `;`; statements with a block (`if`, `while`, `loop`, `switch`, choice)
end with their closing brace. Indentation and line breaks carry no meaning.

### Comments

`#` starts a comment that runs to the end of the line. `/* … */` encloses a comment that may span
lines and stand inside an expression; it does not nest. `//` is floor division, not a comment.

### Names and keywords

Names consist of letters, digits and `_` and do not start with a digit. These words are reserved
and cannot be used as names:

`abort`, `and`, `bool`, `case`, `categorical`, `const`, `def`, `default`, `dist`, `else`,
`false`, `if`, `infer`, `loop`, `nat`, `not`, `observe`, `optimize`, `or`, `param`, `plot`,
`print`, `query`, `real`, `return`, `sample`, `skip`, `switch`, `true`, `while`.

The names of the built-in distributions (`unif`, `geometric`, …) cannot be used for functions.
`prodigy convert` renames variables of a `.pgcl` program whose name is reserved in `.ppy` by
appending `_` (`switch` becomes `switch_`).

### Literals

- natural numbers: `0`, `42`;
- decimals: `0.5`, `1e-3`; they are exact (`0.1` is 1/10, not a binary float);
- fractions are written as divisions of literals: `1/3`;
- `true`, `false`;
- infinity: `∞` or `\infty`.

## Types

| Type | Values |
|---|---|
| `nat` | natural numbers 0, 1, 2, … |
| `nat[a, b]` | natural numbers from `a` to `b` (bounds are expressions over literals and constants) |
| `bool` | `true`, `false`; counted as 1 and 0 in the generating function |
| `real` | rational numbers; for parameters |
| `dist` | a distribution, the value of `infer(…)` (see [Nested inference](#nested-inference)) |

A value assigned to a `nat` variable must be a natural number in every state; Prodigy never
rounds (`x = x / 2` is only accepted where `x` is even). A `bool` variable accepts boolean
expressions and expressions that are provably 0 or 1 (e.g. `bernoulli(p)`).

## Declarations and scope

```text
x, y: nat;                  # several variables of one type
z: nat[0, 10] = 3;          # bounds and an initial value
const N: nat = 10;          # a constant; the type may be left out: const N = 10;
param p: real;              # a parameter (real or nat)
```

**Variables.** `names: type;` or `name: type = e;` declares variables. An initial value needs a
single variable (`x, y: nat = 0;` is an error).

- *At the top level, before the first statement,* declarations declare the program's inputs.
  Without a value, they start at 0 unless an input distribution is given (`prodigy main FILE
  INPUT`, the GUI's *Input*). With a value, the value is assigned at that place.
- *At the top level, after the first statement,* a declaration starts at its value, or at 0
  (`false`).
- *Inside a block* (a loop body, a branch, a `case`, a choice), a declaration is **block-scoped**:
  the variable exists from its declaration to the end of the block. It starts at its value or 0 in
  every execution of the block, and at the end of the block it is marginalised out of the
  distribution, so it neither appears in the output nor in a loop's invariant. Only `nat` and
  `bool` variables can be declared in a block. A block variable may not hide another name (a
  variable of the program or function, a constant, a parameter, or a variable of an enclosing
  block); the same name may be declared in blocks that do not contain each other, with the same
  type.

Every variable must be declared before it is used.

**Constants.** `const N: nat = e;` (or `const N = e;`) gives a name to a value. The value is
substituted everywhere before the analysis, also in `loop(N)`, in bounds `nat[0, N]` and in
arguments such as `unif(1, N)`.

**Parameters.** `param p: real;` or `param n: nat;` declares an unknown symbol. Prodigy computes
with it symbolically (e.g. a choice `[p]` or `geometric(p)`), and parameter synthesis searches for
values of it. Parameters have no bounds.

`const`, `param` and `def` are allowed at the top level only.

## Statements

| Statement | Meaning |
|---|---|
| `x = e;` | assignment |
| `x = D;` | sampling from a built-in distribution `D`, e.g. `x = unif(1, 6);` (see [Distributions](#built-in-distributions)) |
| `skip;` | does nothing |
| `abort;` | ends the run without a result; its mass is lost (it does not count as a violated observation) |
| `observe(g);` | conditioning: keeps the runs in which `g` holds; the others count as violated observations and the rest is renormalised |
| `if (g) { … }` | `else { … }` and `else if (g) { … }` are optional; a missing `else` does nothing |
| `{ … } [p] { … }` | probabilistic choice: the first block with probability `p`, the second with `1 − p` |
| `switch (x) { case 1: { … } case 2, 3: { … } default: { … } }` | runs the block of the first `case` whose value equals `x`; `default` (optional) otherwise. No fall-through. `switch` takes a variable, not an expression |
| `while (g) { … }` | loop; how it is analysed is chosen per loop (loop strategies, see {doc}`gui` and {doc}`cli`) |
| `loop(n) { … }` | runs the block `n` times; `n` is a natural literal or a constant |
| `query { x = e; observe(g); }` | kept from `.pgcl`: a block of exactly one assignment and one observation, analysed on a finite distribution |
| `;` | an empty statement |

Probabilities `p` are expressions over literals and parameters. A choice with more than two
branches is a `switch` on a variable drawn from `categorical` (see below), or a nested choice.

## Expressions

### Operators

From the weakest to the strongest binding:

| Operators | Meaning |
|---|---|
| `or`, `\|\|` | disjunction |
| `and`, `&&` | conjunction |
| `not`, `!` | negation |
| `<`, `<=`, `>`, `>=`, `==` | comparison (does not chain: `a < b < c` is an error) |
| `+`, `-` | addition, subtraction (left-associative) |
| `*`, `/`, `//`, `%` | multiplication, rational division, floor division, modulo (left-associative) |
| `-` (unary) | negation |
| `**` | power (right-associative; `-x ** 2` is `-(x ** 2)`) |

The canonical spelling is `and`/`or`/`not`; `prodigy convert` and the pretty printer write it.
As in Python, `not` (and `!`) binds weaker than the comparisons: `not x == 1` is `not (x == 1)`.
Parentheses group as usual. Square brackets are not an expression: there are no Iverson brackets
`[g]` (a condition as the number 0 or 1); a program that uses them, or the former `tick(e)`
statement, is rejected with a syntax error at its position.

**Division.** `/` is rational division and never rounds: in a guard, `x / 2 <= 1` means
`x <= 2`; assigned to a `nat` variable, the result must be a natural number. `e // q` rounds down
as in Python (`7 // 2 == 3`), in guards and assignments alike; `q` must be a positive natural
literal or constant. `e % q` is the remainder. How these are computed on infinite distributions,
and when they are exact, is described in {ref}`pgcl_grammar` (sections "Division", "Modulo" and
"Floor division").

**Guards.** Conditions of `if`, `while`, `observe` and `query Pr[…]` are boolean expressions;
comparisons are evaluated over the rationals (see "Guards" in {ref}`pgcl_grammar` for what can
be filtered on infinite distributions).

### Built-in distributions

A built-in distribution is sampled by assigning it to a variable. All arguments are positional
and required.

| Distribution | Values |
|---|---|
| `bernoulli(p)` | 1 with probability `p`, else 0 |
| `geometric(p)` | `k` with probability `(1 − p)^k p`, `k ≥ 0` |
| `poisson(l)` | Poisson with mean `l` |
| `logdist(p)` | logarithmic distribution |
| `binomial(n, p)` | binomial with `n` trials |
| `negative_binomial(r, p)` | the number of failures before the `r`-th success: `k` with probability `C(k + r − 1, k) (1 − p)^k p^r`, `k ≥ 0`; the same as `iid(geometric(p), r)`. `r` is a natural number, `p` lies in `(0, 1]` |
| `unif(a, b)` | uniform on `a, …, b` (also `unif_d`) |
| `iid(D, n)` | the sum of `n` independent samples of `D`; `n` is a variable or a natural number |
| `sample_pgf(G, x)` | `x` drawn from the generating function `G` in `x` (the second argument is the assigned variable) |
| `categorical({v1: p1, v2: p2, …})` | `v1` with probability `p1`, …; `categorical(p0, p1, …, pn)` for the values `0, …, n` |

The probabilities of `categorical` are literals and must sum to 1; this is checked when the
program is read. `categorical`, `infer` and `sample` stand only as a whole right-hand side (of an
assignment, an initial value, a `return` or an argument).

## Functions

```text
def f(a: nat, b: nat) -> nat {
    s: nat = a + b;
    return s;
}

x = f(2, 3);
y = f(2, b = 3);
```

- A function takes parameters with types and returns a natural number (`-> nat` may be left out;
  other return types are not implemented yet). Its body ends with `return e;`.
- Variables declared in the body are local to the function and start at their value or 0.
- Arguments are passed by position, optionally by name (`b = 3`); named arguments come after the
  positional ones. Calls are checked against the number and the names of the parameters.
- A call `x = f(…);` runs the body on the distribution of the arguments and assigns the
  distribution of the returned value to `x`.
- A function may be declared after the code that calls it.

## Nested inference

```text
def claims(risk: nat) -> nat {
    c: nat = binomial(3, 1/4);
    observe(c >= risk);
    return c;
}
d: dist = infer(claims, 1);   # the distribution of c, given c >= 1
x: nat = sample(d);           # an independent draw from d
y: nat = sample(d);           # another one, independent of x
if (Ex[d] > 1 and Pr[d == 1] < 3/4) { … }
```

- `infer(f, args…)` is the distribution of the value that `f(args…)` returns, **given** that the
  call terminates and passes every `observe` in `f`: the sub-distribution of the results divided by
  its mass Z. A value of type `dist` holds such a distribution. If Z = 0 (no run terminates and
  passes the observations), the value is undefined and the analysis ends with an error
  (`UndefinedDistributionError`); this is not a violated observation of the program. The
  observations inside `f` do not condition the program that calls `infer`.
- `sample(d)` draws a value from `d`, independently of every other draw: two `sample(d)` give two
  independent values. `e = d` copies the distribution; assigning `d` later does not change `e`.
- `Pr[d == k]` (also `<`, `<=`, `>`, `>=`) and `Ex[d]` are `real` numbers, for a `dist` variable
  `d` and a natural number `k`. They may stand wherever a `real` expression may, e.g. in guards and
  as probabilities: `{ … } [Pr[d == 0]] { … }`, `bernoulli(Ex[d] / 10)`. `Ex[d]` may be `∞`.
- A `dist` may be passed to a function: `def g(e: dist) -> nat { a: nat = sample(e); return a; }`.
- `infer` and `sample` are the whole right-hand side of an assignment or a declaration
  (`return sample(e);` works, too). `infer` is not available inside functions, since functions
  cannot call functions yet.
- `Pr[…]` and `Ex[…]` inside a program are only about `dist` values. For the probability of an
  event of the program state, wrap the code in a function and use `infer`. Queries are about the
  program state, so a `dist` cannot appear in a query; query a draw instead
  (`y = sample(d); query Ex[y];`, which by the law of total expectation is the mean of `Ex[d]`).

The `.pgcl` syntax is `dist d;`, `d := infer { f(2) }`, `x := sample { d }`, `Pr[d = 1]` and
`Ex[d]`.

### What the analysis computes

Every program above has a well-defined semantics (see {doc}`language-design`). The analysis
computes these cases exactly, on all engines:

| Case | How |
|---|---|
| the arguments of `infer` do not depend on the program state | the closed-form generating function G of the normalised distribution; `x = sample(d)` multiplies by G(x) |
| copies, several draws, a `dist` as function argument | one factor G per draw |
| `Pr[d == k]`, `Ex[d]` | a coefficient or the derivative at 1 of G, substituted as a number |
| arguments that depend on the state with finitely many values | a case split over the values |
| arguments with infinitely many values, `f` without `observe` and terminating almost surely | `sample(d)` is the call `f(a)` (Z = 1 is proven when `infer` runs) |
| a loop that reads `dist` variables but assigns none | the `dist` values are fixed while the loop runs, so the loop strategies do not see them |

Z > 0 is proven, and the normalisation is exact: if the analysis of `f` gives only a bound (a
loop in `f` with a lower or upper bound), `infer` is not computed. These cases end with a "not
implemented" error that names the construct:

- arguments with infinitely many values when `f` has `observe` (Z may depend on them);
- `Pr`/`Ex` of a `dist` that depends on the state with infinitely many values;
- assigning a `dist` inside a loop (a `dist` value in the state of a loop);
- `Pr[d == k]` with a `k` that depends on the state, and `Pr`/`Ex` that depend on parameters;
- an irrational `Pr`/`Ex` (e.g. Pr[d == 0] = e⁻¹ for `poisson(1)`) used as a probability or in
  arithmetic; comparisons with it are decided exactly.

At the end, and at the queries, the `dist` variables are not part of the program's distribution;
their values are reported separately (the section `distributions` of the result JSON, a card in
the GUI). Examples: `pgfexamples/inference/nested/`.

## Queries

Queries come after all statements and are answered on the output distribution, in this order:

| Query | Answer |
|---|---|
| `query Pr[g];` | the probability of the condition `g` |
| `query Pr[x];` | the marginal distribution of the variable `x` |
| `query Ex[e];` | the expected value of `e` (a variable, a literal, or `+`, `-`, `*`, `//` of them) |
| `print;` | the whole distribution |
| `plot(x);`, `plot(x, y);` | a bar chart of the marginal of `x` or the joint distribution of `x` and `y`; an optional third argument limits it: a natural number of terms, a probability threshold, or `∞` |
| `optimize(e, p, MAX);` | the value of the parameter `p` that maximises (`MIN`: minimises) `e` on the output distribution |

Results name queries, loop guards and instructions in the syntax of the program: a `.ppy`
program gets `x == 1 and y < 3`, a `.pgcl` program `(x = 1) & (y < 3)` (the result JSON states the
syntax in `syntax`, see {doc}`gui`).

## Differences from `.pgcl`

| `.pgcl` | `.ppy` |
|---|---|
| `x := e` | `x = e` |
| `x = e` (comparison) | `x == e` |
| `&`, `\|\|`, `not` | `and`, `or`, `not` (also `&&`, `\|\|`, `!`) |
| `x^2` | `x ** 2` |
| `//` comment | `#` comment, `/* … */`; `//` is floor division |
| `nat x;` at the start | `x: nat;` anywhere, block-scoped in blocks |
| `nat x [0, 10];` | `x: nat[0, 10];` |
| `rparam p; nparam n;` | `param p: real; param n: nat;` |
| `const N := 10;` | `const N: nat = 10;` |
| `else { skip }` required | `else` optional, `else if` |
| `x := 1 : 1/6 + 2 : 5/6` | `x = categorical({1: 1/6, 2: 5/6});` |
| `fun f := { nat a; … return e; }` | `def f(a: nat) -> nat { … return e; }` |
| `f(a := 2)` | `f(a = 2)` |
| `d := infer { f(2) }`, `x := sample { d }` | `d = infer(f, 2)`, `x = sample(d)` |
| `?Pr[g]`, `?Ex[e]`, `!Print`, `!Plot[x, y, 10]`, `?Opt[e, p, MAX]` | `query Pr[g];`, `query Ex[e];`, `print;`, `plot(x, y, 10);`, `optimize(e, p, MAX);` |

## Grammar

The [Lark](https://lark-parser.readthedocs.io/) grammar that {mod}`prodigy.pgcl.ppy_parser` is
built from. The parser turns sugar (`else if`, `switch`, `categorical`, declarations with values)
into the program representation shared with `.pgcl`, and reports errors with line and column.

```{literalinclude} _generated/ppy.lark
:language: text
```
