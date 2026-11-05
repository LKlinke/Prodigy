# Design: a Python-like surface syntax (`.ppy`)

Status: **agreed design** ([#105](https://github.com/LKlinke/Prodigy/issues/105)). Implementation:
[#106](https://github.com/LKlinke/Prodigy/issues/106); the parser, the pretty printer and the
converter exist (`prodigy/pgcl/ppy_parser.py`, `prodigy/pgcl/ppy_printer.py`,
`prodigy/pgcl/converter.py`). The CLI reads `.ppy` files and converts `.pgcl` files with
`prodigy convert` (see {doc}`cli`); the GUI and the examples in `pgfexamples/` still use `.pgcl`.
Block-scoped declarations and `//` are implemented (see {ref}`pgcl_grammar`, sections "Floor
division" and "Block-scoped declarations"). Semantics of nested inference:
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

`skip`, `abort`, `while`, `loop(n)`, `observe(g)`, `tick(e)`, the built-in distributions (`unif`,
`geometric`, `poisson`, `binomial`, `bernoulli`, `iid`, …) and Iverson brackets `[g]` stay as they
are. Comments start with `#`. `//` is floor division in `.ppy`, so it no longer starts a comment.

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

```
d: dist = infer(f, 10);
x = sample(d);
```

- `infer(f, args…)` takes a function and its arguments and gives the normalised output distribution
  of the call as a value of type `dist`. `sample(d)` draws independently from it.
- The syntax allows every case. For cases the analysis cannot handle yet, it ends with "not
  implemented" and a clear message. The semantics is designed in
  [#104](https://github.com/LKlinke/Prodigy/issues/104). Example: an argument that depends on the
  random state together with `observe` in `f`, where the normalising constant depends on that
  argument.

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

- How `dist` values appear in the result JSON and in the GUI
  ([#104](https://github.com/LKlinke/Prodigy/issues/104)).
- The exact form of `plot` and `optimize`.
- Whether the `.pgcl` files in `pgfexamples/` are converted in the repository, and when the old
  syntax is retired.

## Notes for the converter

`prodigy convert` (see {doc}`cli`) is built on the AST and the pretty printer.

- Comments are not part of the AST. The converter carries them over by their lines, so their
  position in the output is approximate (a comment at the end of a block moves behind the block).
- Names that are keywords in `.ppy` (`switch`, `case`, `default`, `def`, `param`, `print`, …) are
  renamed by appending `_` until the name is neither a keyword nor another name of the program,
  e.g. `switch` in `pgfexamples/inference/loop_free/conditioning/monty_hall_nested.pgcl` becomes
  `switch_`. The converter reports every renaming. The pretty printer alone refuses such names.
