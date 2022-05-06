# Prodigy: PRObability DIstributions via GeneratingfunctionologY

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/LKlinke/Prodigy/tree/ae)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4725465.svg)](https://doi.org/10.5281/zenodo.6511363)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://github.com/LKlinke/Prodigy/blob/ae/LICENSE)

Prodigy is a tool for analyzing probabilistic integer programs with `while`-loops. It is based on (probability) generating functions.

Given an almost-surely terminating loop `while(G) {B}` and a loop-free (specification) program `I` (also called _invariant_), prodigy checks whether `while(G) {B}` and `I` are _equivalent_ programs, i.e., they yield the same output distribution on every possible input distribution.

For more technical details, please refer to our CAV'22 paper:
> Mingshuai Chen, Joost-Pieter Katoen, Lutz Klinkenberg, Tobias Winkler:
Does a Program Yield the Right Distribution? Verifying Probabilistic Programs via Generating Functions. In Proc. of CAV'22, to appear (as enclosed in the zipfile).

Experiments in the paper are carried out on a 2,4GHz Intel i5 Quad-Core processor with 16GB RAM running macOS Monterey 12.0.1.

## Contents

* Building & loading the Docker image
* Smoke test
* Replicating the results from the paper
* Running your own example
* Supported program syntax

## Building & loading the Docker image

1. Install Docker (https://www.docker.com/get-started/) in case you do not have it yet.
2. Checkout the repository on the artifact-evaluation branch:
    ```bash
    git clone --single-branch --branch ae https://github.com/LKlinke/prodigy.git
    ```
3. Locate the repository and build the docker image:
   ```bash
   cd prodigy && make docker-build
   ```
4. Load the Docker image of Prodigy (~1min):
   ```bash
   docker image load -i prodigy.tar.gz
   ```
   
The structure of the artifact is as follows (`ls -l`).

```bash
/root/artifact
├── docs                   # documentation of Prodigy
├── pgfexamples            
|   └── paper_examples     # example pGCL-programs #1 - #11 with their corresponding invariants
├── prodigy                # source code of Prodigy
├── tests                  # Prodigy unit tests
├── load_env.sh            # script for loading the virtual python environment
├── pyproject.toml         # Prodigy dependencies
├── reproduce_results.sh   # script for reproducing the results in the paper
└── ...
```

## Smoke test

For a quick test to see if everything works:

5. Run Prodigy via Docker:
   ```bash
   docker run -it prodigy
   ```
You shall see a welcome message by Prodigy and be directed into the Docker container (which can be exited via `exit`).


## Replicating the results from the paper

### Reproducible elements in the paper

The equivalence check for Examples #1 - #11 presented in the paper, as well as the corresponding queries (on the output distributions) for Examples #1, #3, and #9.

### Reproducing the results

6. Reproduce the results presented in the paper:
    ```bash
    ./reproduce_results.sh
    ```


    > _Note:_ The script uses predefined backends. Prodigy currently supports `ginac` and `sympy`. The former enables our C++ backend based on the GiNaC package. The latter employs the python computer algebra package sympy. `ginac` is generally faster than `sympy`, however for computing queries on final distributions, the current implementation relies on `sympy`.

Observe that in case of _nested_ or _sequential_ loops (e.g., Example #1, #5, and #11), the invariants (e.g., outer_inv and inner_inv) are verified jointly and thus the timings shall be totaled.

## Running your own example

To experiment with Prodigy on a customized example, you need to create two files: 1) a program consisting of a single `while`-loop and 2) a loop-free invariant program (the supported syntax is specified further below).

1. Open an editor by typing, e.g., `nano myexample.pgcl`.
2. Write a program with a single `while`-loop such as
    ```bash
    nat n; nat x;
    while(n > 0) {
        n := n - 1;
        { x := 0; } [1/2] { x := 1; }
    }
    ```
3. The effect of the above program can be summarized as follows: If `n > 0` initially, then upon termination `n` is set to zero and `x` is randomly assigned either `0` or `1`. This can be formally verified by writing the following loop-free invariant program (e.g., `nano myexample_inv.pgcl`):
    ```bash
    nat n; nat x;
    if(n > 0) {
        { x := 0; } [1/2] { x := 1; }
        n := 0;
    } else {}
    ```
4. Check that your invariant is correct via
    ```
    python prodigy/cli.py --engine ginac check_equality myexample.pgcl myexample_inv.pgcl 
    ``` 
5. Queries can be specified below the `while`-loop itself. Assume that we are interested in the expected value of `x` after termination with initial distribution `n^5`, we can append `?Ex[x]` to the end of `myexample.pgcl` and then invoke Prodigy in the `main` mode:
    ```bash
    python prodigy/cli.py main myexample.pgcl "n^5"
    ```
    when prompted press `1` and confirm your input with `[ENTER]`. Now give the path to the invariant       file:
    ```bash
    myexample_inv.pgcl
    ```
    and confirm your input with `[ENTER]` again.
    
    Similarly you can query probabilities by appending `?Pr[...]` to `myexample.pgcl`.

## Supported program syntax

```
start: declarations instructions queries

declarations: declaration* -> declarations

    declaration: "bool" var                  -> bool
               | "nat" var bounds?           -> nat
               | "real" var bounds?          -> real
               | "const" var ":=" expression -> const
               
    bounds: "[" expression "," expression "]"
    instructions: instruction* -> instructions
    
    queries: query* -> queries
    
    instruction: "skip"                                      -> skip
               | "while" "(" expression ")" block            -> while
               | "if" "(" expression ")" block "else"? block -> if
               | var ":=" rvalue                             -> assign
               | block "[" expression "]" block              -> choice
               | "loop" "(" INT ")" block                    -> loop
    
    query: "?Ex" "[" expression "]"                          -> expectation
               | "?Pr" "[" expression "]"                    -> prquery
               | "!Print"                                    -> print
               
               
    block: "{" instruction* "}"
    rvalue: "unif_d" "(" expression "," expression ")" -> duniform
               | "unif" "(" expression "," expression ")" -> duniform
               | "unif_c" "(" expression "," expression ")" -> cuniform
               | "geometric" "(" expression ")" -> geometric
               | "poisson" "(" expression ")" -> poisson
               | "logdist" "(" expression ")" -> logdist
               | "binomial" "(" expression "," expression ")" -> binomial
               | "bernoulli" "(" expression ")" -> bernoulli
               | "iid" "(" rvalue "," var ")" -> iid
               | expression
               
    literal: "true"  -> true
               | "false" -> false
               | INT     -> nat
               | FLOAT   -> real
               | "∞"     -> infinity
               | "\infty" -> infinity
    var: CNAME
    
    
    %ignore /#.*$/m
    %ignore /\\/\\/.*$/m
    %ignore WS
    %ignore ";"
    %import common.CNAME
    %import common.INT
    %import common.FLOAT
    %import common.WS
    
    // create the expression table (this is auto-generated in code)
    // Operator precedences are also encoded by this format.
    
    ?expression: expression_0
    ?expression_0: expression_1
               | expression_0 "||" expression_1 -> or
    ?expression_1: expression_2
               | expression_1 "&" expression_2 -> and
    ?expression_2: expression_3
               | expression_2 "<=" expression_3 -> leq
               | expression_2 "<" expression_3 -> le
               | expression_2 ">" expression_3 -> ge
               | expression_2 ">=" expression_3 -> geq
               | expression_2 "=" expression_3 -> eq
    ?expression_3: expression_4
               | expression_3 "+" expression_4 -> plus
               | expression_3 "-" expression_4 -> minus
    ?expression_4: expression_5
               | expression_4 "*" expression_5 -> times
               | expression_4 "/" expression_5 -> divide
    ?expression_5: expression_6
               | expression_6 "^" expression_5 -> power
    ?expression_6: expression_7
               | expression_6 ":" expression_7 -> likely
    ?expression_7: expression_8
               | expression_7 "%" expression_8 -> mod
    ?expression_8: "not " expression_8 -> neg
               | "(" expression ")" -> parens
               | "[" expression "]" -> iverson
               | literal -> literal
               | var -> var
```
