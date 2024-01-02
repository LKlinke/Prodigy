# Prodigy: PRObability DIstributions via GeneratingfunctionologY

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/LKlinke/Prodigy/tree/ae)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10451092.svg)](https://doi.org/10.5281/zenodo.10451092)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://github.com/LKlinke/Prodigy/blob/ae-oopsla/LICENSE)

Prodigy is a tool for inferring posterior distributions described by probabilistic integer programs with `while`-loops. It is based on (probability) generating functions.

In case of loopy programs, given an almost-surely terminating loop `while(G) {B}` and a loop-free (specification) program `I` (also called _invariant_), prodigy checks whether `while(G) {B}` and `I` are _equivalent_ programs, i.e., they yield the same output distribution on every possible input distribution.

For more technical details, please refer to our OOPSLA'24 paper:
> Lutz Klinkenberg, Christian Blumenthal, Mingshuai Chen, Darion Haase, Joost-Pieter Katoen: Exact Bayesian Inference for Loopy Probabilistic Programs. Conditionally accepted for OOPSLA'24 (as enclosed in the zipfile).

## Contents

* Requirements
* Building the Docker image (Optional)
* Smoke test
* Replicating the results from the paper
* Running your own example
* Supported program syntax

## Requirements

* Install Docker (https://www.docker.com/get-started/) in case you do not have it yet.
* Experiments in the paper are carried out on a native 2,4GHz Intel i5 Quad-Core processor with 16GB RAM running MacOS Sonoma 14.0. Make sure to have similar specs when comparing timing results and consider differences running in a sanbox (docker).

## Building & loading the Docker image
1. Checkout the repository on the artifact-evaluation branch:
    ```bash
    git clone --single-branch --branch ae-oopsla https://github.com/LKlinke/prodigy.git
    ```
2. Locate the repository
   ```bash
   cd prodigy 
   ```
3. Building the docker image
   ```bash
   docker build -t prodigy:latest .
   ```


The structure of the artifact is as follows (`ls -l`).

```bash
/root/artifact
├── pgfexamples            
|   └── Table 4            # the pGCL Programs described in Table 4 of the paper (and variations of them)
|   └── Appendix           # the pGCL Programs described in Table 5 of the Appendix (and variations of them)
|   └── Misc               # Additional other example pGCL programs (not further described)
├── prodigy                # source code of Prodigy
├── tests                  # Prodigy unit tests
├── load_env.sh            # script for loading the virtual python environment
├── pyproject.toml         # Prodigy dependencies
├── reproduce_results.sh   # script for reproducing the results in the paper
└── ...
```

## Smoke test

For a quick test to see if everything works:

If you *did not* build the docker image yourself, load the provided docker image using

4. Loading the docker image
  ```bash
  docker load -i prodigy.tar
  ```

Then run the docker container

5. Run Prodigy via Docker:
   ```bash
   docker run -it prodigy
   ```
You shall see a welcome message by Prodigy and be directed into the Docker container (which can be exited via `exit`).


## Replicating the results from the paper

### Reproducible elements in the paper

All Prodigy timings measured in Table 4 and Table 5 in the appendix can be reproduced.

### Reproducing the results

6. Reproduce the results presented in the paper:
    ```bash
    ./reproduce_results.sh
    ```
> _Note:_ The script uses predefined backends. Prodigy currently supports `ginac` and `sympy`. The former enables our C++ backend based on the GiNaC package. The latter employs the python computer algebra package sympy. `ginac` is generally faster than `sympy`, however for computing queries on final distributions, the current implementation relies on `sympy`.

## Running your own example

### Loop-free programs

To experiment with Prodigy on a customized loop-free example, you can just write a pGCL program and invoke prodigy.
1. Create an example file, e.g., `nano myexample.pgcl`.
```bash=
nat a;
nat b;
nat c;
nat r1;
nat r2;
nat r3;


a := 3
b := 4
c := 5

r1 := geometric(1/2);
r2 := geometric(1/2);
r3 := geometric(1/2);

observe(r1 > a)
observe(r2 > b)
observe(r3 > c)
```

2. Invoke prodigy on your example.
```bash
python prodigy/cli.py main example.pgcl 
```

### Loopy Programs
To experiment with Prodigy on a customized example containing loops, you need to create two files: 1) a program consisting of a single `while`-loop and 2) a loop-free invariant program (the supported syntax is specified further below).

1. Open an editor by typing, e.g., `nano myexample.pgcl`.
2. Write a program with a single `while`-loop such as
    ```bash=
    nat n; nat x;
    while(n > 0) {
        n := n - 1;
        { x := 0; } [1/2] { x := 1; }
    }
    ```
3. The effect of the above program can be summarized as follows: If `n > 0` initially, then upon termination `n` is set to zero and `x` is randomly assigned either `0` or `1`. This can be formally verified by writing the following loop-free invariant program (e.g., `nano myexample_inv.pgcl`):
    ```bash=
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
This is a [LARK](https://github.com/lark-parser/lark) grammar.
```=
    start: declarations instructions queries

    declarations: declaration* -> declarations

    declaration: "bool" var                  -> bool
               | "real" var bounds?          -> real
               | "const" var ":=" expression -> const
               | "rparam" var                -> rparam
               | "nparam" var                -> nparam
               | "fun" var ":=" function     -> fun
               | "nat" var bounds?           -> nat
               | "dist" var                  -> dist

    bounds: "[" expression "," expression "]"

    function: "{" declarations instructions "return" rvalue "}"

    instructions: instruction* -> instructions

    queries: query* -> queries

    instruction: "skip"                                      -> skip
               | "abort"                                     -> abort
               | "while" "(" expression ")" block            -> while
               | "if" "(" expression ")" block "else"? block -> if
               | var ":=" rvalue                             -> assign
               | block "[" expression "]" block              -> choice
               | "tick" "(" expression ")"                   -> tick
               | "observe" "(" expression ")"                -> observe
               | "loop" "(" INT ")" block                    -> loop
               | "query" "{" instructions "}"                -> query

    query: "?Ex" "[" expression "]"                          -> expectation
               | "?Pr" "[" expression "]"                    -> prquery
               | "!Print"                                    -> print
               | "!Plot" "[" var ("," var)? ("," literal)?"]"-> plot
               | "?Opt" "[" expression "," var "," var "]"   -> optimize


    block: "{" instruction* "}"

    rvalue: expression
          | "infer" "{" function_call "}" -> infer
          | "sample" "{" var "}"          -> sample

    function_call: var "(" parameter_list? ")"

    parameter_list: positional_params "," named_params 
                  | positional_params 
                  | named_params

    positional_params: positional_param
                     | positional_params "," positional_param

    positional_param: rvalue

    named_params: named_param 
                | named_params "," named_param

    named_param: var ":=" rvalue

    literal: "true"  -> true
           | "false" -> false
           | INT     -> nat
           | FLOAT   -> real
           | "∞"     -> infinity
           | "\\infty" -> infinity

    var: CNAME
```