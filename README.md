# Prodigy: PRObability DIstributions via GeneratingfunctionologY

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/LKlinke/Prodigy/tree/oopsla24-artifact)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10451092.svg)](https://doi.org/10.5281/zenodo.10451092)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://github.com/LKlinke/Prodigy/blob/oopsla24-artifact/LICENSE)

Prodigy is a tool for inferring posterior distributions described by probabilistic integer programs potentially using `while`-loops. It is based on (probability) generating functions.

In case of loopy programs, given an almost-surely terminating loop `while(G) {B}` and a loop-free (specification) program `I` (also called _invariant_), Prodigy checks whether `while(G) {B}` and `I` are _equivalent_ programs, i.e., whether they yield the same output distribution on every possible input distribution.

For more technical details, please refer to our OOPSLA'24 paper:
> Lutz Klinkenberg, Christian Blumenthal, Mingshuai Chen, Darion Haase, Joost-Pieter Katoen. 2024. Exact Bayesian Inference for Loopy Probabilistic Programs using Generating Functions. Proc. ACM Program. Lang. 8, OOPSLA1, Article 127 (April 2024). https://doi.org/10.1145/3649844 (as enclosed in the artifact)

## Contents

* Requirements
* Building the Docker image (Optional)
* Smoke test
* Replicating the results from the paper
* Running your own example
* Supported program syntax

## Requirements

* Install Docker (https://www.docker.com/get-started/) in case you do not have it yet.
* Experiments in the paper are carried out on a native 2,4GHz Intel i5 Quad-Core processor with 16GB RAM running MacOS Sonoma 14.0. Make sure to have similar specs when comparing timing results and consider differences running in a sandbox (docker).

## Building & loading the Docker image

If you want to use the provided Docker image, you can skip this step.

0.  Build the Docker image from sources:
    1. Checkout the repository on the oopsla24-artifact branch:
        ```bash
        git clone --single-branch --branch oopsla24-artifact https://github.com/LKlinke/prodigy.git
        ```
    2. Locate the repository:
        ```bash
        cd prodigy 
        ```
    3. Build the docker image:
        ```bash
        docker build -t prodigy:latest .
        ```


The structure of the artifact is as follows (`ls -l`).

```bash
/root/artifact
├── pgfexamples            
|   └── loop_free          # the pGCL Programs described in Table 4 
|   └── loopy              # the pGCL Programs described in Table 5 
|   └── Misc               # Additional example pGCL programs
├── prodigy                # source code of Prodigy
├── tests                  # Prodigy unit tests
├── load_env.sh            # script for loading the virtual python environment
├── pyproject.toml         # Prodigy dependencies
├── benchmark.py           # script for reproducing the results in the paper
└── ...
```

## Smoke test

For a quick test to see if everything works:

If you *did not* build the docker image yourself, load the provided docker image using

1. Load the docker image:
   ```bash
   docker load -i prodigy.tar.gz
   ```

Then run the docker container:

2. Run Prodigy via Docker:
   ```bash
   docker run -it prodigy
   ```
You shall see a welcome message by Prodigy and be directed into the docker container (which can be exited via `exit`).


## Replicating the results from the paper

### Reproducible elements in the paper

All Prodigy timings measured in Table 4 and Table 5 can be reproduced.

### Reproducing the results

3. Reproduce the results presented in the paper by typing (all benchmarks take ~150mins):
    ```bash
    python benchmark.py
    ```
> _Note:_ The script uses predefined backends. Prodigy currently supports `ginac` and `sympy`. The former enables our C++ backend based on the GiNaC package. The latter employs the python computer algebra package SymPy. `ginac` is generally faster than `sympy`, however for computing queries on final distributions, the current implementation relies on `sympy`.

### Results by other tools

This artifact only includes the Prodigy tool.
The paper focuses on programs featuring unbounded loops and this artifact is intended to support this.
To also reproduce the (loop-free program) results of λ-PSI and Genfer presented in Table 4, obtain and build the tools according to their documentation:
    
- λ-PSI: Build from https://github.com/eth-sri/psi using commit `9db68ba9581b7a1211f1514e44e7927af24bd398`.
- Genfer: Build from https://github.com/fzaiser/genfer using commit `5911de13f16bc3c28703f1631c5c4847f9ebac9a`.

The Genfer repository also contains λ-PSI code for the benchmarks of Table 4 in the `benchmarks` folder.
Results for individual benchmarks can be obtained by timing:

- λ-PSI (symbolic): `./psi <benchmark>.psi`
- λ-PSI (dp): `./psi --dp <benchmark>.psi`
- Genfer (exact): `./genfer --rational <benchmark>.sgcl`

## Running your own examples

### Loop-free programs

To experiment with Prodigy on a customized loop-free example, you can just write a pGCL program (the supported syntax is specified further below) and invoke Prodigy.
1. Create an example file, e.g., `nano myexample.pgcl`:
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

2. Invoke Prodigy on your example:
    ```bash
    python prodigy/cli.py main myexample.pgcl 
    ```

3. In case you want to run it with a different backend:
    ```bash
    python prodigy/cli.py --engine ginac main myexample.pgcl
    ```

### Loopy Programs
To experiment with Prodigy on a customized example containing loops, you need to create two files: 1) a program consisting of a single `while`-loop and 2) a loop-free invariant program (see supported program syntax below).

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
    when prompted press `1` and confirm your input with `[ENTER]`. Now give the path to the invariant file:
    ```bash
    myexample_inv.pgcl
    ```
    and confirm your input with `[ENTER]` again.
    
    Similarly you can query probabilities by appending `?Pr[...]` to `myexample.pgcl`.

## Supported program syntax
This is a [LARK](https://github.com/lark-parser/lark) grammar:
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