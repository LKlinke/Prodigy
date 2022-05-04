Please note: This file is best viewed with a markdown editor.

# prodigy: PRObability DIstributions via GeneratingfunctionologY

prodigy is a prototypical tool for the analysis of probabilistic integer programs with `while`-loops. It is based on (probability) generating functions.

Given an almost-surely terminating loop `while(G) {B}` and a loop-free (specification) program `I` (also called _invariant_), prodigy checks whether `while(G) {B}` and `I` are _equivalent_ programs, i.e., they yield the same output distribution on every possible input distribution.

## Contents

1. Booting the VM
2. Smoke test
3. Reproducing the examples from the paper
4. Running your own example

## Booting the VM

Download the VM image and open it in your virtualization software. We have tested everything with VirtualBox (https://www.virtualbox.org), using 8GB of RAM.
The login credentials for the VM are user `CAV` and password `ae`.

## Smoke test

For a quick test to see if everything works you may execute the following steps:
* Open a terminal by right-clicking on the desktop.
* `cd prodigy/frontend/`
* Run `poetry shell`
* Run `python probably/cli.py`

You should see the output `Usage: cli.py [OPTIONS] COMMAND [ARGS]...` in the terminal.

## Reproducing the examples from the paper

Our example programs are located in the `~/Desktop/prodigy/frontend/pgfexamples` folder.
We now describe how you can run our tool on these examples.
* Open a terminal by right-clicking on the desktop.
* `cd prodigy/frontend/`
* Run `poetry shell`
* Execute `./run_equiv_examples.sh --engine prodigy` to run the single-loop examples from Section 6 and Appendix E of the paper. This should not take more than 1 minute.

> _Hint:_ You can exchange the computer algebra backend with the `--engine` option. Possible alternatives are `prodigy` and `sympy`. The former enables our C++ backend based on the GiNaC package. The latter employs the python computer algebra package sympy. `prodigy` is generally faster than `sympy`.

The examples involving more than one loop, i.e. nested (Example 5) or sequential (Example 11), require some more manual effort:
* Run `python probably/cli.py --engine prodigy main pgfexamples/nested_while.pgcl` for the nested loop example.
* When prompted, choose option _[1] Solve using invariants_ by typing `1` in the terminal
* Specify the outer loop's invariant by typing `pgfexamples/nested_while_inv_outer.pgcl`
* Choose option _[1] Solve using invariants_ again to specify the second invariant.
* Specify the inner loop's invariant by typing `pgfexamples/nested_while_inv_inner.pgcl`

Observe that in the case of nested loops, the two invariants are verified jointly once both have been specified.

Next consider the sequential loops example:
* Run `python probably/cli.py --engine prodigy main pgfexamples/sequential_loops.pgcl` for the sequential loops example.
* When prompted, choose option _[1] Solve using invariants_ by typing `1` in the terminal
* Specify the first loop's invariant by typing `pgfexamples/sequential_loops_first_inv.pgcl`
* Choose option _[1] Solve using invariants_ again to specify the second invariant.
* Specify the second loop's invariant by typing `pgfexamples/sequential_loops_second_inv.pgcl`



## Running your own example

To experiment with our tool on a custom example, you need to create two files: A program consisting of a single `while`-loop and a loop-free invariant/specification program (the supported syntax is specified further below).
* In folder `prodigy/frontend/` open an editor by typing e.g. `nano myexample.pgcl`
* Write a program with a single `while`-loop such as this:
    ```
    nat n; nat x;
    while(n > 0) {
        n := n - 1;
        { x := 0; } [1/2] { x := 1; }
    }
    ```
* The effect of the above program can be summarized as follows: If `n > 0` initially, then `n` is set to zero and `x` is randomly assigned either `0` or `1`. This can be formally verified by writing the following loop-free invariant program (open a new editor by typing `nano myexample_inv.pgcl`):
    ```
    nat n; nat x;
    if(n > 0) {
        { x := 0; } [1/2] { x := 1; }
        n := 0;
    } else {}
    ```
* Check using that your invariant is correct using
    ```
    python probably/cli.py --engine prodigy main myexample.pgcl 
    ```
    and when promted type `1` followed by your invariant file `myexample_inv.pgcl`.
* Optionally, you can append an input distribution in PGF notation to the above line, e.g. `n^5` (meaning that `n=5` initially with probability 1). The tool will then output the resulting distribution after the loop. In the above example, you should see `1/2 + 1/2*x` which means that `x=0` and `x=1` hold with probability `1/2` each.


### Supported Grammar

```
start: declarations instructions queries

declarations: declaration* -> declarations

    declaration: "bool" var                  -> bool
               | "nat" var bounds?           -> nat
               | "real" var bounds?          -> real
               | "const" var ":=" expression -> const
               | "rparam" var                -> rparam
               | "nparam" var                -> nparam
               
    bounds: "[" expression "," expression "]"
    instructions: instruction* -> instructions
    
    queries: query* -> queries
    
    instruction: "skip"                                      -> skip
               | "while" "(" expression ")" block            -> while
               | "if" "(" expression ")" block "else"? block -> if
               | var ":=" rvalue                             -> assign
               | block "[" expression "]" block              -> choice
               | "tick" "(" expression ")"                   -> tick
               | "observe" "(" expression ")"                -> observe
               | "loop" "(" INT ")" block                    -> loop
    
    query: "?Ex" "[" expression "]"                          -> expectation
               | "?Pr" "[" expression "]"                    -> prquery
               | "!Print"                                    -> print
               | "!Plot" "[" var ("," var)? ("," literal)?"]"-> plot
               | "?Opt" "[" expression "," var "," var "]"   -> optimize
               
               
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
