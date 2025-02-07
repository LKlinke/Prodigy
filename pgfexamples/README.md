# Examples

The examples are structured as follows:

```
pgfexamples/
├── comparison                      # Comparisons with other tools / paradigms
│   ├── ADDcomparison               # Comparisons with Algebraic Decision Diagrams
│   └── psicomparison               # Comparisons with Psi
│       ├── inference
│       └── psi_fails               # Examples where Psi fails 
├── equivalence                     # Equivalence of two programs
│   ├── loop_free                   # Programs without loops and their counterparts
│   └── loopy                       # Programs with loops
│       └── invariants              # ... and their loop-free counterpart
├── independence                    # Checks for independent variables
├── inference                       # Calculation of posterior distribution
│   ├── loop_free                   # ... without while-loops
│   │   └── conditioning            # ... with observe-statements
│   └── loopy                       # ... with while-loops
│       └── conditioning            # ... with observe-statements 
├── invariant_synthesis             # Synthesis of (expected visiting time) invariants
└── template_parameter_synthesis    # Equality of loopy and loop-free programs with parameters
    └── invariants                  # Corresponding loop-free programs
```

## File structure

* Files have a comment in the first line indicating which arguments should be used in order to execute it

> ```bash
>python prodigy/cli.py METHOD FILE ARGS 
>```
> results in
>```
># METHOD FILE ARGS
>```
> in the first line of the corresponding program or just
> ```
> # skip
> ```
> if the file should not be automatically tested.

* Naming convention:
    * In `pgfexamples/equivalence/loop_free` the files are named `file_name.pgcl` for the first and
      `file_name2.pgcl` for the second program
    * In `pgfexamples/equivalence/loopy` and `pgfexamples/template_parameter_synthesis` the files are named
      `file_name.pgcl` in the main folder and their corresponding invariants with `file_name_invariant.pgcl` in the
      invariants folder
    * Invariants and second programs are to be marked with `skip`