# Examples

The examples are structured as follows:

```
pgfexamples/
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

The programs are written in the `.ppy` syntax (see the language reference in the documentation).
They were converted from the original pGCL syntax with `prodigy convert`; `.pgcl` programs are
still supported.

* Files have a comment in the first line indicating which arguments should be used in order to execute it

> ```bash
>prodigy METHOD FILE ARGS
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
    * In `pgfexamples/equivalence/loop_free` the files are named `file_name.ppy` for the first and
      `file_name2.ppy` for the second program
    * In `pgfexamples/equivalence/loopy` and `pgfexamples/template_parameter_synthesis` the files are named
      `file_name.ppy` in the main folder and their corresponding invariants with `file_name_invariant.ppy` in the
      invariants folder
    * If there are multiple invariant files for one program, e.g. for a nested loop, the invariants are to be marked
      with numbers indicating the order they should be used (i.e. the invariant for the outer-most while loop has number
      1 and so on). The equivalence test compares the program with `file_name_invariant.ppy`, the loop-free
      counterpart of the whole program (e.g. for sequential loops); for nested loops it may be missing, and the
      invariant of the outer-most loop (number 1) is used instead
    * A program that the equivalence test should skip is marked with `skip`, and the test states the reason
      (`LOOPY_SKIPPED` in `tests/analysis/equivalence/test_equivalence_check.py`)
    * Invariants and second programs are to be marked with `skip`