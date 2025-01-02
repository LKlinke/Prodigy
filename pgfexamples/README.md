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
│   ├── loop_free                   # Programs without loops
│       └── invariants              # ... and their counterparts
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