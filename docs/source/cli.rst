Command-line interface
======================

Program syntax
--------------

The commands that read programs (``analyze``, ``main``, ``outline``, ``check_equality``,
``independent_vars``, ``invariant_synthesis``, ``evt_lower_bound``, ``evt_sandwich``) read programs
in the ``.ppy`` syntax (see :doc:`language`), whatever the file extension.

.. note::

   The ``.pgcl`` syntax was **removed in v0.6.0**, together with ``prodigy convert``. A file ending
   in ``.pgcl`` or ``--syntax pgcl`` ends with the error "The .pgcl syntax was removed in v0.6.0.
   Convert the program to .ppy with Prodigy v0.5.x (`prodigy convert program.pgcl`, or Convert to
   .ppy in its GUI)."

   The Z3 solver is being rebuilt (#474): ``--solver z3`` ends with the error "the Z3 solver is being
   rebuilt (#474); use sympy". Use ``--solver sympy`` (the default).

   The SymEngine engine was **removed in v0.6.0**: ``--engine symengine`` ends with the error
   "the SymEngine engine was removed in v0.6.0; use sympy or ginac". Use ``--engine sympy`` (the
   reference, the default) or ``--engine ginac`` (the fast one); any other name is an error as well.

An invariant given as program text, with ``--loop ID=invariant:invariant=...`` or
``invariant_file=...``, is read in the syntax of the program. Generating functions (``--invariant``
of ``invariant_synthesis`` and ``evt_lower_bound``, the EVT strategies) are not program text and
keep their own syntax.

Output speaks the syntax of the program: queries, loop guards and certificates are written as
``x == 1 and y < 3``, in the summary of ``analyze``, the output of ``main`` and the JSON result
(whose key ``syntax`` states the syntax).

Syntax errors in ``.ppy`` programs are reported with line and column, the source line and a hint
for constructs of the old syntax, e.g.:

.. code-block:: text

   Error: dice.ppy: line 2, column 4: Unexpected '=', expected ... (assignments in .ppy are written 'x = e').
       x := 3;
          ^

Simplification of the shown result
----------------------------------

The option does not change the analysis, which always uses the same normal form (closed forms with
factored numerator and denominator, expanded polynomials): it is fast to compute with, but often not
the shortest form to read. Only the result that is printed (``main``,
``analyze``; in the JSON result ``distribution.closed_form`` and ``distribution.latex``) is rewritten,
with bounded effort: the shortest of the exact rewrites ``together``, ``cancel`` and ``factor`` (tried in
this order, the GCD-based ones only below a size and degree threshold), each checked to be equal to the
computed form, within a time limit of 2 seconds (:mod:`prodigy.distribution.display`).
``distribution.display`` says which form is shown and why, and ``distribution.computed_form`` keeps the
computed form if it differs. ``prodigy --no-simplification ...`` prints the computed form. The option
changes no value, only its form.

Loop strategies
---------------

``prodigy loop_strategies`` lists the strategies for while loops in four groups, with the same titles
and one-sentence summaries as the interactive menu of ``prodigy main`` and the GUI (see
:doc:`gui`, "Choosing a loop strategy"):

- *Automatic*: ``auto`` (a short EVT invariant synthesis whose result is kept only if it is exact,
  otherwise unrolling ``iterations`` times);
- *Exact result*: ``invariant``, ``evt_invariant``, ``evt_synthesis``, ``ai_invariant``;
- *Bounds*: ``unroll``, ``mass_threshold``, ``evt_subinvariant``, ``evt_sandwich``;
- *Explore*: ``evt_iterate``, the iterates H_N of the expected visiting times (in the JSON result
  under the loop certificate's ``exploration``, with a guessed EVT invariant to check with
  ``evt_invariant``); the loop's result is only a lower bound.

A strategy is chosen per loop with ``--loop ID=STRATEGY[:NAME=VALUE,...]``, e.g.
``--loop '*=auto'`` or ``--loop 0=evt_iterate:iterations=20``. The names and the menu numbers of the
older strategies are unchanged.

Comparing two programs
----------------------

``prodigy check_equality A B`` checks whether the programs in the files ``A`` and ``B`` are
equivalent, i.e. have the same output distribution for every input (e.g. a loop and a loop-free
invariant program). The loops of both programs are decided interactively, as in ``prodigy main``.
The check runs both programs on a second-order generating function that stands for every input at
once (:mod:`prodigy.analysis.equivalence.equivalence_check`). Its verdict is the same as in the
GUI's Compare mode (:mod:`prodigy.analysis.equivalence.verdict`) and never claims more than was
proven:

- **equivalent** (exit code 0): the check proves equality and both analyses are exact.
- **not equivalent** (exit code 3): only with a *witness*, an input and an output state whose
  probabilities provably differ in the two programs (their sound intervals are disjoint). Prodigy
  searches it by analysing both programs again on the input of the check's counterexample (or on the
  input where every variable is 0), with the loop decisions given for the check.
- **undecided** (exit code 4): everything else, with the reason. If a loop was handled by an
  invariant (Park's lemma), its result is only an upper bound, and proven equality proves one
  direction only: the output reads ``one direction proven: first_below_second`` (the output
  distribution of A lies coefficient-wise below that of B, equal if the loops of A terminate almost
  surely, which Prodigy does not check). Programs that declare different variables, a step that the
  check cannot do ("not implemented") and a difference that the solver cannot decide end here too.

Exit codes 1 and 2 are errors as for every command (e.g. a syntax error or a missing file).

The output names both files, what the check found for the computed distributions with the guarantee
of each analysis (``exact``, ``lower``, ``upper``), the verdict and its reason, the parameter values
under which equality was proven, and the witness with the probability of its state in each program
(an interval for a bound), e.g.:

.. code-block:: text

   Program A: loop.ppy
   Program B: closed.ppy
   Check for every input: the computed distributions differ (analysis of A: lower, of B: exact)
   Verdict: A is not equivalent to B
   Reason: The programs give provably different probabilities to an output state (witness below).
   Witness: on the input every variable 0, the output state x = 0, c = 0 has probability
     in [1/2, 524289/1048576] in program A (lower)
     1/3 in program B (exact)

Stepping through an analysis
----------------------------

``prodigy analyze --steps FILE --loop ...`` records the *step trace* of the analysis
(:mod:`prodigy.analysis.trace`): for every instruction its line and column, the distribution
before and after it with the guarantee that holds there (exact, lower bound with residual mass,
upper bound, no guarantee), its mass, the first coefficients of its marginals, the time it took
and the size of the closed form. The instructions inside the branches of ``if`` and of
probabilistic choices are steps of their own, on the sub-distribution of their branch. A loop is
one step that tells what its strategy did: unrolling strategies keep their iterates (the part that
has left the loop grows, the mass still inside shrinks; ``--step-iterations`` iterates at most,
spread evenly, plus the last), invariant strategies the checks of their certificate in words,
including why a check failed. Two hints point to the step where the closed form grows most and
the step where most of the time goes.

Without ``--json`` the steps are printed one per instruction; with ``--json`` they are the keys
``steps`` and ``trace`` of the JSON result, for front ends such as the timeline of the GUI's analysis card.
``--watch 'Pr[x == 3]'`` (repeatable, in the program's syntax) evaluates a query after every
top-level step, as if it stood right after that instruction, with the same sound intervals as the
program's queries:

.. code-block:: text

   $ prodigy analyze --steps --watch 'Pr[c == 0]' --loop 0=unroll:iterations=8 coin.ppy
   line 2: x = 1;
       → x
       mass 1 · exact · 0.000138 s
       Pr[c == 0] = 1
   line 4: while (x == 1)
       → c**7/256 + c**6/128 + c**5/64 + c**4/32 + c**3/16 + c**2/8 + c/4 + 1/2
       mass 255/256 · lower bound, residual mass 1/256 · 0.0869 s
       loop 0 (unroll): 9 iterate(s) recorded, 8 iteration(s)
       ...

The interactive ``prodigy main`` prints the same steps while it runs with
``prodigy --intermediate-results main FILE``, and waits for Enter after each step with
``prodigy --stepwise main FILE``.

Commands
--------

.. click:: prodigy.cli:cli
   :prog: prodigy
   :nested: full
