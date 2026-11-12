Command-line interface
======================

Program syntax
--------------

The commands that read programs (``analyze``, ``main``, ``outline``, ``check_equality``,
``independent_vars``, ``invariant_synthesis``, ``evt_lower_bound``, ``evt_sandwich``) accept both
surface syntaxes. The file extension decides: ``.ppy`` is read in the Python-like syntax (see
:doc:`language`), every other file in the original pGCL syntax (see :ref:`pgcl_grammar`). ``--syntax pgcl`` or
``--syntax ppy`` overrides the extension, e.g. for files without one. ``check_equality`` reads each
of its two files by its own extension, unless ``--syntax`` is given.

An invariant given as program text, with ``--loop ID=invariant:invariant=...`` or
``invariant_file=...``, is read in the syntax of the program. Generating functions (``--invariant``
of ``invariant_synthesis`` and ``evt_lower_bound``, the EVT strategies) are not program text and
keep their own syntax.

Output speaks the syntax of the program: queries, loop guards and certificates are written as
``x == 1 and y < 3`` for a ``.ppy`` program and as ``(x = 1) & (y < 3)`` for a ``.pgcl`` program, in
the summary of ``analyze``, the output of ``main`` and the JSON result (whose key ``syntax``
states the syntax).

Syntax errors in ``.ppy`` programs are reported with line and column, the source line and a hint
for constructs of the old syntax, e.g.:

.. code-block:: text

   Error: dice.ppy: line 2, column 4: Unexpected '=', expected ... (assignments in .ppy are written 'x = e').
       x := 3;
          ^

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

Converting programs to ``.ppy``
-------------------------------

``prodigy convert FILE.pgcl [-o FILE.ppy]`` translates a program from the old syntax into
``.ppy`` and prints it, or writes it to ``-o``. The result parses to the same program:

- Names that are keywords in ``.ppy`` (``switch``, ``case``, ``default``, ``def``, ``param``,
  ``print``, ``sample``, …) are renamed by appending ``_``, as often as needed to avoid a keyword or
  another name of the program, e.g. ``switch`` becomes ``switch_``. Every renaming is reported on
  stderr.
- Comments are kept. A comment on a line of its own is placed before the next statement or
  declaration, a comment after code after the statement that ends on its line; ``//`` comments
  become ``#`` comments. Since the converter works on the parsed program, a comment can move
  slightly, e.g. a comment at the end of a block moves behind the block. Empty lines that separate
  parts of the program are kept; the code itself is laid out anew.

The library function is :func:`prodigy.pgcl.converter.convert_pgcl`.

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
