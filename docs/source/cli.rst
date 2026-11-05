Command-line interface
======================

Program syntax
--------------

The commands that read programs (``analyze``, ``main``, ``outline``, ``check_equality``,
``independent_vars``, ``invariant_synthesis``, ``evt_lower_bound``, ``evt_sandwich``) accept both
surface syntaxes. The file extension decides: ``.ppy`` is read in the Python-like syntax (see
:doc:`language-design`), every other file in the original pGCL syntax. ``--syntax pgcl`` or
``--syntax ppy`` overrides the extension, e.g. for files without one. ``check_equality`` reads each
of its two files by its own extension, unless ``--syntax`` is given.

An invariant given as program text, with ``--loop ID=invariant:invariant=...`` or
``invariant_file=...``, is read in the syntax of the program. Generating functions (``--invariant``
of ``invariant_synthesis`` and ``evt_lower_bound``, the EVT strategies) are not program text and
keep their own syntax.

Syntax errors in ``.ppy`` programs are reported with line and column, the source line and a hint
for constructs of the old syntax, e.g.:

.. code-block:: text

   Error: dice.ppy: line 2, column 4: Unexpected '=', expected ... (assignments in .ppy are written 'x = e').
       x := 3;
          ^

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

Commands
--------

.. click:: prodigy.cli:cli
   :prog: prodigy
   :nested: full
