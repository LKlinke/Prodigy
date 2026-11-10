.. _pgcl_grammar:

pGCL grammar (``.pgcl``)
========================

This page is the reference for the original pGCL syntax, which Prodigy still reads (files that do
not end in ``.ppy``). Programs are written in ``.ppy`` now (see :doc:`language`);
``prodigy convert`` translates a ``.pgcl`` program into it. The notes on division, guards, modulo,
floor division, block-scoped declarations and sampling below describe the semantics and hold for
both syntaxes.

The `Lark <https://lark-parser.readthedocs.io/>`_ grammar that :mod:`prodigy.pgcl.parser` is
built from. Expressions are parsed with the operator table of
:mod:`prodigy.pgcl.util.lark_expr_parser`, which is appended as the rule ``expression``.

.. literalinclude:: _generated/pgcl.lark
   :language: text

Division
--------

``/`` is rational division everywhere; it never rounds. Rounding division is ``//`` (``.ppy``
only, see "Floor division" below).

- In a guard, ``x/2 <= 1`` means ``x <= 2``, not ``floor(x/2) <= 1`` (which would also hold for
  ``x = 3``).
- An assignment to a ``nat`` variable requires the result to be a natural number in every state,
  e.g. ``x := x/2`` only where ``x`` is always even. Otherwise the update is rejected; it is never
  rounded.

Subtraction
-----------

``-`` on natural numbers is truncated subtraction (monus): ``a - b = max(a - b, 0)``, in every
intermediate result of assignments, guards and queries, e.g. ``(n - 7) + 7`` is ``7`` for
``n = 5``. The reason and the consequences are described in :doc:`language-design` (section
"Subtraction"). An assignment ``t := a - b`` splits the distribution into the states with
``a < b``, which get ``0``, and the others, which get the difference. If the difference is provably
never negative (:mod:`prodigy.distribution.negative_values`), nothing is split. The split is exact
whenever the guard ``a < b`` can be filtered: always if ``b`` (or ``a``) is a constant, and for two
variables if one of them has a finite marginal. Otherwise (two variables with infinite marginals
and a difference that can be negative) the analysis ends with "not implemented". A guard, ``Pr[…]``
or ``Ex[…]`` with a subtraction is computed through fresh variables ``t := a - b``, exactly as an
assignment; a subtraction that involves ``/``, a constant that is not a natural number or a
parameter is rational subtraction.

Guards
------

Comparisons in guards (``if``, ``while``, ``observe``) are compared over the rationals. Guards
with one variable can be filtered on every distribution, including infinite ones; other guards
need a variable with a finite marginal. Linear guards are first rewritten exactly into guards with
natural numbers only:

.. automodule:: prodigy.distribution.linear_guards
   :no-index:

Modulo
------

``x % q`` with a natural number ``q`` can be used on infinite distributions, both in guards
(``x % 2 = 0``) and in assignments. If the generating function is rational in ``x``, the residue
classes are computed exactly without roots of unity, and the results are rational again:

.. automodule:: prodigy.distribution.multisection
   :no-index:

Otherwise they are extracted with complex roots of unity, which is exact if the generating function
is single-valued in ``x``, e.g. ``exp(l*(x - 1))``. On other generating functions, e.g. after a loop
whose invariant contains a root, the instruction is rejected:

.. automodule:: prodigy.distribution.roots_of_unity
   :no-index:

Floor division
--------------

``e // q`` (``.ppy`` only; in ``.pgcl`` ``//`` starts a comment) rounds down, as in Python:
``7 // 2 == 3``. It is implemented for a natural expression ``e`` and a positive natural constant
``q`` (a literal or a ``const``); other divisors are rejected by the type checker. Guards and
assignments round alike, so ``x // 2 <= 1`` holds for ``x = 3``.

As an update, ``t := e // q`` splits the distribution into the residue classes of ``e`` modulo
``q`` (as for ``e % q`` above) and divides ``e - r`` by ``q`` exactly on the class of ``r``. This
is exact for finite marginals of ``e`` and for generating functions that are rational in ``e``; for
other infinite marginals (e.g. ``poisson``) the update is rejected. A guard or an expectation with
``e // q`` is computed through a fresh variable ``t := e // q``, which is marginalised out again.
Floor division is not supported in the bounds of a declaration or by the lower-bound strategies
(``evt_subinvariant``, ``evt_sandwich``), which reject such loops.

Block-scoped declarations
-------------------------

In ``.ppy``, a declaration inside a block (a loop body, a branch, a ``case``) declares a variable
visible to the end of that block:

.. automodule:: prodigy.pgcl.scope
   :no-index:

Sampling
--------

``x := D`` samples ``x`` from one of the predefined distributions ``bernoulli(p)``,
``geometric(p)``, ``poisson(p)``, ``logdist(p)``, ``binomial(n, p)``, ``negative_binomial(r, p)``,
``unif(a, b)`` (also ``unif_d``); all parameters are positional and required.
``negative_binomial(r, p)`` is the number of failures before the ``r``-th success, i.e.
``iid(geometric(p), r)``; ``r`` is a natural number (a ``const`` counts as one) and ``p`` a
probability in ``(0, 1]`` or an expression in parameters. Two forms take a distribution as
argument:

- ``x := iid(D, n)`` sets ``x`` to the sum of ``n`` independent samples of ``D``. ``D`` is a
  predefined distribution or a generating function in ``x``. ``n`` is a program variable or a
  natural number (a ``const`` counts as one); other expressions and parameters are rejected.
- ``x := sample_pgf(G, x)`` samples ``x`` from the generating function ``G`` in ``x``. The second
  argument must be the assigned variable; sampling in another variable is rejected.
