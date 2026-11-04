.. _pgcl_grammar:

pGCL grammar
============

The `Lark <https://lark-parser.readthedocs.io/>`_ grammar that :mod:`prodigy.pgcl.parser` is
built from. Expressions are parsed with the operator table of
:mod:`prodigy.pgcl.util.lark_expr_parser`, which is appended as the rule ``expression``.

.. literalinclude:: _generated/pgcl.lark
   :language: text

Division
--------

``/`` is rational division everywhere; it never rounds. Floor division (``//``) does not exist
yet; it is planned in `#105 <https://github.com/LKlinke/Prodigy/issues/105>`_.

- In a guard, ``x/2 <= 1`` means ``x <= 2``, not ``floor(x/2) <= 1`` (which would also hold for
  ``x = 3``).
- An assignment to a ``nat`` variable requires the result to be a natural number in every state,
  e.g. ``x := x/2`` only where ``x`` is always even. Otherwise the update is rejected; it is never
  rounded.

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
(``x % 2 = 0``) and in assignments. The residue classes are extracted with complex roots of unity,
which is exact if the generating function is rational in ``x`` (more generally, single-valued, see
below). On other generating functions, e.g. after a loop whose invariant contains a root, the
instruction is rejected:

.. automodule:: prodigy.distribution.roots_of_unity
   :no-index:
