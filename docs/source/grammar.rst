.. _pgcl_grammar:

pGCL grammar
============

The `Lark <https://lark-parser.readthedocs.io/>`_ grammar that :mod:`prodigy.pgcl.parser` is
built from. Expressions are parsed with the operator table of
:mod:`prodigy.pgcl.util.lark_expr_parser`, which is appended as the rule ``expression``.

.. literalinclude:: _generated/pgcl.lark
   :language: text

Guards
------

Comparisons in guards (``if``, ``while``, ``observe``) are compared over the rationals. Guards
with one variable can be filtered on every distribution, including infinite ones; other guards
need a variable with a finite marginal. Linear guards are first rewritten exactly into guards with
natural numbers only:

.. automodule:: prodigy.distribution.linear_guards
   :no-index:
