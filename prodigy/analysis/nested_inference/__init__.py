"""
Nested inference in the forward analysis: values of type ``dist``, ``infer`` and ``sample``.

The semantics is the one of ``docs/source/language-design.md`` ("Nested inference"):
``infer(f, a)`` is the distribution of the results of ``f(a)``, given that ``f(a)`` terminates and
passes its observations (the sub-distribution divided by its mass Z; Z = 0 is an error), and every
``sample(d)`` is an independent draw from ``d``.

Encoding
--------
A value of type ``dist`` is a *site*, :class:`DistSite`: a function and its arguments. The analysis
numbers the sites it meets (:class:`DistRegistry`, ids from 1), and a ``dist`` variable ``d`` is a
natural number in the joint generating function: the id of its site, 0 while it has no value.
An argument of a site is a natural number, another (closed) site, or a :class:`Slot`: the value
of a state-dependent argument, kept in the slot variables of ``d`` (``d_arg0__``, ...). A site
without slots is *closed*; its normalised generating function is computed once, exactly, when
``infer`` meets it (:meth:`DistRegistry.closed_form`).

The exact cases of the design are computed as follows:

- ``x = sample(d)``: split the distribution by the value of ``d`` (finitely many sites); for a
  closed site, marginalise ``x`` and multiply by G_d(x), as ``iid`` does. A site with slots exists
  only for a function without ``observe`` that terminates almost surely for every argument
  (Z ≡ 1, proven when ``infer`` creates it); then the draw is the call ``f(a)`` itself.
- ``d = infer(f, a)``: arguments without program variables are constants; arguments that depend
  on the state are split into their (finite) values, one closed site each; with infinite
  support they become slots, if Z ≡ 1 is proven, and are not implemented otherwise.
- ``d2 = d`` copies the id and the slots; a call ``g(d)`` passes them to the parameter.
- ``Pr[d == k]``, ``Ex[d]``: split by the value of ``d``, compute the number from G_d and
  substitute it into the instruction.
- A loop that reads ``dist`` variables but assigns none: split by their values, take them out of
  the distribution and bind them (:attr:`ProgramInfo.dist_bindings`) while the loop runs, so the
  loop strategies never see them. Assigning a ``dist`` variable inside a loop (a ``dist`` value
  in the loop's state) is not implemented.

Every other case ends with an :class:`~prodigy.analysis.exceptions.UnsupportedFeatureError` that
names the construct. At the queries and at the end of the program the ``dist`` variables are
taken out of the distribution; :func:`dist_values` describes their values.

Modules
-------
:mod:`~.sites` (sites, the registry, the encoding, splitting by values), :mod:`~.functions` (``infer``, ``sample``,
copies, calls on the joint distribution), :mod:`~.queries` (``Pr[d …]``, ``Ex[d]``), :mod:`~.loops` (loops that
read ``dist`` variables) and :mod:`~.values` (the values in the result and the trace). The package exports their
public names.
"""
from prodigy.analysis.nested_inference.sites import (VALUE_VARIABLE, Slot, DistSite, ClosedDist, DistRegistry,
                                                     slot_variable, dist_variables, encoding_variables, without,
                                                     strip_dist_variables, DistPart, split_by_value, split_by_values)
from prodigy.analysis.nested_inference.functions import (function_program_info, assign_return_value, closed_form, infer,
                                                         function_name, call_in_joint, sample, copy_dist)
from prodigy.analysis.nested_inference.queries import (query_value, resolve_queries)
from prodigy.analysis.nested_inference.loops import (assigned_dist_variables, run_loop)
from prodigy.analysis.nested_inference.values import (DistValue, dist_values, dist_values_json, for_trace)

__all__ = ["ClosedDist", "DistPart", "DistRegistry", "DistSite", "DistValue", "Slot", "VALUE_VARIABLE",
           "assign_return_value", "assigned_dist_variables", "call_in_joint", "closed_form", "copy_dist", "dist_values",
           "dist_values_json", "dist_variables", "encoding_variables", "for_trace", "function_name",
           "function_program_info", "infer", "query_value", "resolve_queries", "run_loop", "sample", "slot_variable",
           "split_by_value", "split_by_values", "strip_dist_variables", "without"]
