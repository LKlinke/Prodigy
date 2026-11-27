"""
-------------
Type Checking
-------------

The type checker of the language. It rejects programs without a well-defined semantics, with a
:class:`CheckFail` that names the node and says why; what the analysis cannot compute is its business, not the
checker's (``docs/source/language-design.md``).

The checker is one small handler per node class, dispatched with :func:`functools.singledispatch`:

- :mod:`~prodigy.language.check.context`: :class:`CheckFail`, the context of a check and the dispatch functions;
- :mod:`~prodigy.language.check.expressions`: the type of each expression (:func:`get_type`);
- :mod:`~prodigy.language.check.instructions`: the check of each instruction (:func:`check_instr`);
- :mod:`~prodigy.language.check.calls`: calls of distributions and functions;
- :mod:`~prodigy.language.check.dist`: values of type ``dist`` in expressions and queries;
- :mod:`~prodigy.language.check.compatibility`: the implicit coercions ``bool ⊑ nat ⊑ real``;
- :mod:`~prodigy.language.check.program`: programs, functions, declarations and guards.
"""
from .calls import sample_predefined_functions
from .compatibility import is_compatible, type_name
from .context import CheckFail
from .dist import mentions_dist
from .expressions import get_type
from .instructions import check_instr
from .program import check_function, check_program, non_bool_guard

__all__ = ["CheckFail", "sample_predefined_functions", "get_type", "mentions_dist", "is_compatible", "check_instr",
           "check_program", "check_function", "type_name", "non_bool_guard"]
