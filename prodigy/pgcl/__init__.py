"""
================
``prodigy.pgcl``
================

Deprecated: the package was renamed to :mod:`prodigy.language` in v0.7.0 (#446). Importing ``prodigy.pgcl``
or one of its modules gives a ``DeprecationWarning`` and the module of :mod:`prodigy.language`
(``prodigy.pgcl.pgcl_checks`` is :mod:`prodigy.language.checks`, ``prodigy.pgcl.pgcl_operations`` is
:mod:`prodigy.language.operations`); ``parse_pgcl`` and ``compile_pgcl`` are
:func:`~prodigy.language.parser.parse_program` and :func:`~prodigy.language.compiler.compile_program`.
This alias is removed in v0.8.0.
"""
# pylint: disable=wildcard-import,unused-wildcard-import
import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType
from typing import Dict, Optional, Sequence

from prodigy.language import *
from prodigy.language import compiler as _compiler
from prodigy.language import parser as _parser

warnings.warn("prodigy.pgcl was renamed to prodigy.language in v0.7.0; the old name is removed in v0.8.0",
              DeprecationWarning, stacklevel=2)

_OLD = "prodigy.pgcl"
_NEW = "prodigy.language"
_RENAMED_MODULES: Dict[str, str] = {"pgcl_checks": "checks", "pgcl_operations": "operations"}

# the old function names, on the modules that old code imports them from
parse_pgcl = _parser.parse_program
setattr(_parser, "parse_pgcl", _parser.parse_program)
setattr(_compiler, "compile_pgcl", _compiler.compile_program)


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports ``prodigy.pgcl.<module>`` as the (same) module object of ``prodigy.language.<module>``."""

    def find_spec(self, fullname: str, path: Optional[Sequence[str]],
                  target: Optional[ModuleType] = None) -> Optional[importlib.machinery.ModuleSpec]:
        del path, target  # the alias does not depend on where the finder looks
        if not fullname.startswith(_OLD + "."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        parts = spec.name[len(_OLD) + 1:].split(".")
        parts[0] = _RENAMED_MODULES.get(parts[0], parts[0])
        return importlib.import_module(".".join([_NEW] + parts))

    def exec_module(self, module: ModuleType) -> None:
        """The module of :mod:`prodigy.language` is already executed."""


if not any(isinstance(finder, _Alias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
