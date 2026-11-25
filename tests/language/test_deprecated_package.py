"""The old package name ``prodigy.pgcl`` (#446): it imports the modules of ``prodigy.language`` with a
``DeprecationWarning`` until v0.8.0. Each case runs in a fresh interpreter, since the warning is given once per
process."""
import subprocess
import sys

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-W", "error::DeprecationWarning:__main__", "-c", code],
                          capture_output=True, text=True, timeout=60, check=False)


def test_the_old_name_warns():
    result = _run("import prodigy.pgcl")
    assert result.returncode != 0
    assert "prodigy.pgcl was renamed to prodigy.language" in result.stderr


@pytest.mark.parametrize("old, new", [
    ("prodigy.pgcl.ast", "prodigy.language.ast"),
    ("prodigy.pgcl.ast.walk", "prodigy.language.ast.walk"),
    ("prodigy.pgcl.util.ref", "prodigy.language.util.ref"),
    ("prodigy.pgcl.pgcl_checks", "prodigy.language.checks"),
    ("prodigy.pgcl.pgcl_operations", "prodigy.language.operations"),
])
def test_old_modules_are_the_new_ones(old, new):
    code = ("import warnings; warnings.simplefilter('ignore', DeprecationWarning)\n"
            f"import importlib; assert importlib.import_module({old!r}) is importlib.import_module({new!r})")
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_old_function_names():
    code = ("import warnings; warnings.simplefilter('ignore', DeprecationWarning)\n"
            "from prodigy.pgcl import parse_pgcl\n"
            "from prodigy.pgcl.compiler import compile_pgcl\n"
            "from prodigy.language import parse_program\n"
            "from prodigy.language.compiler import compile_program\n"
            "from prodigy.pgcl.ast import Program\n"
            "assert parse_pgcl is parse_program and compile_pgcl is compile_program\n"
            "assert isinstance(compile_pgcl('x: nat;\\nx = 1;\\n'), Program)")
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_the_new_name_does_not_carry_the_old_names():
    code = ("import prodigy.language, prodigy.language.compiler\n"
            "assert not hasattr(prodigy.language, 'parse_pgcl')\n"
            "assert not hasattr(prodigy.language.compiler, 'compile_pgcl')")
    result = _run(code)
    assert result.returncode == 0, result.stderr
