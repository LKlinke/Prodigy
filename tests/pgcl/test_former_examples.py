"""
The examples in ``pgfexamples/`` were converted from ``.pgcl`` to ``.ppy`` with ``prodigy convert``.

``tests/pgcl/former_examples/`` keeps a representative set of the former ``.pgcl`` programs (same
relative paths). They keep the original syntax covered by real programs, and they show that the
conversion changed nothing:

* the converter turns each of them into a program with the same AST (up to renamed keywords);
* the ``.ppy`` example in ``pgfexamples/`` has the same AST as its former ``.pgcl`` version;
* analysing the ``.ppy`` example gives the results stored in ``results.json``, which were computed
  from the ``.pgcl`` version before the switch (``prodigy analyze FILE --json --coefficients 8
  --loop '*=unroll:iterations=15'``, i.e. with simplification).

If an example is changed on purpose, update or remove its former version here.
"""
import json
from pathlib import Path

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl.converter import convert_pgcl, rename
from prodigy.pgcl.parser import parse_pgcl
from prodigy.pgcl.ppy_parser import parse_ppy

FORMER = Path(__file__).parent / "former_examples"
EXAMPLES = Path(__file__).parents[2] / "pgfexamples"
PROGRAMS = sorted(FORMER.rglob("*.pgcl"))
RESULTS = json.loads((FORMER / "results.json").read_text(encoding="utf-8"))
# the only name the converter had to rename (a keyword in .ppy)
RENAMINGS = {"inference/loop_free/conditioning/monty_hall_nested": {"switch": "switch_"}}


def _name(path: Path) -> str:
    return path.relative_to(FORMER).with_suffix("").as_posix()


def _same(new, old):
    assert new == old
    assert repr(new) == repr(old)


def test_former_examples_exist():
    assert len(PROGRAMS) >= 15
    assert set(RESULTS) <= {_name(path) for path in PROGRAMS}


@pytest.mark.parametrize("path", PROGRAMS, ids=_name)
def test_converter_keeps_the_program(path: Path):
    source = path.read_text(encoding="utf-8")
    conversion = convert_pgcl(source)
    assert conversion.renamings == RENAMINGS.get(_name(path), {})
    _same(parse_ppy(conversion.text), rename(parse_pgcl(source, "pgcl"), conversion.renamings))


@pytest.mark.parametrize("path", PROGRAMS, ids=_name)
def test_example_is_the_converted_program(path: Path):
    example = EXAMPLES / path.relative_to(FORMER).with_suffix(".ppy")
    assert example.is_file(), f"{example} is missing"
    assert not example.with_suffix(".pgcl").exists()
    old = rename(parse_pgcl(path.read_text(encoding="utf-8"), path=path), RENAMINGS.get(_name(path), {}))
    _same(parse_pgcl(example.read_text(encoding="utf-8"), path=example), old)


def _sym(text):
    return sympy.sympify(text.replace("^", "**")) if text is not None else None


def _equal(new, old):
    """Results as Prodigy prints them; generating functions are compared as expressions."""
    if new == old:
        return True
    try:
        return sympy.simplify(_sym(new) - _sym(old)) == 0
    except (sympy.SympifyError, TypeError, AttributeError):
        return False


@pytest.mark.parametrize("name", sorted(RESULTS))
def test_example_gives_the_former_results(name: str):
    expected = RESULTS[name]
    example = EXAMPLES / f"{name}.ppy"
    source = example.read_text(encoding="utf-8")
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy", config=ForwardAnalysisConfig(use_simplification=True),
                              decisions={"*": "unroll:iterations=15"}, coefficient_limit=8, capture_output=True)
    result = session.run().to_json()
    assert result["status"] == expected["status"], result.get("error")
    assert result["bounds"]["kind"] == expected["bound"]
    assert _equal(result["distribution"]["closed_form"], expected["closed_form"])
    assert _equal(result["distribution"]["error_probability"], expected["error_probability"])
    assert len(result["queries"]) == len(expected["queries"])
    for query, old in zip(result["queries"], expected["queries"]):
        assert (query["kind"], query["bound"]) == (old["kind"], old["bound"])
        for key in ("value", "lower", "upper"):
            assert _equal(query[key], old[key]), (key, query[key], old[key])
