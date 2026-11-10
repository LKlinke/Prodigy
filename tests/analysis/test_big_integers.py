"""
#235: exact coefficients with more than 4300 digits (Python's default limit for integer strings).
"""
import sys
import threading

import pytest

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.session import AnalysisSession
from prodigy.pgcl.parser import parse_pgcl
from prodigy.util.integers import unlimited_int_digits

Engine = ForwardAnalysisConfig.Engine
SOURCE = "x: nat;\n{x = 1;} [1/2 ** 15000] {x = 2;}\nquery Pr[x == 1];\n"
"""Pr[x == 1] = 1/2^15000 has 4516 digits."""


@pytest.mark.parametrize("engine, simplification", [
    (Engine.SYMPY, False), (Engine.SYMPY, True), (Engine.GINAC, False), (Engine.SYMENGINE, False)])
def test_probability_with_more_than_4300_digits(engine, simplification):
    previous = sys.get_int_max_str_digits()
    config = ForwardAnalysisConfig(engine=engine, use_simplification=simplification)
    session = AnalysisSession(parse_pgcl(SOURCE, "ppy"), source=SOURCE, syntax="ppy", config=config,
                              capture_output=True)
    result = session.run().to_json()
    assert result["status"] == "ok", result["error"]
    with unlimited_int_digits():
        assert result["queries"][0]["value"] == f"1/{2 ** 15000}"
    assert sys.get_int_max_str_digits() == previous


def test_unlimited_int_digits_restores_the_limit():
    previous = sys.get_int_max_str_digits()
    with unlimited_int_digits():
        with unlimited_int_digits():
            assert sys.get_int_max_str_digits() == 0
        assert sys.get_int_max_str_digits() == 0  # still inside the outer block
    assert sys.get_int_max_str_digits() == previous


def test_unlimited_int_digits_with_threads():
    # the limit is global: it is restored only when the last block of any thread ends
    previous = sys.get_int_max_str_digits()
    inside, release = threading.Event(), threading.Event()

    def worker():
        with unlimited_int_digits():
            inside.set()
            release.wait(10)

    thread = threading.Thread(target=worker)
    thread.start()
    inside.wait(10)
    with unlimited_int_digits():
        pass
    assert sys.get_int_max_str_digits() == 0  # the worker's block is still open
    release.set()
    thread.join(10)
    assert sys.get_int_max_str_digits() == previous
