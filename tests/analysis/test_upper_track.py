"""Proven per-state upper bounds of a loop carried to the result (#451, :mod:`prodigy.analysis.upper_track`)."""
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest
import sympy

import prodigy.analysis.lowerbounds.sandwich as sandwich_module
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.joint_grid import joint_grid
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.lowerbounds.parameters import TemplateShape
from prodigy.analysis.lowerbounds.sandwich import evt_sandwich
from prodigy.analysis.result import CERTIFICATE, RESIDUAL, upper_with_source, with_upper
from prodigy.analysis.session import AnalysisSession
from prodigy.analysis.upper_track import UpperTrack, normalized
from prodigy.distribution import known_convergent_series
from prodigy.language import parse_program

SANDWICH = Path(__file__).resolve().parents[2] / "pgfexamples" / "lower_bounds" / "sandwich"
# the biased walk with a step counter c; its output is known exactly (see exact_c)
LOOP = (SANDWICH / "rw_counter_23.ppy").read_text().split("query Pr", 1)[0]
# small fixed templates, so that the tests do not wait for the parameter search
LOWER = [TemplateShape([Fraction(0), Fraction(17, 20)], [1, 2], [6, 12], reach_box=[6, 14])]
UPPER = [TemplateShape([Fraction(1, 2), Fraction(9, 10)], [2, 2], [8, 16])]


def _fixed_sandwich(problem, **kwargs):
    return evt_sandwich(problem, lower_shapes=LOWER, upper_shapes=UPPER, verbose=False,
                        **{k: v for k, v in kwargs.items() if k not in ("lower_shapes", "upper_shapes", "verbose")})


def exact_c(k):
    """Pr[c = k] for the walk from 1 that steps down with 2/3: c = 2j + 1 with Catalan(j) paths."""
    if k % 2 == 0:
        return Fraction(0)
    j = (k - 1) // 2
    return Fraction(comb(2 * j, j), j + 1) * Fraction(2, 3) ** (j + 1) * Fraction(1, 3) ** j


def analyse(source, decisions, monkeypatch=None, engine=ForwardAnalysisConfig.Engine.SYMPY, coefficients=12):
    if monkeypatch is not None:
        monkeypatch.setattr(sandwich_module, "evt_sandwich", _fixed_sandwich)
    session = AnalysisSession(parse_program(source, "ppy"), source=source, syntax="ppy",
                              config=ForwardAnalysisConfig(engine=engine), decisions=decisions,
                              capture_output=True, coefficient_limit=coefficients)
    result = session.run().to_json()
    assert result["status"] == "ok", result.get("error")
    return result


def frac(value):
    return Fraction(str(sympy.nsimplify(sympy.sympify(value.replace("^", "**")))))


def rows(result, variable):
    return result["distribution"]["marginals"][variable]["rows"]


def query(result, expr):
    return next(q for q in result["queries"] if q["expr"] == expr)


def assert_sound_rows(table_rows, exact):
    """lower <= exact <= upper per row, and every upper end is min(U, lower + residual, 1)."""
    for row in table_rows:
        lower, upper = frac(row["prob"]), frac(row["upper"])
        assert lower <= exact(row["value"]) <= upper, row
        assert upper <= 1 and row["upper_source"] in (CERTIFICATE, RESIDUAL)


@pytest.mark.parametrize("engine", [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.GINAC])
def test_per_state_upper_ends_come_from_the_certificate(monkeypatch, engine):
    # loop-free code after the loop (if, assignment) is replayed on the upper bound
    source = LOOP + "if (c > 3) {\n    x = 1;\n}\nquery Pr[x == 1];\nquery Ex[c];\n"
    result = analyse(source, {"0": "evt_sandwich"}, monkeypatch, engine)
    assert result["bounds"]["kind"] == Bound.LOWER
    assert result["bounds"]["upper"]["origin"] == "loop 0 (evt_sandwich)"
    residual = frac(result["bounds"]["residual_mass"])
    c_rows = rows(result, "c")
    assert_sound_rows(c_rows, exact_c)
    # the certificate bounds the states far below lower + residual
    own = [row for row in c_rows if row["upper_source"] == CERTIFICATE]
    assert len(own) >= 6
    assert all(frac(row["upper"]) < frac(row["prob"]) + residual for row in own)
    # x = 1 exactly when c > 3
    above = 1 - exact_c(1) - exact_c(3)
    x_one = query(result, "x == 1")
    assert frac(x_one["lower"]) <= above <= frac(x_one["upper"])
    steps = query(result, "c")
    assert steps["upper"] is not None and steps["upper_source"] == CERTIFICATE
    assert frac(steps["lower"]) <= 3 <= frac(steps["upper"])  # E[c] = 1 / (2/3 - 1/3)


def test_observe_after_the_loop_gives_sound_normalised_intervals(monkeypatch):
    source = LOOP + "observe(c <= 7);\nquery Pr[c == 3];\nquery Pr[c == 7];\n"
    result = analyse(source, {"0": "evt_sandwich"}, monkeypatch)
    assert result["bounds"]["upper"] is not None
    valid = sum(exact_c(k) for k in range(8))  # the walk terminates almost surely

    def conditioned(k):
        return exact_c(k) / valid if k <= 7 else Fraction(0)
    assert_sound_rows(rows(result, "c"), conditioned)
    for k in (3, 7):
        answer = query(result, f"c == {k}")
        assert frac(answer["lower"]) <= conditioned(k) <= frac(answer["upper"])
    assert any(row["upper_source"] == CERTIFICATE for row in rows(result, "c"))


def test_the_exit_mass_of_a_walk_without_counter_is_bounded_through_observe():
    # terminates with probability 1/2; then Pr[y == 1 | observations] = (1/4) / (1 - 1/4)
    source = (SANDWICH / "escape_walk.ppy").read_text().replace("x: nat;", "x: nat;\ny: nat;").replace(
        "query Pr[true];", "{ y = 1; } [1/2] { y = 2; }\nobserve(y == 1);\nquery Pr[y == 1];\n")
    result = analyse(source, {"0": "evt_sandwich"})
    answer = query(result, "y == 1")
    assert frac(answer["lower"]) <= Fraction(1, 3) <= frac(answer["upper"])
    assert answer["upper_source"] == CERTIFICATE


def test_a_later_loop_ends_the_track(monkeypatch):
    source = LOOP + "while (x < 2) {\n    { x = x + 1; } [1/2] { skip; }\n}\nquery Ex[c];\n"
    result = analyse(source, {"0": "evt_sandwich", "1": "unroll:iterations=3"}, monkeypatch)
    assert result["bounds"]["upper"] is None
    assert all(row["upper_source"] == RESIDUAL for row in rows(result, "c"))
    assert query(result, "c")["upper"] is None


def test_no_track_from_a_loop_whose_input_is_only_a_lower_bound(monkeypatch):
    # the certificate bounds the output for the incoming lower bound, not for the exact input
    source = LOOP.replace("c = 0;", "c = 0;\nwhile (c < 1) {\n    { c = c + 1; } [1/2] { skip; }\n}") \
        + "query Ex[c];\n"
    result = analyse(source, {"0": "unroll:iterations=3", "1": "evt_sandwich"}, monkeypatch)
    assert result["bounds"]["kind"] == Bound.LOWER
    assert result["bounds"]["upper"] is None


def test_results_without_upper_certificate_are_unchanged():
    source = "x: nat;\nwhile (x < 3) {\n    { x = x + 1; } [1/2] { x = 0; }\n}\nquery Pr[x == 3];\n"
    result = analyse(source, {"0": "unroll:iterations=4"})
    assert result["bounds"]["kind"] == Bound.LOWER and result["bounds"]["upper"] is None
    residual = frac(result["bounds"]["residual_mass"])
    for row in rows(result, "x"):
        assert frac(row["upper"]) == min(frac(row["prob"]) + residual, 1)
        assert row["upper_source"] == RESIDUAL
    answer = query(result, "x == 3")
    assert frac(answer["upper"]) == frac(answer["lower"]) + residual
    assert answer["upper_source"] == RESIDUAL


def test_exact_results_have_no_upper_source():
    result = analyse("x: nat;\n{ x = 1; } [1/2] { x = 2; }\nquery Pr[x == 1];\n", {})
    assert query(result, "x == 1")["upper_source"] is None
    assert "upper_source" not in rows(result, "x")[0] and result["bounds"]["upper"] is None


def test_upper_with_source():
    assert upper_with_source("1/4", "1/2") == (sympy.Rational(3, 4), RESIDUAL)
    assert upper_with_source("1/4", "1/2", "1/3") == (sympy.Rational(1, 3), CERTIFICATE)
    assert upper_with_source("1/4", "1/2", "3/4") == (sympy.Rational(3, 4), RESIDUAL)  # a tie keeps the residual
    assert upper_with_source("3/4", "1/2", "2") == (sympy.Integer(1), RESIDUAL)
    assert upper_with_source("1/4", "1/2", "oo") == (sympy.Rational(3, 4), RESIDUAL)
    # a bound below the lower end contradicts the bounds and is not used
    assert upper_with_source("1/4", "1/2", "1/8") == (sympy.Rational(3, 4), RESIDUAL)
    # parameters: undecided, the residual end stays
    assert upper_with_source("p/4", "1/2", "p/3")[1] == RESIDUAL


def test_with_upper():
    assert with_upper(None, None) is None
    assert with_upper(None, 1) == (None, 1)
    assert with_upper((0, None), 1) == (0, 1)
    assert with_upper((0, sympy.Rational(1, 2)), sympy.Rational(1, 3)) == (0, sympy.Rational(1, 3))
    assert with_upper((0, sympy.Rational(1, 4)), sympy.Rational(1, 3)) == (0, sympy.Rational(1, 4))


@pytest.mark.parametrize("engine", [ForwardAnalysisConfig.Engine.SYMPY, ForwardAnalysisConfig.Engine.GINAC])
def test_normalisation_divides_by_the_largest_possible_valid_mass(engine):
    factory = ForwardAnalysisConfig(engine=engine).factory
    upper = factory.from_expr("x/2", "x")
    with known_convergent_series(True):
        track = UpperTrack(upper, factory.from_expr("1/2", "x"), "loop 0")
        # violation mass at most min(1/2, 1/8 + 1/8) = 1/4
        result = normalized(track, "1/8", "1/8")
        assert result is not None and result.error_prob.is_zero_dist()
        assert sympy.sympify(str(result.distribution.get_probability_mass())) == sympy.Rational(2, 3)
        # the violation mass may be all of it: no bound
        assert normalized(UpperTrack(upper, factory.from_expr("1", "x"), "loop 0"), "1", "0") is None


def test_joint_grid_cells_take_the_smaller_upper_end():
    factory = ForwardAnalysisConfig().factory
    with known_convergent_series(True):
        lower = factory.from_expr("x*y/2", "x", "y")
        upper = factory.from_expr("x*y/2 + x^2*y/4 + x^2*y^2/8", "x", "y")
        grid = joint_grid(lower, "x", "y", x_window=(0, 2), y_window=(0, 2), kind=Bound.LOWER,
                          residual="1/4", upper=upper)
    cells = grid["cells"]
    assert (cells[1][1]["upper"], cells[1][1]["upper_source"]) == ("1/2", CERTIFICATE)
    assert (cells[1][2]["upper"], cells[1][2]["upper_source"]) == ("1/4", RESIDUAL)  # a tie
    assert (cells[2][2]["upper"], cells[2][2]["upper_source"]) == ("1/8", CERTIFICATE)
    assert (cells[0][0]["upper"], cells[0][0]["upper_source"]) == ("0", CERTIFICATE)


def number(value):
    return sympy.sympify(value.replace("^", "**"))


@pytest.mark.slow
@pytest.mark.parametrize("example", sorted(path.name for path in SANDWICH.glob("*.ppy")))
def test_lower_below_upper_on_the_sandwich_examples(example):
    result = analyse((SANDWICH / example).read_text(), {"0": "evt_sandwich"})
    residual = number(result["bounds"]["residual_mass"])
    for table in result["distribution"]["marginals"].values():
        for row in table["rows"]:
            lower, upper = number(row["prob"]), number(row["upper"])
            assert lower <= upper <= sympy.Min(lower + residual, 1), row
            if row["upper_source"] == CERTIFICATE:
                assert result["bounds"]["upper"] is not None and upper < lower + residual
    for answer in result["queries"]:
        if answer["lower"] is not None and answer["upper"] is not None:
            assert number(answer["lower"]) <= number(answer["upper"]), answer
