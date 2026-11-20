"""
Case splits on distributions with parameters (#444). GiNaC put mass on impossible states (``y = x * x``), gave a
negative mass (``z = x % 2``) and a wrong ``Pr`` after ``observe``: its bindings list a state once per term of
the expanded polynomial and write a coefficient with parameters (a sum) without parentheses. The results are
compared as expressions with SymPy's and the exact distribution.
"""
import itertools

import pytest
import sympy

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.equality import decide_equal
from prodigy.pgcl import parse_pgcl

Engine = ForwardAnalysisConfig.Engine
ENGINES = [Engine.SYMPY, Engine.GINAC]

HEADER = "x: nat; y: nat; z: nat; param p: real; param q: real;\n"
CHOICE = "{ x = 1; } [p] { { x = 2; } [1/2] { x = 0; } }\n"

p, q, x, y, z = sympy.symbols("p q x y z")
HALF = sympy.Rational(1, 2)


def _run(source, engine):
    if engine == Engine.GINAC:
        pytest.importorskip("prodigy_ginac")
    source = HEADER + source
    session = AnalysisSession(parse_pgcl(source, "ppy"), source=source, syntax="ppy",
                              config=ForwardAnalysisConfig(engine=engine), decisions={}, capture_output=True)
    return session.run().to_json()


def _expr(text):
    parsed = sym(text)
    return parsed.xreplace({s: sympy.Symbol(s.name) for s in parsed.free_symbols})  # without assumptions


def _distribution(result):
    assert result["status"] == "ok", result.get("error")
    return _expr(result["distribution"]["closed_form"])


# --- the reproductions of the issue ---------------------------------------------------------------------------

REPRODUCTIONS = [
    pytest.param(CHOICE + "y = x * x;",
                 (1 - p) * HALF * x**2 * y**4 + p * x * y + (1 - p) * HALF, id="square"),
    pytest.param(CHOICE + "z = x % 2;",
                 (1 - p) * HALF * x**2 + p * x * z + (1 - p) * HALF, id="modulo"),
    pytest.param(CHOICE + "y = x * x;\nobserve(not (x == 2));",
                 (p * x * y + (1 - p) * HALF) / (p + (1 - p) * HALF), id="observe"),
    pytest.param("{ x = 1; } [p] { x = 0; }\n{ x = x + 1; } [q] { x = x; }\ny = x * x;",
                 (1 - p) * (1 - q) + ((1 - p) * q + p * (1 - q)) * x * y + p * q * x**2 * y**4, id="two_parameters"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("source, expected", REPRODUCTIONS)
def test_reproductions_are_exact(engine, source, expected):
    assert decide_equal(_distribution(_run(source, engine)), expected) is True


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("update", ["y = x * x;", "z = x % 2;"])
@pytest.mark.parametrize("observation", ["not (x == 2)", "not (x * x == 4)"])
def test_probability_after_observe(engine, update, observation):
    # GiNaC gave (2-p)/(2+p) after `z = x % 2` and -p/(1+p) after observe(not (y == 4))
    observation = observation.replace("x * x", "y") if update.startswith("y") else observation
    result = _run(CHOICE + f"{update}\nobserve({observation});\nquery Pr[x == 0];", engine)
    assert result["status"] == "ok", result.get("error")
    assert decide_equal(_expr(result["queries"][0]["value"]), (1 - p) / (1 + p)) is True


@pytest.mark.parametrize("engine", ENGINES)
def test_no_negative_mass_after_modulo(engine):
    # the state x = 2 got the mass -p/2 on GiNaC
    dist = sympy.expand(_distribution(_run(CHOICE + "z = x % 2;", engine)))
    assert decide_equal(dist.coeff(x, 2).subs(z, 0), (1 - p) * HALF) is True


def test_approximated_case_split_with_parameters_is_not_implemented():
    # the bindings would count the value 0 of the approximated marginal once per term (`1` and `-p`)
    pytest.importorskip("prodigy_ginac")
    # pylint: disable=import-outside-toplevel
    from prodigy.distribution.fast_generating_function import FPS
    geometric = FPS("(1-p)/(1-p*x)", "x")
    with pytest.raises(NotImplementedError, match="parameters"):
        geometric._update_product("y", "x", "x", "0.5")  # pylint: disable=protected-access


# --- a sweep of parametric programs: GiNaC agrees with SymPy -----------------------------------------------------

CHOICES = [
    CHOICE,
    "{ x = 1; } [p] { x = 0; }\n{ x = x + 1; } [q] { x = x; }\n",
    "{ x = 3; } [p] { { x = 2; } [q] { x = 1; } }\n",
    "{ x = 1; z = 2; } [p] { x = 0; }\n",
    "{ x = 2; } [1/3] { { x = 1; } [p] { x = 0; } }\n",
]
UPDATES = [
    "y = x * x;", "y = x % 2;", "y = x // 2;", "y = x * z;", "y = z % (x + 1);", "y = x * 3;", "y = x ** 2;",
    "y = 2 ** x;", "observe(not (x == 2));", "if (x == 1) { y = 5; } else { y = x + 1; }",
    "if (x >= 1) { { y = 1; } [q] { y = 2; } } else { y = 0; }", "y = x * x;\nobserve(y < 4);",
    "z = x % 2;\nobserve(z == 0);",
]


@pytest.mark.parametrize("source", [pytest.param(choice + update, id=f"choice{index}-{update}")
                                    for (index, choice), update in itertools.product(enumerate(CHOICES), UPDATES)])
def test_ginac_agrees_with_sympy(source):
    expected = _run(source, Engine.SYMPY)
    result = _run(source, Engine.GINAC)
    assert result["status"] == expected["status"], (result.get("error"), expected.get("error"))
    if expected["status"] == "ok":
        assert decide_equal(_distribution(result), _distribution(expected)) is True
