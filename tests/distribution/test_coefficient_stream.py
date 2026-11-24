"""
Streamed coefficients (#405) are exactly the coefficients of the existing computation: the same values,
coefficient by coefficient, on all engines (GiNaC where it is installed), for rational, exp/log and algebraic
generating functions (with infinite support: finite marginals are listed by their terms) and for bounds;
and they arrive in order (marginals) or by total degree (grids).
"""
import itertools

import pytest
import sympy

from prodigy.analysis.joint_grid import joint_grid
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import describe_distribution, sym, tidy, univariate_table
from prodigy.analysis.streaming import Batcher
from prodigy.distribution import known_convergent_series
from prodigy.distribution.coefficient_stream import doubling, marginal_stream, taylor_stream
from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.rational_series import rational_taylor_coefficients, rational_taylor_stream
from prodigy.web.jobs import _Sender, merge_stream


UNIVARIATE = [
    pytest.param("1/(3 - 2*x)", id="rational"),
    pytest.param("(1 - p)/(1 - p*x)", id="rational-parametric"),
    pytest.param("(x^3 + 1)/(2*(2 - x))", id="rational-numerator"),
    pytest.param("exp(2*x - 2)", id="exp"),
    pytest.param("log(1 - x/3)/log(2/3)", id="log"),
    pytest.param("1 - sqrt(1 - x)", id="algebraic"),
]
LIMIT = 24


def _reference(marginal, variable, limit):
    """The coefficients as they were computed before #405 (prodigy.analysis.result._series_coefficients)."""
    if type(marginal).__name__ == "FPS":
        return [prob for prob, _ in itertools.islice(iter(marginal), limit)]
    expr = sym(str(marginal)) if not isinstance(getattr(marginal, "_function", None), sympy.Basic) \
        else marginal._function  # pylint: disable=protected-access
    symbol = next((s for s in expr.free_symbols if s.name == variable), None)
    if symbol is None:
        return [expr] + [0] * (limit - 1)
    series = sympy.expand(sympy.series(expr, symbol, 0, limit).removeO())
    return [tidy(series.coeff(symbol, n)) for n in range(limit)]


def _same(left, right) -> bool:
    difference = sym(left) - sym(right)
    return difference == 0 or sympy.simplify(difference) == 0


@pytest.mark.parametrize("closed_form", UNIVARIATE)
@pytest.mark.parametrize("incremental", [True, False], ids=["streamed", "at-once"])
def test_streamed_coefficients_equal_the_existing_ones(factory, closed_form, incremental):
    dist = factory.from_expr(closed_form, "x")
    streamed = list(marginal_stream(dist.marginal("x"), "x", LIMIT, incremental=incremental))
    reference = _reference(dist.marginal("x"), "x", LIMIT)
    assert len(streamed) == LIMIT
    for n, (value, expected) in enumerate(zip(streamed, reference)):
        assert _same(value, expected), f"coefficient {n}: {value} != {expected}"
    # and the same on every engine as on SymPy
    symbolic = _reference(SympyPGF.from_expr(closed_form, "x").marginal("x"), "x", LIMIT)
    assert all(_same(value, expected) for value, expected in zip(streamed, symbolic))


@pytest.mark.parametrize("closed_form", [UNIVARIATE[0], UNIVARIATE[3]])
def test_streamed_rows_of_a_lower_bound_are_the_final_rows(factory, closed_form):
    # a coefficient of a lower bound with residual mass is a lower bound: it keeps "upper" while streaming
    dist = factory.from_expr(f"9/10 * ({closed_form})", "x", "y")
    events = []
    summary = describe_distribution(dist, limit=12, kind=Bound.LOWER, residual="1/10", stream=events.append,
                                    expectations=False)
    state = {}
    for event in events:
        merge_stream(state, event)
    assert state["bound"] == Bound.LOWER and state["residual_mass"] == "1/10"
    assert state["variables"] == ["x", "y"]
    for variable in ("x", "y"):
        streamed = state["marginals"][variable]
        assert streamed["done"]
        assert streamed["rows"] == summary["marginals"][variable]["rows"]
    assert all("upper" in row for row in state["marginals"]["x"]["rows"])


def test_rows_arrive_in_batches_in_order():
    ticks = itertools.count()
    events = []
    batcher = Batcher(events.append, "rows", {"kind": "coefficients", "variable": "x"},
                      clock=lambda: next(ticks) * 0.04)
    table = univariate_table(SympyPGF.from_expr("1/(2 - x)", "x"), "x", 20, stream=batcher)
    batcher.flush(done=True)
    assert len(events) > 3
    assert [row for event in events for row in event["rows"]] == table["rows"]
    assert events[-1]["done"]


@pytest.mark.parametrize("closed_form", [
    pytest.param("1/((2 - x)*(3 - y))", id="rational"),
    pytest.param("exp(x - 1) / (2 - y)", id="exp"),
    pytest.param("(1 - sqrt(1 - x*y)) * (x + y)/2", id="algebraic"),
])
def test_streamed_grid_equals_the_grid_and_grows_by_total_degree(factory, closed_form):
    dist = factory.from_expr(closed_form, "x", "y")
    events = []
    # as for the GUI's grid of an analysis result (run_joint): the algebraic closed form has radius 1 (#398)
    with known_convergent_series(True):
        grid = joint_grid(dist, "x", "y", x_window=(1, 4), y_window=(0, 3), kind=Bound.LOWER, residual="1/10",
                          stream=events.append)
    state = {}
    for event in events:
        merge_stream(state, event)
    assert state["grid"]["cells"] == grid["cells"]
    assert state["grid"]["bound"] == Bound.LOWER and state["grid"]["x_values"] == [1, 2, 3, 4]
    degrees = [r + c for event in events for r, c, _ in event["cells"]]
    assert degrees == sorted(degrees) and len(degrees) == 16
    # the cells are the Taylor coefficients of the closed form
    expr = sym(closed_form)
    x, y = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
    for r, j in enumerate(range(4)):
        for c, i in enumerate(range(1, 5)):
            derivative = sympy.diff(expr, x, i, y, j).subs({x: 0, y: 0})
            assert _same(grid["cells"][r][c]["prob"], derivative / (sympy.factorial(i) * sympy.factorial(j)))


def test_doubling_yields_complete_prefixes():
    orders = []

    def prefix(order):
        orders.append(order)
        return [sympy.Integer(n) for n in range(order)]

    slow = itertools.count(step=10.0)  # every expansion takes 10 s: never jump to the full order
    assert list(doubling(prefix, 37, clock=lambda: next(slow))) == list(range(37))
    assert orders == [4, 8, 16, 32, 37]
    orders.clear()
    assert list(doubling(prefix, 37, clock=lambda: 0.0)) == list(range(37))
    assert orders == [4, 37]
    orders.clear()
    assert list(doubling(prefix, 37, incremental=False)) == list(range(37))
    assert orders == [37]
    # before each expansion, what is computed so far can be sent (the rows of a step do not wait for the next one)
    seen = []
    stream = doubling(prefix, 20, clock=lambda: next(slow), on_wait=lambda: seen.append(len(produced)))
    produced = []
    for value in stream:
        produced.append(value)
    assert seen == [0, 4, 8, 16]


def test_rational_stream_is_the_recursion():
    x, p = sympy.symbols("x p")
    for function in (1 / (3 - 2 * x), (1 - p) / (1 - p * x) ** 2, (x ** 3 + 1) / (2 - x), (1 + x) ** 3):
        stream = rational_taylor_stream(function, x)
        assert list(itertools.islice(stream, 15)) == rational_taylor_coefficients(function, x, 15)
    assert rational_taylor_stream(sympy.exp(x), x) is None
    assert list(taylor_stream((1 + x) ** 2, x, 5)) == [1, 2, 1, 0, 0]
    assert list(taylor_stream(p, x, 3)) == [p, 0, 0]


def test_stream_events_are_never_dropped():
    sent = []

    class Connection:  # pylint: disable=too-few-public-methods
        def send(self, message):
            sent.append(message)

    sender = _Sender(Connection())
    for n in range(5):
        sender.progress({"kind": "coefficients", "variable": "x", "rows": [n]})
        sender.progress({"loop": 0, "strategy": "unroll", "done": n, "total": 10})
    assert [payload["rows"] for kind, payload in sent if payload.get("kind")] == [[n] for n in range(5)]


def test_upper_ends_of_lower_bounds_are_capped_at_one():
    # the interval of a coefficient of a lower bound is [p, min(1, p + r)] (#405)
    table = univariate_table(SympyPGF.from_expr("1/(2 - x)", "x"), "x", 3, kind=Bound.LOWER, residual="3/4")
    assert [(row["prob"], row["upper"]) for row in table["rows"]] == [("1/2", "1"), ("1/4", "1"), ("1/8", "7/8")]
    grid = joint_grid(SympyPGF.from_expr("1/((2 - x)*(2 - y))", "x", "y"), "x", "y", x_window=(0, 1),
                      y_window=(0, 0), kind=Bound.LOWER, residual="3/4")
    assert [cell["upper"] for cell in grid["cells"][0]] == ["1", "7/8"]
    # with a parameter, the sum stays
    symbolic = univariate_table(SympyPGF.from_expr("(1 - p)/(1 - p*x)", "x"), "x", 1, kind=Bound.LOWER, residual="1/2")
    assert sym(symbolic["rows"][0]["upper"]) == sym("3/2 - p")
