"""
Measures the FLINT rational kernel (prodigy.distribution.kernel, #417) against the engines on the operations of
the #348 benchmark set (benchmarks/suite.json). Not part of the test suite; the results are in docs/source/kernel.md.

The kernel is not wired into the analysis, so the programs cannot run on it. Instead each workload is the
sequence of generating-function operations the analysis performs on a loop of the suite, written once per
backend with the same semantics:

* ``kernel``: :class:`~prodigy.distribution.kernel.RationalGF` (FLINT ``fmpq_mpoly``, cancelled after every
  operation).
* ``sympy-cas``: the same operations on SymPy expressions with ``sympy.cancel`` after every iteration: SymPy's
  cost without Prodigy's case analysis (what step 2 of #417 would replace).
* ``sympy``, ``ginac``, ``symengine``: the engines' ``Distribution`` (``filter``, ``update``, arithmetic),
  i.e. what an unrolled loop costs in the analysis today, including the shared case analysis.

Workloads:

* ``unroll_st_petersburg``: ``while (stop == 0) {payout = 2*payout} [1/2] {stop = 1}``, 30 iterations
  (suite case ``st_petersburg``; sparse polynomials of degree 2^k).
* ``unroll_galton_watson``: the loop of ``galton_watson_critical``, 15 iterations (a geometric draw in the
  body: rational functions with denominators (2 - alive)^k).
* ``unroll_galton_watson_40``: the same with 40 iterations (how the cost grows with the size).
* ``unroll_random_walk``: the loop of ``random_walk``, 25 iterations from s = 1 (Laurent shifts s -> s +- 1).
* ``cancel_equal``: decide that the uncancelled sum of the parts of 12 Galton-Watson unrollings (the exit parts
  and the remaining part) equals its cancelled form (the core of an invariant or equivalence check).
* ``coefficients_100``: 100 Taylor coefficients of the generating function in ``alive`` after 15
  Galton-Watson iterations (with total = 1): a rational function with a denominator of degree 15.
* ``convert``: SymPy -> kernel -> SymPy of that function (the cost of using the kernel as a hook).

    timeout 600 uv run python benchmarks/kernel.py                  # all, table on stdout (run in background)
    timeout 120 uv run python benchmarks/kernel.py --only kernel    # one backend
    uv run python benchmarks/kernel.py --run unroll_galton_watson kernel   # one measurement (seconds, JSON)

Every measurement runs in a fresh process with a time limit (``--timeout``, default 60 s).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position,import-outside-toplevel
import sympy  # noqa: E402

from prodigy.distribution.kernel import RationalGF  # noqa: E402

BACKENDS = ("kernel", "sympy-cas", "sympy", "ginac", "symengine")
WORKLOADS = ("unroll_st_petersburg", "unroll_galton_watson", "unroll_galton_watson_40", "unroll_random_walk",
             "cancel_equal", "coefficients_100", "convert")
ENGINES = ("sympy", "ginac", "symengine")

PAYOUT, STOP, ALIVE, TOTAL, S, C = sympy.symbols("payout stop alive total s c")


# --- the loops in the kernel ---------------------------------------------------------------------------------

def kernel_st_petersburg(iterations: int = 30) -> RationalGF:
    """The exit mass plus the remaining part after `iterations` unrollings (kernel)."""
    names = ("payout", "stop")
    g = RationalGF.variable("payout", names)
    stop = RationalGF.variable("stop", names)
    doubled = RationalGF.variable("payout", names) ** 2
    exited = RationalGF.constant(0, names)
    for _ in range(iterations):
        inside = g.subs("stop", 0)  # stop is 0 or 1: the coefficient of stop^0
        exited = exited + (g - inside)
        g = (inside.subs("payout", doubled) + inside * stop) / 2
    return exited + g


def kernel_galton_watson(iterations: int = 15) -> RationalGF:
    """The remaining part after `iterations` unrollings (kernel)."""
    return galton_watson_parts(iterations)[-1]


def galton_watson_parts(iterations: int) -> List[RationalGF]:
    """The exit part of each of `iterations` unrollings, then the remaining part (kernel)."""
    names = ("alive", "total")
    alive, total = RationalGF.variable("alive", names), RationalGF.variable("total", names)
    geometric = 1 / (2 - alive)  # alive = alive + geometric(1/2)
    g, parts = alive, []
    for _ in range(iterations):
        inside = g - g.subs("alive", 0)  # alive > 0
        parts.append(g - inside)
        g = inside / alive * total * geometric  # alive - 1, total + 1, + geometric(1/2)
    return parts + [g]


def kernel_random_walk(iterations: int = 25) -> RationalGF:
    """The exit mass plus the remaining part after `iterations` unrollings (kernel)."""
    names = ("c", "s")
    s, c = RationalGF.variable("s", names), RationalGF.variable("c", names)
    step = (s + 1 / s) * c / 2
    g, exited = s, RationalGF.constant(0, names)
    for _ in range(iterations):
        inside = g - g.subs("s", 0)
        exited = exited + (g - inside)
        g = inside * step
    return exited + g


# --- the loops on SymPy expressions --------------------------------------------------------------------------

def sympy_st_petersburg(iterations: int = 30) -> sympy.Expr:
    g, exited = PAYOUT, sympy.Integer(0)
    for _ in range(iterations):
        inside = g.subs(STOP, 0)
        exited = sympy.cancel(exited + g - inside)
        g = sympy.cancel((inside.subs(PAYOUT, PAYOUT ** 2) + inside * STOP) / 2)
    return sympy.cancel(exited + g)


def sympy_galton_watson(iterations: int = 15) -> sympy.Expr:
    g = ALIVE
    for _ in range(iterations):
        inside = sympy.cancel(g - g.subs(ALIVE, 0))
        g = sympy.cancel(inside / ALIVE * TOTAL / (2 - ALIVE))
    return g


def sympy_random_walk(iterations: int = 25) -> sympy.Expr:
    g, exited = S, sympy.Integer(0)
    for _ in range(iterations):
        inside = sympy.cancel(g - g.subs(S, 0))
        exited = exited + (g - inside)
        g = sympy.cancel(inside * (S + 1 / S) * C / 2)
    return sympy.cancel(exited + g)


# --- the loops on the engines' Distribution ------------------------------------------------------------------

def _factory(engine: str):  # -> Type[CommonDistributionsFactory]
    if engine == "sympy":
        from prodigy.distribution.generating_function import SympyPGF
        return SympyPGF
    if engine == "ginac":
        from prodigy.distribution.fast_generating_function import ProdigyPGF
        return ProdigyPGF
    from prodigy.distribution.symengine_distribution import SymenginePGF
    return SymenginePGF


def _expr(text: str):  # -> Expr
    """A guard or an assignment of the form `var op value` or `var = var op value`."""
    from prodigy.pgcl.ast import Binop, BinopExpr, NatLitExpr, VarExpr
    operators = {"==": Binop.EQ, ">": Binop.GT, "+": Binop.PLUS, "-": Binop.MINUS, "*": Binop.TIMES}

    def atom(token: str):
        return NatLitExpr(int(token)) if token.isdigit() else VarExpr(token)

    tokens = text.split()
    if tokens[1] == "=":
        rhs = atom(tokens[2]) if len(tokens) == 3 else BinopExpr(operators[tokens[3]], atom(tokens[2]),
                                                                  atom(tokens[4]))
        return BinopExpr(Binop.EQ, VarExpr(tokens[0]), rhs)
    return BinopExpr(operators[tokens[1]], atom(tokens[0]), atom(tokens[2]))


def engine_st_petersburg(engine: str, iterations: int = 30):
    factory = _factory(engine)
    g = factory.from_expr("payout", "payout", "stop")
    exited = factory.zero("payout", "stop")
    guard, double, stop = _expr("stop == 0"), _expr("payout = 2 * payout"), _expr("stop = 1")
    for _ in range(iterations):
        inside = g.filter(guard)
        exited = exited + (g - inside)
        g = inside.update(double) * "1/2" + inside.update(stop) * "1/2"
    return exited + g


def engine_galton_watson(engine: str, iterations: int = 15):
    factory = _factory(engine)
    g = factory.from_expr("alive", "alive", "total")
    exited = factory.zero("alive", "total")
    guard, decrement, increment = _expr("alive > 0"), _expr("alive = alive - 1"), _expr("total = total + 1")
    for _ in range(iterations):
        inside = g.filter(guard)
        exited = exited + (g - inside)
        g = inside.update(decrement).update(increment) * "1/(2 - alive)"
    return g


def engine_random_walk(engine: str, iterations: int = 25):
    factory = _factory(engine)
    g = factory.from_expr("s", "s", "c")
    exited = factory.zero("s", "c")
    guard, up, down, count = _expr("s > 0"), _expr("s = s + 1"), _expr("s = s - 1"), _expr("c = c + 1")
    for _ in range(iterations):
        inside = g.filter(guard)
        exited = exited + (g - inside)
        g = (inside.update(up) * "1/2" + inside.update(down) * "1/2").update(count)
    return exited + g


# --- the workloads -------------------------------------------------------------------------------------------

def _cancel_inputs() -> tuple:
    """
    The exit parts and the remaining part of 12 Galton-Watson unrollings (each cancelled) summed without
    cancellation, and the cancelled sum, as SymPy expressions (the unrolled function equals its rewriting).
    """
    symbols = {"alive": ALIVE, "total": TOTAL}
    parts = galton_watson_parts(12)
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return sympy.Add(*[part.to_sympy(symbols) for part in parts]), total.to_sympy(symbols)


def _text(expression: sympy.Expr) -> str:
    """`expression` in the syntax all engines read (``^`` for powers)."""
    return str(expression).replace("**", "^")


def _coefficient_input() -> sympy.Expr:
    return kernel_galton_watson(15).subs("total", 1).to_sympy({"alive": ALIVE})


def prepare(workload: str, backend: str) -> Callable[[], object]:
    """The measured call of `workload` on `backend` (the inputs are built before)."""
    # pylint: disable=too-many-return-statements
    loops = {"unroll_st_petersburg": (kernel_st_petersburg, sympy_st_petersburg, engine_st_petersburg),
             "unroll_galton_watson": (kernel_galton_watson, sympy_galton_watson, engine_galton_watson),
             "unroll_random_walk": (kernel_random_walk, sympy_random_walk, engine_random_walk)}
    iterations = {"unroll_galton_watson_40": 40}.get(workload)
    if iterations is not None:
        workload = workload.rsplit("_", 1)[0]
    if workload in loops:
        kernel, cas, engine = loops[workload]
        if iterations is None:
            return kernel if backend == "kernel" else cas if backend == "sympy-cas" else lambda: engine(backend)
        if backend == "kernel":
            return lambda: kernel(iterations)
        if backend == "sympy-cas":
            return lambda: cas(iterations)
        return lambda: engine(backend, iterations)
    if workload == "cancel_equal":
        raw, cancelled = _cancel_inputs()
        if backend == "kernel":
            return lambda: RationalGF.from_sympy(raw) == RationalGF.from_sympy(cancelled)
        if backend == "sympy-cas":
            return lambda: sympy.cancel(sympy.together(raw - cancelled)) == 0
        factory = _factory(backend)
        texts = _text(raw), _text(cancelled)
        return lambda: factory.from_expr(texts[0], "alive", "total") == factory.from_expr(texts[1], "alive",
                                                                                           "total")
    if workload == "coefficients_100":
        function = _coefficient_input()
        if backend == "kernel":
            return lambda: [c.to_sympy() for c in RationalGF.from_sympy(function).coefficient_list("alive", 100)]
        if backend == "sympy-cas":
            from prodigy.distribution.rational_series import rational_taylor_coefficients
            return lambda: rational_taylor_coefficients(function, ALIVE, 100)
        from prodigy.distribution.coefficient_stream import marginal_stream
        dist = _factory(backend).from_expr(_text(function), "alive")
        return lambda: list(marginal_stream(dist, "alive", 100, incremental=False))
    if workload == "convert":
        function = kernel_galton_watson(15).to_sympy({"alive": ALIVE, "total": TOTAL})
        if backend == "kernel":
            return lambda: RationalGF.from_sympy(function).to_sympy()
        raise NotImplementedError("only the kernel converts")
    raise ValueError(f"unknown workload {workload}")


def measure(workload: str, backend: str) -> Dict[str, object]:
    """
    Seconds of one call (calls under 0.2 s are repeated and the fastest of up to 5 counts), and the result
    if it is a truth value (the verdict of ``cancel_equal``).
    """
    call = prepare(workload, backend)
    best = float("inf")
    value: object = None
    for _ in range(5):
        start = time.perf_counter()
        value = call()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        if elapsed > 0.2:
            break
    return {"seconds": best, "result": value if isinstance(value, bool) else None}


def check() -> None:
    """The kernel and the SymPy loops agree (so the workloads compute the same functions)."""
    pairs = [(kernel_st_petersburg(12), sympy_st_petersburg(12)), (kernel_galton_watson(8), sympy_galton_watson(8)),
             (kernel_random_walk(10), sympy_random_walk(10))]
    for kernel, cas in pairs:
        assert sympy.cancel(kernel.to_sympy() - cas) == 0, (kernel, cas)
    raw, cancelled = _cancel_inputs()
    assert sympy.cancel(raw - cancelled) == 0


def run_one(workload: str, backend: str, timeout: float) -> Optional[float]:
    """One measurement in a fresh process; None on a time-out, NaN if the backend fails or answers wrongly."""
    command = [sys.executable, str(Path(__file__).resolve()), "--run", workload, backend]
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None
    try:
        measured = json.loads(process.stdout.strip().splitlines()[-1])
        if measured["result"] is False:
            print(f"{workload}/{backend}: the equality is not shown", file=sys.stderr)
            return float("nan")
        return float(measured["seconds"])
    except (IndexError, ValueError, KeyError):
        print(f"{workload}/{backend} failed: {process.stderr[-400:]}", file=sys.stderr)
        return float("nan")


def _cell(seconds: Optional[float], kernel: Optional[float], timeout: float) -> str:
    if seconds is None:
        return f"> {timeout:.0f} s"
    if math.isnan(seconds):
        return "–"
    text = f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"
    if kernel and not math.isnan(kernel) and seconds != kernel:
        text += f" ({seconds / kernel:.0f}×)"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--run", nargs=2, metavar=("WORKLOAD", "BACKEND"), help="one measurement (internal)")
    parser.add_argument("--only", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.run:
        print(json.dumps(measure(*args.run)))
        return
    check()
    print(f"| workload | {' | '.join(args.only)} |")
    print(f"|---|{'---|' * len(args.only)}")
    for workload in args.workloads:
        results: Dict[str, Optional[float]] = {}
        for backend in args.only:
            if workload == "convert" and backend != "kernel":
                results[backend] = float("nan")
                continue
            results[backend] = run_one(workload, backend, args.timeout)
        kernel = results.get("kernel")
        cells: List[str] = [_cell(results[b], kernel, args.timeout) for b in args.only]
        print(f"| {workload} | {' | '.join(cells)} |", flush=True)


if __name__ == "__main__":
    main()
