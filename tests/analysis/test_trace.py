"""The step trace of the forward analysis (issue #148): source positions, steps with their
distributions and guarantees, loop iterates, certificate checks in words, watches, hints and the
command-line flags built on it."""
import json

import pytest
from click.testing import CliRunner

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.loop_strategies import Bound
from prodigy.analysis.result import sym
from prodigy.analysis.session import AnalysisSession
from prodigy.analysis.trace import TraceLimits, Tracer, _LoopFrame, certificate_checks, parse_watch
from prodigy.cli import cli
from prodigy.pgcl import parse_pgcl
from prodigy.pgcl.ast.walk import Walk, walk_instrs
from prodigy.pgcl.positions import span_of

Engine = ForwardAnalysisConfig.Engine
ENGINES = [Engine.SYMPY, Engine.SYMENGINE, Engine.GINAC]

PPY = """# a coin is flipped until heads
x: nat = 1;
c: nat;
while (x == 1) {
    { x = 0; } [1/2] { c = c + 1; }
}
if (c > 2) {
    c = 2;
} else {
    skip;
}
observe(c < 2);
query Pr[c == 0];
"""

PGCL = """// a coin is flipped until heads
nat x;
nat c;
x := 1
while (x = 1) {
    {x := 0} [1/2] {c := c + 1}
}
if (c > 2) {
    c := 2
} else {
    skip
}
observe(c < 2)
?Pr[c = 0]
"""

# line of each top-level step and of the two branch steps, in the order of the trace
PPY_LINES = [2, 3, 4, 7, 8, 10, 12, 13]
PGCL_LINES = [4, 5, 8, 9, 11, 13, 14]


def _run(source, syntax, engine=Engine.SYMPY, decision="unroll:iterations=6", **kwargs):
    config = ForwardAnalysisConfig(engine=engine)
    session = AnalysisSession(parse_pgcl(source, syntax), source=source, syntax=syntax, config=config,
                              decisions={"0": decision}, record_steps=True, capture_output=True, **kwargs)
    return session.run().to_json()


# ---- source positions ------------------------------------------------------------------

def test_every_parsed_instruction_has_a_span_in_both_syntaxes():
    for source, syntax in ((PPY, "ppy"), (PGCL, "pgcl")):
        program = parse_pgcl(source, syntax)
        lines = source.split("\n")
        for ref in walk_instrs(Walk.DOWN, program.instructions):
            span = span_of(ref.val)
            assert span is not None, (syntax, ref.val)
            start = lines[span.line - 1][span.column - 1:]
            assert start and not start[0].isspace(), (syntax, start)
            assert span.end_line >= span.line


def test_spans_survive_block_lowering_and_switch():
    source = "n: nat = 3;\nwhile (n > 0) {\n    c: nat = n % 2;\n    n = n - 1 - c;\n}\n" \
             "switch (n) { case 0: { n = 1; } default: { skip; } }\n"
    program = parse_pgcl(source, "ppy")
    loop = program.instructions[1]
    assert (span_of(loop).line, span_of(loop).end_line) == (2, 5)
    assert [span_of(i).line if span_of(i) else None for i in loop.body] == [3, 4, None]  # c := 0 is generated
    assert (span_of(loop.body[0]).column, span_of(program.instructions[2]).line) == (5, 6)


def test_equality_and_printing_ignore_spans():
    first, second = parse_pgcl("nat x;\nx := 1", "pgcl"), parse_pgcl("nat x;\n\n\nx := 1", "pgcl")
    assert span_of(first.instructions[0]).line != span_of(second.instructions[0]).line
    assert first.instructions == second.instructions and str(first) == str(second)


# ---- steps -------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("syntax", ["ppy", "pgcl"])
def test_trace_steps_with_positions_distributions_and_guarantees(engine, syntax):
    source, lines = (PPY, PPY_LINES) if syntax == "ppy" else (PGCL, PGCL_LINES)
    result = _run(source, syntax, engine, watches=["Pr[c == 0]" if syntax == "ppy" else "Pr[c = 0]"])
    assert result["status"] == "ok", result["error"]
    steps = result["steps"]
    if syntax == "ppy":
        steps = [s for s in steps if s["kind"] != "normalize"]
    assert [s["line"] for s in steps if s["kind"] != "normalize"] == lines
    by_kind = {s["kind"]: s for s in steps}
    loop_step, if_step = by_kind["while"], by_kind["if"]
    assert loop_step["column"] == 1 and loop_step["end_line"] == loop_step["line"] + 2
    # before the loop all mass is at x = 1, c = 0; afterwards a lower bound with residual 1/64
    assert sym(loop_step["before"]["mass"]) == 1 and loop_step["before"]["bound"] == Bound.EXACT
    assert sym(loop_step["after"]["mass"]) == sym("63/64")
    assert loop_step["bound"] == Bound.LOWER and sym(loop_step["residual_mass"]) == sym("1/64")
    assert loop_step["after"]["marginals"]["c"]["rows"][0]["value"] == 0
    assert sym(loop_step["after"]["marginals"]["c"]["rows"][0]["prob"]) == sym("1/2")
    # the branches are nested steps on their sub-distributions
    then_step, else_step = [s for s in steps if s["parent"] == if_step["index"]]
    assert (then_step["branch"], else_step["branch"], then_step["depth"]) == ("then", "else", 1)
    assert sym(then_step["before"]["mass"]) == sym("7/64")
    assert sym(else_step["before"]["mass"]) == sym("7/8")
    # watches answer as a query at that place would, with the same interval rules
    observe_step = by_kind["observe"]
    watch = observe_step["watches"][0]
    # D = 1 - 15/64 = 49/64 after the observation; the residual 1/64 is rescaled to 1/49
    assert sym(watch["lower"]) == sym("32/49") and sym(watch["upper"]) == sym("33/49")
    assert watch["bound"] == Bound.LOWER
    assert sym(steps[0]["watches"][0]["value"]) == 1 and not then_step["watches"]
    assert result["queries"][0]["lower"] == watch["lower"]
    assert all(s["time"] >= 0 and s["size_after"] > 0 for s in steps)
    assert {h["kind"] for h in result["trace"]["hints"]} >= {"growth", "time"}


@pytest.mark.parametrize("engine", ENGINES)
def test_unrolling_iterates_shrink_the_residual(engine):
    result = _run(PPY, "ppy", engine, decision="unroll:iterations=6")
    loop = next(s for s in result["steps"] if s["kind"] == "while")["loop"]
    assert loop["strategy"] == "unroll" and loop["kind"] == Bound.LOWER and loop["iterations"] == 6
    iterates = loop["iterates"]
    assert [i["iteration"] for i in iterates] == list(range(7))
    inside = [sym(i["inside_mass"]) for i in iterates]
    assert inside == [sym(1) / 2 ** k for k in range(7)]
    terminated = [sym(i["distribution"]["mass"]) for i in iterates]
    assert terminated == [1 - m for m in inside]
    assert all(i["distribution"]["bound"] == Bound.LOWER for i in iterates)
    assert sym(iterates[-1]["distribution"]["residual_mass"]) == sym("1/64")
    assert [c["status"] for c in loop["checks"]] == ["info", "info"]


def test_iterates_are_thinned_evenly_and_keep_the_last():
    result = _run(PPY, "ppy", decision="unroll:iterations=30", trace_limits=TraceLimits(iterates=5, coefficients=0))
    loop = next(s for s in result["steps"] if s["kind"] == "while")["loop"]
    iterations = [i["iteration"] for i in loop["iterates"]]
    assert iterations[0] == 0 and iterations[-1] == 30 and len(iterations) <= 7
    assert all(i % loop["stride"] == 0 for i in iterations[:-1])


@pytest.mark.parametrize("engine", ENGINES)
def test_fixed_point_iteration_is_exact(engine):
    source = "x: nat;\nwhile (x < 3) {\n    x = x + 1;\n}\nquery Pr[x == 3];\n"
    result = _run(source, "ppy", engine, decision="unroll:iterations=10")
    loop_step = result["steps"][0]  # a declaration without a value is no instruction
    assert loop_step["kind"] == "while"
    assert loop_step["bound"] == Bound.EXACT
    assert loop_step["loop"]["iterates"][-1]["distribution"]["bound"] == Bound.EXACT
    assert {c["label"]: c["status"] for c in loop_step["loop"]["checks"]}["Residual mass"] == "passed"


@pytest.mark.parametrize("engine", ENGINES)
def test_evt_iteration_iterates(engine):
    result = _run(PPY.replace("observe(c < 2);\n", ""), "ppy", engine, decision="evt_iterate:iterations=4")
    loop = next(s for s in result["steps"] if s["kind"] == "while")["loop"]
    masses = [sym(i["distribution"]["mass"]) for i in loop["iterates"]]
    # the first application of Ψ only gives the input, nothing has left the loop yet
    assert masses == [0, sym("1/2"), sym("3/4"), sym("7/8")]
    assert sym(loop["iterates"][-1]["inside_mass"]) == sym("1/8")
    assert all(i["distribution"]["bound"] == Bound.LOWER for i in loop["iterates"])


@pytest.mark.parametrize("engine", ENGINES)
def test_certified_evt_invariant_lists_its_checks(engine):
    result = _run(PPY.replace("observe(c < 2);\n", ""), "ppy", engine,
                  decision="evt_invariant:invariant=(2*x + 1)/(2 - c)")
    assert result["status"] == "ok", result["error"]
    loop_step = next(s for s in result["steps"] if s["kind"] == "while")
    assert loop_step["bound"] == Bound.EXACT
    checks = loop_step["loop"]["checks"]
    assert [c["label"] for c in checks] == ["Fixed point", "Non-negative", "Finite mass"]
    assert all(c["status"] == "passed" for c in checks)
    # the check of the invariant is a sub-analysis: no steps from inside it
    assert all(s["depth"] == 0 or s["parent"] is not None for s in result["steps"])
    assert not loop_step["loop"]["iterates"]


@pytest.mark.parametrize("engine", ENGINES)
def test_refuted_invariant_explains_why_and_has_no_distribution(engine):
    result = _run(PPY.replace("observe(c < 2);\n", ""), "ppy", engine, decision="evt_invariant:invariant=1/(1-c)")
    assert result["status"] == "error"
    loop_step = next(s for s in result["steps"] if s["kind"] == "while")
    assert loop_step["after"] is None and loop_step["bound"] == Bound.UNKNOWN and loop_step["error"]
    assert loop_step["loop"]["failed"]
    check = loop_step["loop"]["checks"][0]
    assert check["status"] == "failed" and "is not a fixed point" in check["text"] and "Φ(I) − I" in check["text"]
    assert loop_step is result["steps"][-1] or all(s["index"] < loop_step["index"] for s in result["steps"][:-1])


def test_failed_subinvariant_names_the_negative_coefficient():
    result = _run(PPY.replace("observe(c < 2);\n", ""), "ppy", decision="evt_subinvariant:subinvariant=(x+1)/(1-c)")
    check = next(s for s in result["steps"] if s["kind"] == "while")["loop"]["checks"][0]
    assert check["status"] == "failed"
    assert check["text"].startswith("Φ(ν) − ν has the negative coefficient -1/2 at")


def test_park_invariant_is_an_upper_bound_not_exact():
    source = "nat x; nat c; nat temp;\nx := 1\nwhile (x = 1) { {x := 0} [1/2] {c := c + 1}; temp := 0 }\n?Pr[c = 0]\n"
    invariant = ("nat x; nat c; nat temp;\n"
                 "if (x = 1) { temp := geometric(1/2); c := c + temp; x := 0; temp := 0 } else { skip }\n")
    result = _run(source, "pgcl", decision={"strategy": "invariant", "params": {"invariant": invariant}})
    assert result["status"] == "ok", result["error"]
    loop_step = result["steps"][1]
    assert loop_step["bound"] == Bound.UPPER and loop_step["after"]["bound"] == Bound.UPPER
    statuses = {c["label"]: c["status"] for c in loop_step["loop"]["checks"]}
    assert statuses == {"Fixed point": "passed", "Park's lemma": "info"}


def test_iterate_after_an_upper_bound_has_no_guarantee():
    session = AnalysisSession(parse_pgcl("nat x;", "pgcl"), record_steps=True)
    tracer = session.tracer
    dist = ForwardAnalysisConfig().factory.from_expr("x/2", "x")
    frame = _LoopFrame(task=None, entry={}, kind_before=Bound.UPPER, residual_before=0)
    tracer._record_iterate(frame, 1, dist, None, "1/2", Bound.LOWER)  # pylint: disable=protected-access
    snapshot = frame.iterates[0]["distribution"]
    assert snapshot["bound"] == Bound.UNKNOWN and snapshot["residual_mass"] is None
    frame = _LoopFrame(task=None, entry={}, kind_before=Bound.LOWER, residual_before="1/4")
    tracer._record_iterate(frame, 1, dist, None, "1/2", Bound.LOWER)  # pylint: disable=protected-access
    assert sym(frame.iterates[0]["distribution"]["residual_mass"]) == sym("3/4")


def test_large_closed_forms_are_cut_and_marginals_skipped():
    result = _run(PPY, "ppy", decision="unroll:iterations=6", trace_limits=TraceLimits(closed_form_chars=20))
    loop_step = next(s for s in result["steps"] if s["kind"] == "while")
    assert loop_step["after"]["truncated"] and len(loop_step["after"]["closed_form"]) == 20
    assert loop_step["after"]["size"] > 20 and not loop_step["after"]["marginals"]
    assert loop_step["after"]["marginals_omitted"] == "closed form too large"
    assert any(h["kind"] == "limits" for h in result["trace"]["hints"])


def test_without_record_steps_there_is_no_trace():
    session = AnalysisSession(parse_pgcl(PPY, "ppy"), source=PPY, syntax="ppy", decisions={"0": "unroll"})
    result = session.run().to_json()
    assert result["steps"] == [] and result["trace"] is None and session.tracer is None


def test_normalisation_at_the_end_is_a_step():
    result = _run(PPY.replace("query Pr[c == 0];\n", ""), "ppy")
    last = result["steps"][-1]
    assert last["kind"] == "normalize" and last["line"] is None
    assert sym(last["after"]["mass"]) == sym(result["distribution"]["mass"])


# ---- watches and checks --------------------------------------------------------------------

def test_parse_watch():
    assert parse_watch("Pr[x == 3]", "ppy").kind == "probability"
    assert parse_watch("query Ex[x + 1];", "ppy").kind == "expectation"
    assert parse_watch("?Pr[x = 3]", "pgcl").kind == "probability"
    assert parse_watch("E[x]", "pgcl").kind == "expectation"
    for text in ("Pr[x", "Foo[x]", "Pr[]", "Pr[__import__('os').system('true')]", "Pr[x = = 3]"):
        with pytest.raises(ValueError):
            parse_watch(text, "ppy")
    with pytest.raises(ValueError, match="not a variable"):
        parse_watch("Pr[y == 1]", "ppy", ["x"])


def test_certificate_checks_never_claim_unproven_facts():
    checks = certificate_checks({"type": "evt_invariant", "invariant": "I", "nonnegative": None, "finite_mass": None})
    assert [c["status"] for c in checks] == ["passed", "unknown", "unknown"]
    checks = certificate_checks({"type": "evt_invariant", "invariant": "I", "nonnegative": True, "finite_mass": False})
    assert checks[2]["status"] == "unknown" and "spurious" in checks[2]["text"]
    checks = certificate_checks({"type": "invariant", "verdict": "unknown", "difference": "d"},
                                failed="Could not determine whether invariant is valid.")
    assert checks[0]["status"] == "unknown"
    assert certificate_checks({}, failed="boom") == [{"label": "Strategy", "status": "failed", "text": "boom"}]


def test_tracer_ignores_silent_scopes():
    session = AnalysisSession(parse_pgcl("nat x;", "pgcl"), record_steps=True)
    with session.sub_analysis("check"):
        assert not session.tracer.wants(None)
    assert isinstance(session.tracer, Tracer) and session.tracer.wants(None)


# ---- command line ------------------------------------------------------------------------------

def _program_file(tmp_path, source=PPY, name="coin.ppy"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_analyze_steps_json_and_text(tmp_path):
    path = _program_file(tmp_path)
    runner = CliRunner()
    out = runner.invoke(cli, ["analyze", "--steps", "--json", "--watch", "Pr[c == 0]", "--loop", "0=unroll:iterations=4",
                              "--step-iterations", "3", path])
    assert out.exit_code == 0, out.output
    result = json.loads(out.output)
    loop = next(s for s in result["steps"] if s["kind"] == "while")["loop"]
    assert loop["iterates"][-1]["iteration"] == 4 and len(loop["iterates"]) <= 5
    assert result["trace"]["watches"] == [{"text": "Pr[c == 0]", "kind": "probability"}]
    out = runner.invoke(cli, ["analyze", "--steps", "--loop", "0=unroll:iterations=4", path])
    assert out.exit_code == 0, out.output
    assert "line 4: while (x == 1)\n" in out.output and "  line 8: [then] c = 2;" in out.output
    assert "Hint:" in out.output


def test_analyze_rejects_bad_watches(tmp_path):
    path = _program_file(tmp_path)
    out = CliRunner().invoke(cli, ["analyze", "--steps", "--watch", "Pr[y == 1]", "--loop", "0=unroll", path])
    assert out.exit_code == 2 and "not a variable" in out.output
    out = CliRunner().invoke(cli, ["analyze", "--watch", "Pr[c == 1]", "--loop", "0=unroll", path])
    assert out.exit_code == 2 and "--watch needs --steps" in out.output


def test_intermediate_results_and_stepwise_print_the_trace(tmp_path):
    path = _program_file(tmp_path)
    out = CliRunner().invoke(cli, ["--intermediate-results", "main", path, "--loop", "0=unroll:iterations=3"])
    assert out.exit_code == 0, out.output
    assert "line 2: x = 1;" in out.output and "line 4: while (x == 1)\n" in out.output
    out = CliRunner().invoke(cli, ["--stepwise", "main", path, "--loop", "0=unroll:iterations=3"], input="\n" * 20)
    assert out.exit_code == 0, out.output
    assert out.output.count("Next step [Enter]") >= 8
