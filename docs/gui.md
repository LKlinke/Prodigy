# Prodigy web GUI and analysis API

`prodigy web` starts a local web server and opens the GUI in the browser. You write a pGCL
program, choose how each `while` loop is analysed, and get the output distribution and the
answers to the queries — each with what is *guaranteed* about it.

```sh
poetry install
poetry run prodigy web                    # http://localhost:8080
poetry run prodigy --engine ginac web     # GiNaC as default engine (usually much faster)
poetry run prodigy web --port 9000 --time-limit 300 --no-browser
```

The GUI needs no Node.js or internet connection; the editor (CodeMirror 6) and KaTeX are
vendored under `prodigy/web/static/vendor/`.

Prodigy is tested with Python 3.11 and 3.14. On Python 3.13 and later, `poetry install` compiles
numpy 1.26.4 from source (a few minutes; networkx 2.5, which `probably` pins, does not work
with numpy 2).

## What the GUI shows

* **Program** (left): editor with pGCL highlighting, syntax errors (and type-checker
  warnings) as you type. Above each loop a pill shows its strategy and, after a run, what it
  certified. After each `?Pr`/`?Ex` query the value appears inline.
* **Setup** (right): initial distribution, one strategy per loop with its parameters,
  analysis options (engine, conditioning, simplification, number of coefficients, time
  limit, EVT synthesis strategy, solver, recording of intermediate distributions).
* **Result** (right): a summary of the guarantee, the queries with sound intervals, the
  output distribution (closed form, mass, expectations, marginal bar charts and tables), the
  certificate of every loop, console output, and the full result as JSON.
* **Examples**: the programs in `pgfexamples/`, with a suggested setup derived from their
  first line (e.g. `check_equality` examples come with their invariant).

Keyboard: ⌘/Ctrl + Return analyses, ⌘/Ctrl + . (or Esc) stops. Every analysis runs in its
own process with a time limit and can be stopped at any time.

## Guarantees

Every loop strategy reports what is known about its result, and the session combines this
for the whole program:

| Kind | Meaning | Produced by |
| --- | --- | --- |
| exact | equal to the semantics | loop-free code; unrolling/iteration that reaches a fixed point or leaves no mass in the loop; an EVT fixed point with non-negative coefficients **and finite mass** |
| lower | coefficient-wise below the semantics; at most the *residual mass* r is unaccounted for | unroll N, iterate until mass captured, EVT iteration |
| upper | coefficient-wise above the semantics (Park's lemma) | invariant check (exact if the loop terminates almost surely), EVT invariants with non-negative coefficients |
| unknown | no guaranteed direction | mixing lower and upper approximations, unchecked side conditions |

For a lower bound, every probability query becomes an interval
μ_N(φ) ≤ μ(φ) ≤ μ_N(φ) + r **without any termination assumption**: the unaccounted mass may
terminate with any value, violate an observation, or never terminate. When conditioning on
observations, the residual is rescaled together with the distribution (division by
1 − known violation mass), and the interval stays sound. Expectations of monotone,
non-negative expressions get the lower bound only.

EVT invariants (fixed points I = G + ⟦body⟧(⟨I⟩_guard)) are only upper bounds in general:
an invariant with infinite mass can be a *spurious* fixed point. The GUI checks
non-negativity with the configured positivity heuristic and decides finiteness of the mass
for rational closed forms (Pringsheim: no pole of I(t,…,t) in (0, 1]); only then is the
result reported as exact. Example: for `while (x > 0) { {x := x-1} [1/3] {x := x+1} }`
from x = 1, invariant synthesis finds a fixed point with infinite mass and Prodigy's value
Pr[x = 0] = 1 is reported as an upper bound — the true value is 1/2. Unrolling 25
iterations gives Pr[x = 0] ∈ [0.4956, 1].

## Non-interactive analysis API

The while handler no longer asks questions itself. An `AnalysisSession`
(`prodigy/analysis/session.py`) carries one decision per loop; loops are identified by
their index in textual order (`prodigy outline FILE` lists them).

```python
from probably.pgcl import parse_pgcl
from prodigy.analysis.session import AnalysisSession

session = AnalysisSession(parse_pgcl(source), source=source,
                          decisions={"0": {"strategy": "unroll", "params": {"iterations": 50}}},
                          capture_output=True)
result = session.run().to_json()      # schema "prodigy/analysis-result@1"
```

From the command line:

```sh
prodigy analyze prog.pgcl --loop 0=unroll:iterations=50 --json
prodigy analyze prog.pgcl --loop '*=evt_synthesis'          # same strategy for all loops
prodigy analyze prog.pgcl --decisions decisions.json --output result.json
prodigy loop_strategies                                     # strategies and parameters
prodigy main prog.pgcl --loop 0=unroll:iterations=10        # interactive, with presets
```

`prodigy main` without `--loop` behaves exactly as before (same menu, same prompts).

### Adding a loop strategy

```python
from prodigy.analysis.loop_strategies import Bound, LoopOutcome, LoopStrategy, ParamSpec, register_loop_strategy

def run(task):            # task.instruction, task.distribution, task.param("n"), task.progress(i, n), ...
    ...
    return LoopOutcome(distribution, error_prob, kind=Bound.LOWER, residual=r,
                       certificate={"type": "my_rule", ...}, messages=["..."])

register_loop_strategy(LoopStrategy(
    name="my_rule", title="My rule", menu_text="My rule (CLI menu text)", soundness=Bound.LOWER,
    description="One or two sentences for the GUI.", run=run,
    params=(ParamSpec("n", "int", "Iterations", prompt="Iterations: ", default=10, minimum=0),)))
```

The strategy then appears in the CLI menu (after the six historic entries), in
`prodigy analyze`, and in the GUI's strategy menu with a form for its parameters.

## HTTP API (used by the GUI)

| Endpoint | |
| --- | --- |
| `GET /api/meta` | version, engines, loop strategies with parameter specs, limits |
| `POST /api/check` `{source}` | diagnostics `{severity, message, line, column, end_line, end_column}` and outline (variables, loops with ids and lines, queries) |
| `POST /api/analyze` | starts an analysis (`source, input_distribution, engine, normalize, simplification, decisions, coefficients, record_steps, synthesis_strategy, solver, time_limit`) and returns a job id |
| `GET /api/jobs/<id>?log_offset=n` | state (`running`, `done`, `timeout`, `cancelled`, `crashed`), new console output, progress per loop, result |
| `DELETE /api/jobs/<id>` | cancels the analysis |
| `GET /api/examples`, `GET /api/examples/<path>` | example library and suggested setup |

## Security and hosting

The server binds to localhost. API requests must be JSON and come from the page itself
(Host and Origin are checked), which keeps other web sites from driving a local server.

Prodigy parses distributions and invariants with SymPy, whose parser evaluates Python code.
**Do not expose the server to untrusted users without a sandbox** (container without
network and file system access, resource limits). For a hosted teaching instance:
`--host 0.0.0.0 --allowed-host prodigy.example.org --max-jobs 4 --time-limit 60
--memory-limit 2048` behind a reverse proxy, inside such a container.

## Design notes

The interface follows Apple's Human Interface Guidelines as far as they apply to a web
app (foundations: accessibility, colour, typography, layout, writing).

* Colours are tokens by role (`prodigy/web/static/app/app.css`), with light, dark and
  increased-contrast variants and no app-specific appearance switch. Text contrast on the
  window and content surfaces: text 16.8/15.5:1 (light) and 15.6/12.8:1 (dark), secondary
  text 5.1/4.7:1 and 6.6/5.4:1; status colours 4.95–6.8:1 (light) and 5.0–8.9:1 (dark).
* One colour, one meaning: green = exact, purple = bound, orange = no guarantee, red =
  error, blue = interaction. Status is never conveyed by colour alone (glyph ✓ ≥ ≤ ! and a
  word).
* The signature element is the *sandwich*: what is certain is drawn solid, what is only
  bounded is hatched — in the interval bar of each query and as the "?" bar of the
  residual mass in the marginal charts.
* System font for interface text (13 px body, macOS scale), monospace for code and exact
  values, KaTeX for closed forms. Controls are at least 28 px high.

## Reuse in VS Code

`prodigy/web/static/app/components/` (result view, charts, formatting) depends only on the
result JSON, the CSS variables of `result-view.css` and optionally KaTeX, so a VS Code
webview can import it unchanged and map the variables to `--vscode-*` theme colours.

## Updating the vendored libraries

See `prodigy/web/frontend/README.md` (`npm install && npm run build`).
