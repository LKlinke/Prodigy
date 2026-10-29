# prodigy: PRObability DIstributions via GeneratingfunctionologY

prodigy is a prototypical tool for the analysis of probabilistic non-negative integer programs with `while`-loops. It is based on (probability) generating functions.

It provides a command-line interface and a web GUI.

## Contents

1. Installation
2. Basic Usage


## Installation

To install Prodigy, follow these steps:

1. Make sure [Python](https://www.python.org/downloads/) 3.11 or newer is installed (tested with 3.11 and 3.14).
2. Install [GiNaC](https://www.ginac.de/Download.html), which also requires [CLN](https://www.ginac.de/CLN/).
3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
4. Clone or download the Prodigy repository, open a command prompt in its root folder, and type `./install.sh`. This installs Prodigy with the dependency versions locked in `uv.lock` into `.venv`, and the Z3 solver for pysmt.

If no errors occurred, you have successfully installed Prodigy.
Now there are two options on how to use the tool. You can either use the command-line interface (CLI) or the web GUI. The next section (_Basic Usage_) details how to achieve both of these things.

### Troubleshooting

If you run into trouble during installation, this subsection might be of help. It lists some common problems that might occur during installation of Prodigy and how you can (hopefully) solve them.

- If you're using Windows, don't (not even WSL). We heavily recommend using either Linux or MacOS, as some of the required libraries have compatibility problems with Windows.
- If you're using Linux, make sure all of these packages are installed (if you're not using a Debian/Ubuntu based distribution, these might differ slightly):
  - `CMake`
  - `pkg-config`
  - `build-essential` (includes `make`, `g++`, `gcc`, and more)
  - `python3-dev` (called `python3-devel` on some other package managers such as `yam`) 

## Basic Usage
### CLI

1. In the project directory, run prodigy with ``uv run prodigy ...`` (or activate the virtual environment with ``source .venv/bin/activate`` and type ``prodigy ...``).
2. For more information, see the help message (``uv run prodigy --help``) or have a look into the documentation.


### Web GUI
1. Type ``uv run prodigy web`` (or ``uv run prodigy --engine ginac web`` for the faster GiNaC engine). The GUI opens in your browser.
2. Write or load a program, choose a strategy for each loop under *Setup*, and press *Analyze* (⌘/Ctrl + Return).

Results come with their guarantee (exact, lower bound with residual mass, upper bound), and
queries are answered with sound intervals. See [docs/gui.md](docs/gui.md), which also
describes the non-interactive analysis API (`prodigy analyze --json`).

## Development

- `make lint`, `make mypy` and `make test` run the checks of the CI (pylint, mypy, and the tests with coverage; see `.github/workflows/main.yml`).
- Dependencies are locked in `uv.lock`. After changing them in `pyproject.toml`, run `uv lock` and commit both files. `uv lock --upgrade` updates all dependencies within their constraints; Dependabot proposes such updates monthly as pull requests.


## Lower bounds for loops (occupation subinvariants)

The package `prodigy.analysis.lowerbounds` adds two analysis modes for programs with a single loop. They implement the
rules of the paper *Finite Mass Is All It Takes: Lower Bounds for Probabilistic Loops without Termination
Assumptions*. The occupation measure of a loop is what Prodigy calls its
expected visiting times (EVT). Both modes are loop strategies (`evt_subinvariant` and `evt_sandwich`): in the web GUI,
in `prodigy analyze --loop 0=evt_sandwich`, interactively in the while handler (options `[7]` and `[8]`), and as CLI
commands:

| command | while handler | what it does |
|---|---|---|
| `evt_lower_bound PROGRAM --invariant NU` | `[7]` | **Subinvariant mode.** Verifies that `NU` is an occupation subinvariant of *finite mass* (`NU <= Phi(NU)` coefficient-wise, and the total mass of `NU` is finite). Then `[not guard]*NU` is a lower bound on the output distribution of the loop, **without any termination assumption** (lower induction), and the missing mass is at most `\|Phi(NU)\| - \|NU\|`. |
| `evt_sandwich PROGRAM [--size N] [--lower R;P;B] [--upper R;P;B] [--per-query] [--json F]` | `[8]` | **Synthesis mode.** Synthesises a finite-mass subinvariant (lower bound) and a superinvariant (upper bound, Park induction) with eventually geometric templates by linear programming, checks both in exact arithmetic, and prints two-sided bounds for the `?Pr`/`?Ex` queries after the loop. |

Example (random walk with a step counter `c`, `pgfexamples/lower_bounds/sandwich/rw_counter_23.pgcl`):

```
python -m prodigy.cli evt_sandwich pgfexamples/lower_bounds/sandwich/rw_counter_23.pgcl
...
Probability of termination (after the epilogue): [0.9999958249, 1]
?Pr[c >= 20]: [0.01661845727, 0.0166226323]
?Pr[c = 11]: [0.01517383867, 0.01517383868]
?Ex[c]: [2.999481151, 3.008152854]
```

Template parameters (rates, periods, tail starts) are chosen automatically from a short numerical exploration of the
loop; `--lower`/`--upper` override them, e.g. `--lower "0,17/20;1,2;24,48"`. Every certificate is verified exactly, so
the parameters affect precision and running time but never soundness. `abort` is interpreted as divergence.
Conditioning is not supported: `evt_sandwich` rejects programs with `observe`, and the subinvariant mode rejects
`observe` in the loop body, because violated observations would otherwise be treated like non-termination.
Additional Python dependencies: `scipy` (HiGHS) and `python-flint`. Tests: `pytest tests/analysis/test_lowerbounds.py`.
The examples in `pgfexamples/lower_bounds/exact` are loops whose occupation measures are known in closed form, those in
`pgfexamples/lower_bounds/sandwich` loops for which the synthesis mode computes two-sided bounds.
