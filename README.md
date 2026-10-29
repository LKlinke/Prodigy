# prodigy: PRObability DIstributions via GeneratingfunctionologY

prodigy is a prototypical tool for the analysis of probabilistic non-negative integer programs with `while`-loops. It is based on (probability) generating functions.

It provides a command-line interface as well as a webservice.

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
Now there are two options on how to use the tool. You can either use the command-line interface (CLI) or use the webservice frontend. The next section (_Basic Usage_) details how to achieve both of these things.

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

### Webservice (legacy)
1. Type ``uv run web-service``.
2. Open a browser and locate `localhost:8080`.

## Development

- `make lint`, `make mypy` and `make test` run the checks of the CI (pylint, mypy, and the tests with coverage; see `.github/workflows/main.yml`).
- Dependencies are locked in `uv.lock`. After changing them in `pyproject.toml`, run `uv lock` and commit both files. `uv lock --upgrade` updates all dependencies within their constraints; Dependabot proposes such updates monthly as pull requests.

