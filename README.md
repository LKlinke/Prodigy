<img src="icon.svg" alt="Prodigy icon" width="96" align="right">

# prodigy: PRObability DIstributions via GeneratingfunctionologY

prodigy is a prototypical tool for the analysis of probabilistic non-negative integer programs with `while`-loops. It is based on (probability) generating functions.

It provides a command-line interface and a web GUI.

## Contents

1. Installation
2. Basic Usage


## Installation

### With pip

Prodigy is on PyPI as `prodigy-ppl` (the command and the Python package are called `prodigy`):

```bash
pip install prodigy-ppl
```

This needs Python 3.11 or newer and contains the SymPy, SymEngine and the faster GiNaC engine (`--engine ginac`).
The GiNaC engine comes from [`prodigy-ppl-ginac`](https://pypi.org/project/prodigy-ppl-ginac/), whose wheels bundle
GiNaC; pip installs it on Linux (x86_64, aarch64, glibc) and macOS with Python 3.11–3.14. Elsewhere, the extra `ginac`
builds it from source, which needs a C++17 compiler, CMake and [GiNaC](https://www.ginac.de/Download.html):

```bash
pip install "prodigy-ppl[ginac]"
```

### With Docker

The image contains the command line and the web GUI with the SymPy, GiNaC and SymEngine engines, for amd64 and arm64
(tags: version, e.g. `3.1.0`, minor version `3.1`, and `latest`). Without arguments it starts the web GUI on
http://localhost:8080:

```bash
docker run --rm -p 8080:8080 ghcr.io/lklinke/prodigy
```

Arguments are passed to `prodigy`; mount the folder with your programs to use the command line:

```bash
docker run --rm -v "$PWD":/work ghcr.io/lklinke/prodigy main /work/example.ppy
```

`Dockerfile.artifact` builds the full environment for artifacts (all engines including GiNaC, the sources and
the development tools): `make docker-artifact`, then `docker run -ti prodigy-artifact`.

### From source

To install Prodigy from the repository, follow these steps:

1. Make sure [Python](https://www.python.org/downloads/) 3.11 or newer is installed (tested with 3.11 and 3.14).
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Clone or download the Prodigy repository, open a command prompt in its root folder, and type `./install.sh`. This installs Prodigy with the dependency versions locked in `uv.lock` into `.venv`, and the Z3 solver for pysmt.

If no errors occurred, you have successfully installed Prodigy.
Now there are two options on how to use the tool. You can either use the command-line interface (CLI) or the web GUI. The next section (_Basic Usage_) details how to achieve both of these things.

### Troubleshooting

If you run into trouble during installation, this subsection might be of help. It lists some common problems that might occur during installation of Prodigy and how you can (hopefully) solve them.

- If you're using Windows, don't (not even WSL). We heavily recommend using either Linux or MacOS, as some of the required libraries have compatibility problems with Windows.
- If pip builds `prodigy-ppl-ginac` from source (no wheel for your platform, e.g. Alpine Linux), install GiNaC and make sure all of these packages are installed (if you're not using a Debian/Ubuntu based distribution, these might differ slightly):
  - `CMake`
  - `pkg-config`
  - `build-essential` (includes `make`, `g++`, `gcc`, and more)
  - `python3-dev` (called `python3-devel` on some other package managers such as `yam`) 

## Basic Usage
### CLI

1. In the project directory, run prodigy with ``uv run prodigy ...`` (or activate the virtual environment with ``source .venv/bin/activate`` and type ``prodigy ...``).
2. For more information, see the help message (``uv run prodigy --help``) or have a look into the [documentation](https://lklinke.github.io/Prodigy/).


### Web GUI
1. Type ``uv run prodigy web`` (or ``uv run prodigy --engine ginac web`` for the faster GiNaC engine). The GUI opens in your browser.
2. Write or load a program, choose a strategy for each loop under *Setup*, and press *Analyze* (⌘/Ctrl + Return).

Results come with their guarantee (exact, lower bound with residual mass, upper bound), and
queries are answered with sound intervals. See [docs/gui.md](docs/gui.md), which also
describes the non-interactive analysis API (`prodigy analyze --json`).

## Development

- `make lint`, `make mypy` and `make test` run the checks of the CI (pylint, mypy, and the tests with coverage; see `.github/workflows/main.yml`).
- Releases: push a tag `v<version>` (e.g. `v3.1.0`) on `main`; the package takes its version from the latest such tag (hatch-vcs, commits after it get a `.devN` version), so neither `pyproject.toml` nor `uv.lock` changes. `.github/workflows/release.yml` runs the checks, waits for approval in the environment `release`, and publishes to PyPI, the Docker image (`Dockerfile`) to ghcr.io, and a GitHub release.
- Dependencies are locked in `uv.lock`. After changing them in `pyproject.toml`, run `uv lock` and commit both files. `uv lock --upgrade` updates all dependencies within their constraints; Dependabot proposes such updates monthly as pull requests.
- `make docs` builds the documentation (Sphinx, `docs/source`) into `docs/build/html`: the API reference from the docstrings, the command-line reference from the click commands, and `docs/gui.md`. Warnings fail the build. `.github/workflows/docs.yml` builds it for every pull request; a release publishes it on [GitHub Pages](https://lklinke.github.io/Prodigy/).
