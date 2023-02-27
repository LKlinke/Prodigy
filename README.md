# prodigy: PRObability DIstributions via GeneratingfunctionologY

prodigy is a prototypical tool for the analysis of probabilistic non-negative integer programs with `while`-loops. It is based on (probability) generating functions.

It provides a command-line interface as well as a webservice.

## Contents

1. Installation
2. Basic Usage


## Installation

To install Prodigy, follow these steps:

1. Make sure [Python](https://www.python.org/downloads/) 3.9 or newer is installed.
2. Install [GiNaC](https://www.ginac.de/Download.html), which also requires [CLN](https://www.ginac.de/CLN/).
3. Install [Poetry](https://python-poetry.org/docs/#installation).
4. Clone or download the Prodigy repository, open a command prompt in its root folder, and type `poetry install`. This should automatically install all required dependencies.

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
 - Prodigy also requires [Tk](https://tkdocs.com/tutorial/install.html), which comes with most Python installations, but not all.

## Basic Usage
### CLI

5. In the project directory, type ``poetry shell``. This activates the freshly installed virtual environment.
6. Now, you can use prodigy by executing the script via `python prodigy/cli.py ...`. For more information, see the help message or have a look into the documentation.


### Webservice
5. Activate the cvirtual environment by ``poetry shell``.
6. Type ``web-service``.
7. Open a browser and locate `localhost:8080`.

