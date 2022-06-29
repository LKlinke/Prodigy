# prodigy: PRObability DIstributions via GeneratingfunctionologY

prodigy is a prototypical tool for the analysis of probabilistic non-negative integer programs with `while`-loops. It is based on (probability) generating functions.

It provides a command-line interface as well as a webservice.

## Contents

1. Getting started
2. Test suite


## Getting Started

### Installation

First, you need to install the dependencies and register the project. To do so, run the following steps in order.
1. Once you have cloned (or downloaded) the repository, make sure you have installed `python` version 3.9 as well as `poetry`. More information on how to install `poetry` can be found [here](https://python-poetry.org/docs/#installation).
2. Open a terminal and change directory the project folder.
3. Type ``poetry install``

If no errors occured, you have successfully installed `prodigy`.
Now there are two options on how to use the tool. You can either use the command-line interface (CLI) or use the webservice frontend.

### CLI

4. In the project directory, type ``poetry shell``. This activates the freshly installed virtual environment.
5. Now, you can use prodigy by executing the script via `python prodigy/cli.py ...`. For more information, see the help message or have a look into the documentation.


### Webservice
4. You can start the webservice via ``poetry run web-service``
5. Open a browser and locate `localhost:8080`.

