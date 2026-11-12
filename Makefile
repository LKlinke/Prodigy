.PHONY: help install lint mypy test fuzz bench docs clean docker-build docker-run docker-artifact

.DEFAULT: help
help:
	@echo "Available targets: install, lint, mypy, test, fuzz, bench, docs, clean, format, docker-build, docker-run, docker-artifact"

install:
	uv sync

lint:
	uv run pylint prodigy ${ARGS}

mypy:
	uv run mypy prodigy ${ARGS}

# Coverage options; `make test COV=` runs the tests without measuring coverage
COV = --cov=prodigy --cov-report html --cov-report term
# Parallel workers (pytest-xdist), one per CPU core; worksteal balances the few long tests best; `make test XDIST=` runs the tests in one process
XDIST = -n auto --dist worksteal

# The fuzzing and the comparison of the examples with the simulator (marker `fuzz`) run in `make fuzz`
test:
	uv run pytest ${XDIST} ${COV} -m "not fuzz" --junitxml=testreport.xml ${ARGS}

# Fuzzing and simulation tests (docs/source/testing.md); `PRODIGY_FUZZ=nightly make fuzz` runs many programs
fuzz:
	uv run pytest ${XDIST} -m fuzz tests/fuzzing tests/simulation ${ARGS}

# The benchmark set (benchmarks/suite.json) on every engine against benchmarks/baseline.json (docs/source/testing.md)
bench:
	uv run python benchmarks/run.py ${ARGS}

docs:
	uv run --group docs sphinx-build -M html docs/source docs/build -W --keep-going ${ARGS}

# The version from the Git tags (hatch-vcs); the Docker builds have no .git and get it as a build argument
VERSION = $(shell uvx hatch version 2>/dev/null || echo 0+unknown)

# The released image: command line and web GUI (Dockerfile)
docker-build:
	docker build --build-arg PRODIGY_VERSION=$(VERSION) -t prodigy .

docker-run:
	docker run --rm -p 8080:8080 prodigy

# The full environment for artifacts, with GiNaC and the sources (Dockerfile.artifact)
docker-artifact:
	docker build -f Dockerfile.artifact --build-arg PRODIGY_VERSION=$(VERSION) -t prodigy-artifact .

clean:
	rm -rf docs/build docs/source/_generated

format:
	uv run yapf -r -i prodigy/ tests/
	uv run isort prodigy/ tests/
