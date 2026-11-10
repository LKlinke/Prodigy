.PHONY: help install lint mypy test docs clean docker-build docker-run docker-artifact

.DEFAULT: help
help:
	@echo "Available targets: install, lint, mypy, test, docs, clean, format, docker-build, docker-run, docker-artifact"

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

test:
	uv run pytest ${XDIST} ${COV} --junitxml=testreport.xml ${ARGS}

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
