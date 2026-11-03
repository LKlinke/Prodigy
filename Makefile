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

test:
	uv run pytest --cov=prodigy --cov-report html --cov-report term --junitxml=testreport.xml ${ARGS}

docs:
	uv run bash -c "cd docs && make html"

# The released image: command line and web GUI (Dockerfile)
docker-build:
	docker build -t prodigy .

docker-run:
	docker run --rm -p 8080:8080 prodigy

# The full environment for artifacts, with GiNaC and the sources (Dockerfile.artifact)
docker-artifact:
	docker build -f Dockerfile.artifact -t prodigy-artifact .

clean:
	rm -rf docs/build

format:
	uv run yapf -r -i prodigy/ tests/
	uv run isort prodigy/ tests/
