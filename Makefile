.PHONY: help install lint mypy test docs clean

.DEFAULT: help
help:
	@echo "Available targets: install, lint, mypy, test, docs, clean, format"

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

docker-build:
	docker build -t prodigy .

docker-run:
	docker run -ti prodigy

clean:
	rm -rf docs/build

format:
	uv run yapf -r -i prodigy/ tests/
	uv run isort prodigy/ tests/
