.PHONY: help install lint mypy test docs clean

.DEFAULT: help
help:
	@echo "Available targets: install, lint, mypy, test, docs, clean, format"

install:
	poetry install

lint:
	poetry run pylint prodigy ${ARGS}

mypy:
	poetry run mypy prodigy ${ARGS}

test:
	poetry run pytest --doctest-modules --cov=prodigy --cov-report html --cov-report term --junitxml=testreport.xml tests/ prodigy/ ${ARGS}

docs:
	poetry run bash -c "cd docs && make html"

docker-build:
	docker build -t prodigy .

docker-run:
	docker run -ti prodigy

clean:
	rm -rf docs/build

format:
	poetry run yapf -r -i prodigy/ tests/
	poetry run isort prodigy/ tests/
