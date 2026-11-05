# The released image (ghcr.io/lklinke/prodigy): the command line and the web GUI with the SymPy, GiNaC and
# SymEngine engines, as installed by pip install prodigy-ppl. Dockerfile.artifact builds the full
# environment for artifacts.
#   docker run -p 8080:8080 ghcr.io/lklinke/prodigy                        web GUI on http://localhost:8080
#   docker run -v "$PWD":/work ghcr.io/lklinke/prodigy main /work/prog.ppy   command line
FROM public.ecr.aws/docker/library/python:3.11-slim AS build

RUN pip install --no-cache-dir uv==0.11.32
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/prodigy
WORKDIR /src

# The dependencies exactly as locked in uv.lock, without the development tools;
# in a layer of their own, so that it stays cached when only the code changes
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY prodigy prodigy
COPY pgfexamples pgfexamples
# The version usually comes from the Git tag, but the build context has no .git: the release workflow
# and make docker-build pass it in (after the dependency layer, so that a new version keeps it cached)
ARG PRODIGY_VERSION=0+unknown
RUN SETUPTOOLS_SCM_PRETEND_VERSION="$PRODIGY_VERSION" uv sync --locked --no-default-groups --no-editable

FROM public.ecr.aws/docker/library/python:3.11-slim

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 PATH=/opt/prodigy/bin:$PATH
COPY --from=build /opt/prodigy /opt/prodigy
RUN useradd --create-home prodigy
USER prodigy
WORKDIR /home/prodigy

EXPOSE 8080
ENTRYPOINT ["prodigy"]
CMD ["web", "--host", "0.0.0.0", "--no-browser"]
