FROM public.ecr.aws/docker/library/python:3.11-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Setup timezone
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

# setup git so we can use it with poetry later
RUN apt update && apt install -y git
RUN git config --global user.name "Docker" && git config --global user.email "docker@example.com"

# install common text editors for convenience
RUN apt install -y vim nano

# Setup cmake for ginac development
RUN apt install -y build-essential
RUN apt install -y cmake 
RUN apt install -y make
RUN apt install -y graphviz
RUN apt install -y libginac-dev

# Setup poetry as python package manager
RUN apt install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 - 
RUN ln -s /root/.local/bin/poetry /usr/local/bin/poetry
RUN poetry config virtualenvs.in-project true



WORKDIR /root/artifact
COPY . .

RUN poetry env use 3.11 
RUN poetry update
RUN poetry install --no-interaction

RUN bash -c "source ./.venv/bin/activate && pip install z3-solver"

CMD ["bash", "--rcfile", "./load_env.sh", "-i"]