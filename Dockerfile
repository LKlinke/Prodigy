FROM public.ecr.aws/docker/library/python:3.11-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Setup timezone
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

# Add repository for older python versions on ubuntu.
RUN apt update
RUN apt install -y software-properties-common && apt update

# setup git so we can use it with poetry later
RUN apt install -y git
RUN git config --global user.name "Docker" && git config --global user.email "docker@example.com"

# install common text editors for convenience
RUN apt install -y vim nano

# Setup cmake for ginac development
RUN apt install -y build-essential
RUN apt install -y cmake 
RUN apt install -y make
RUN apt install -y graphviz
RUN apt install -y libginac-dev

# # Setup python environment
# RUN apt install -y python3-dev
# RUN apt install -y python3-pip
# RUN apt install -y python3-apt
# RUN apt install -y build-essential


# Setup poetry as python package manager
RUN apt install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 - && ln -s /root/.local/bin/poetry /usr/local/bin/poetry
# poetry will be installed by adding a line to .profile which is loaded by bash
# login shells. to access it, we'll need to wrap any calls to it with `bash -l -c`.

# for discoverability, add the virtual environment in the project directory instead of
# somewhere in ~/.cache
# ENV PATH=~/.local/share/pypoetry/:$PATH
RUN bash -l -c "poetry config virtualenvs.in-project true"



WORKDIR /root/artifact
COPY . .

RUN bash -l -c 'poetry env use 3.11 && poetry update && poetry install --no-interaction'

CMD ["bash", "-l", "/root/artifact/load_env.sh"]