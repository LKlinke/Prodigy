FROM public.ecr.aws/docker/library/python:3.11-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Setup timezone
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

# setup git so uv can install the dependencies from GitHub
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
RUN apt install -y dos2unix

# Setup uv as python package manager
RUN apt install -y curl
RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /root/artifact
COPY . .

# Install the dependencies exactly as locked in uv.lock into ./.venv, with the image's Python
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
RUN uv sync --locked
RUN uv pip install z3-solver
RUN bash -c "dos2unix load_env.sh"
CMD ["bash", "--rcfile", "./load_env.sh", "-i"]