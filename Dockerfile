FROM ubuntu

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV POETRY_HOME=/etc/poetry

# Setup timezone
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

# Add repository for older python versions on ubuntu.
RUN apt update
RUN apt install -y zsh
SHELL ["bash", "-l", "-c"]
RUN apt install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt update


# Add tools for building prodigy with GiNaC dependencies on python 3.11
RUN apt install -y git
RUN apt install -y cmake
RUN apt install -y make
RUN apt install -y graphviz
RUN apt install -y curl
RUN apt install -y libginac-dev
RUN apt install -y python3.11-dev
RUN apt install -y build-essential
RUN apt install -y dos2unix
# setup git so we can use it with poetry later
RUN git config --global user.name "Test User" && git config --global user.email "you@example.com"

# Get poetry as a python package manager

RUN curl -sSL https://install.python-poetry.org | python3 -
# We need to export the path of poetry as otherwise the install script will not find the executable.
RUN echo 'export PATH="$PATH:/etc/poetry/bin"' >> ~/.zshrc
RUN echo 'export PATH="$PATH:/etc/poetry/bin"' >> ~/.bashrc


# poetry will be installed by adding a line to .profile which is loaded by bash
# login shells. to access it, we'll need to wrap any calls to it with `bash -l -c`.

# for discoverability, add the virtual environment in the project directory instead of
# somewhere in ~/.cache
RUN /etc/poetry/bin/poetry config virtualenvs.create false --local

# install common text editors for convenience
RUN apt install -y vim nano

WORKDIR /root/artifact
COPY . .

RUN /etc/poetry/bin/poetry env use 3.11 
RUN /etc/poetry/bin/poetry install --no-interaction

# Convert shell scripts to unix line ends (the image might be generated on a windows machine.)
RUN dos2unix *.sh

WORKDIR /root/

ENTRYPOINT ["zsh", "-l", "-c", "/root/artifact/load_env.sh"]

# Enter poetry shell in /root/artifact, but then go up one directory and start a shell
# CMD bash /root/artifact/cav_artifact/hello.sh