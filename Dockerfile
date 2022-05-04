FROM ubuntu

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Setup timezone
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

# Add repository for older python versions on ubuntu.
RUN apt update
RUN apt install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt update


RUN apt install -y git cmake make graphviz curl libginac-dev python3.9-dev python3.9-distutils python3-apt python3-distutils  python3-dev build-essential
# setup git so we can use it with poetry later
RUN git config --global user.name "Your Name" && git config --global user.email "you@example.com"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
# poetry will be installed by adding a line to .profile which is loaded by bash
# login shells. to access it, we'll need to wrap any calls to it with `bash -l -c`.

# for discoverability, add the virtual environment in the project directory instead of
# somewhere in ~/.cache
RUN bash -l -c "poetry config virtualenvs.in-project true"

# install common text editors for convenience
RUN apt install -y vim nano

WORKDIR /root/artifact
COPY . .

RUN bash -l -c 'poetry update && poetry env use 3.9 && poetry install --no-interaction'


WORKDIR /root/

# Enter poetry shell in /root/artifact, but then go up one directory and start a shell
# CMD bash /root/artifact/cav_artifact/hello.sh