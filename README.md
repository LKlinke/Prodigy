# Overview

This artifact accompanies the paper “Generating Functions Meet Occupation Measures: Invariant Synthesis for Probabilistic Loops” and supports the experimental claims made therein.
It consists of a Docker image containing all required dependencies and the command-line tool, Prodigy, for analyzing probabilistic programs written in a variant of PGCL.
The primary purpose of this artifact is to:

1. Reproduce the benchmark results reported in the paper.
2. Allow reviewers to run the full benchmark suite with minimal setup.
3. Enable users to analyze their own PGCL programs using the same toolchain.

All experiments can be executed on a standard machine using Docker.
On a MacBook with M3 processor the benchmarks complete in approximately 2 minutes.

## Part I: Getting Started Guide

No additional installation of dependencies is required beyond Docker.

### Artifact Description

The artifact is distributed as a Docker image archived in a tarball.
The Docker image encapsulates:

- The Prodigy analysis tool
- All required solvers and backends
- Benchmark programs used in the paper

When run, the container presents a welcome screen that allows users to immediately execute the full benchmark suite recreating the benchmarks from Table 3 by confirming with "y" and pressing Enter.
Chosing "n" opens a shell session in which the artifact can be explored.

The benchmark results are written to:

- results.csv – benchmark results in the format `file; engine; cli_flags; time_in_s; invariant`
- timeouts.txt – benchmarks that exceeded time limits
- exceptions.txt – benchmarks that terminated with errors

### Installation and Execution Instructions

Load the Docker Image
From the directory containing the artifact tarball:

```bash
docker load -i prodigy.oci.tar
```

Verify that the image is available:

```bash
docker images
```

You should see an entry corresponding to the artifact image.
Now run the artifact by starting the container:

```bash
docker run -it prodigy:esop26-ae
```

You can then either confirm to reproduce all benchmarks, or disagree and you will be spawned in a virtual environment to discover the artifact.

## Part II: Run Custom Examples

In addition to the predefined benchmarks, users can analyze their own PGCL programs.

#### Calling Convention

Inside the container, the Prodigy CLI can be invoked as follows:

```bash
python prodigy/cli.py <target> file_to_analyse.pgcl [--invariant <invariant_template>] [optional_input_distribution]
```

Where:

- target is one of:
  - **main** - run the standard analysis
  - **invariant_synthesis** - run invariant synthesis
- file_to_analyse.pgcl is a PGCL program
- optional_input_distribution is an optional input distribution specified as a probability generating function (PGF), e.g. "1/2 * x^2 + 1/2".
- For `invariant_synthesis` an invariant template can be provided with `--invariant <invariant_template>`. Prodigy will try to instantiate this template. If no template is provided, `invariant_synthesis` will enumerate rational templates with increasing degrees.

#### Additional Options

Various command-line options allow configuration of:

- Solver backends
- Analysis strategies
- Verbose output
- etc.

These options can be explored via:

```bash
python prodigy/cli.py --help
```

#### Example

We adapt the `geometric_counter.pgcl` benchmark with a biased choice and larger increments:

```
nat x;
nat c;

x := 1
while (x = 1){
 {x := x - 1 } [1/3] {c := c+4}
}
```

Prodigy synthesizes an invariant for this program (`geometric_counter_biased.pgcl`) by using Z3 as a solver:

```bash
python prodigy/cli.py --solver z3 invariant_synthesis geometric_counter_biased.pgcl
```

> Invariant synthesis initiated...
> ...
> Invariant: (-3*x - 1)/(2*c**4 - 3)
> CPU-time elapsed: 19.675975 seconds

## Building the Docker image

To build the multi-platform Docker image, run the following command in the `prodigy` folder:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t prodigy:esop26-ae \
  -f Dockerfile \
  --output type=oci,dest=prodigy.oci.tar \
  .
```

The resulting `prodigy.oci.tar` file should contain the multi-platform Docker image.
