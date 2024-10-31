import subprocess
from subprocess import TimeoutExpired
import argparse
import sys
import sympy as sp
import os
from glob import glob

# All files in the pgfexamples folder
all_files: list[str] = [y for x in os.walk("pgfexamples") for y in glob(os.path.join(x[0], '*.pgcl'))] + [
    "example.pgcl"]

# Timeouts / Exception runs
timeouts: list[str] = list(map(str.strip, open("timeouts.txt", "r").readlines())) + list(
    map(str.strip, open("exceptions.txt", "r").readlines()))

# All available engines
engines = [
    "ginac", "symengine", "sympy"
]


class Run:
    """
    Represents a single run of one engine on one file.
    """
    def __init__(self, time, output, file):
        """
        Initializes the Run object.

        :param time: The time to run the benchmark.
        :param output: The output of the command. Given as tuple in form (expr, error_prob).
        :param file: The name of the file.
        """
        self.time: float = time
        self.output: tuple[str, str] = output
        self.file: str = file

    def __str__(self):
        return self.file + ": " + str(self.output) + " in " + str(self.time) + " seconds"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, Run):
            return self.time + other.time
        elif isinstance(other, (int, float)):
            return self.time + other

    def __radd__(self, other):
        return self.__add__(other)


class Configuration:
    """
    Represents the configuration given by the user
    """
    def __init__(self, args):
        """
        Initializes the Configuration object.

        :param args: The command line arguments.
        """

        self.output_file: str | None = args.output
        self.fail_on_error: bool = args.fail_on_error
        self.skip_timeouts: bool = args.skip_timeouts
        self.timeout: int = args.set_timeout
        self.generate_markdown: bool = args.generate_markdown

        assert not self.generate_markdown or self.output_file is not None, \
            "--generate-markdown may only be set if --output is specified."

        self.files: list[str]

        input_files = str(args.input).lower()
        if input_files == "all":
            # All files should be tested
            self.files = all_files
        elif "pgcl" in input_files:
            # A single file should be tested
            self.files = [input_files]
        else:
            # A file containing files that should be tested
            self.files = list(map(str.strip, open(input_files, "r").readlines()))

        self.engine: list[str]
        if args.engine is None:
            self.engine = engines
        else:
            engine = args.engine
            if "," in engine:
                engine = engine.split(",")
            else:
                engine = [engine]

            assert all(e in engines for e in engine), \
                f"Unrecognized engine. Given: {engine}"

            self.engine = engine


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Main input file
    parser.add_argument(
        "input",
        help="File to be analyzed. If set to \"all\", all files in the pgfexamples folder will be analyzed. " +
             "If file does not have a .pgcl extension, it is interpreted to be a file containing files to be tested.",
        type=str
    )

    # --engine
    # Chooses the engine which should be tested
    parser.add_argument(
        "--engine",
        metavar="ENGINE",
        help="The engine that should be tested primarily. Separate multiple engines by ','." +
             f"If unset, all engines are tested. Supported engines: {', '.join(engines)}."
    )

    # -o, --output
    # Specifies where (and if) an output file should be created
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="If set, the output will be printed into the given file (in csv format).",
        default=None,
        type=str
    )

    # --fail-on-error
    # Specifies if the script should terminate after an error occurs
    parser.add_argument(
        "--fail-on-error",
        help="If set, the script stops if the first error occurs.",
        action="store_true",
        default=False
    )

    # --skip-timeouts
    # Specifies whether files in the timeouts-list should be skipped
    parser.add_argument(
        "--skip-timeouts",
        help="If set, the script will skip all timeouts.",
        action="store_true",
        default=False
    )

    # --set-timeout
    # Specifies the time (in seconds) the subprocess should maximally run
    parser.add_argument(
        "--set-timeout",
        metavar="NUMBER",
        help="Sets the time (in seconds) after which the subprocess is terminated. Defaults to 90",
        default=90,
        type=int
    )

    # --generate-markdown
    # If set, generates a markdown table with the results and highlighted entries for fastest entries.
    parser.add_argument(
        "--generate-markdown",
        help="If set, the script will generate a markdown table of the results. Note: only works if 'output' is set.",
        action="store_true"
    )

    return parser


def main(args):
    """
    Parses the arguments and calls main benchmark method.
    """
    parser = create_parser()
    args = parser.parse_args(args)
    config = Configuration(args)
    benchmark(config)


def benchmark(config: Configuration):
    """
    Main benchmark method.
    Runs all given files on all selected engines, compares the results of the engines (if multiple are selected)
    and writes the results to the output file (if selected).

    :param config: The configuration to be benchmarked.
    """

    times: dict[str, list[Run]] = {
        engine: [] for engine in engines
    }

    # If output is wanted, setup the corresponding file
    if config.output_file is not None:
        setup_outfile(config.output_file)

    # For each file, test all engines
    for file in config.files:
        print(f"Now testing: {file}")

        # Check if the current file is in timeouts
        if file in timeouts and config.skip_timeouts:
            print(f"File is in timeouts, skipping...")
            continue

        # For each engine, run the program
        for engine in config.engine:
            print(f"Running {engine}")

            # Check if current file is in timeouts
            # Files can be added dynamically, hence the second check
            if file in timeouts and config.skip_timeouts:
                print(f"File is in timeouts, skipping...")
                continue

            output = ""
            # Execute the program
            try:
                cmd = ["python", "prodigy/cli.py", "--engine", engine, "main", file]
                output = subprocess.check_output(cmd, timeout=config.timeout).decode()
                print(output)
            except TimeoutExpired as e:
                # Command timed out
                with open("timeouts.txt", "a") as f:
                    # Write the file in "timeouts.txt" and add to the list
                    f.write(file + "\n")
                    timeouts.append(file)
                    if config.fail_on_error:
                        raise e
                    continue
            except subprocess.CalledProcessError as e:
                # Error occurred while running the program (e.g. runtime error)
                with open("exceptions.txt", "a") as f:
                    # Write the file in "timeouts.txt" and add to the list
                    f.write(file + "\n")
                    timeouts.append(file)
                    if config.fail_on_error:
                        raise e
                    continue

            if not output:
                raise RuntimeError("No output was captured.")

            # Parse output
            output = output.splitlines()

            # Sometimes other stuff is logged, which is captured
            # but not interesting for the analysis
            # The interesting part begins with "Result: [...]"
            while "Result" not in output[0]:
                output = output[1:]

            # Add time to the dictionary
            times[engine].append(
                Run(
                    time=float(output[-1].split()[-2]),
                    output=tuple(
                        str(
                            output[0]
                            .split("\x1b")[2]
                            .split("[92m")[1]
                        ).removeprefix("(")
                        .removesuffix(")")
                        .split(",")
                    ),  # Very hacky lol
                    file=file
                )
            )

        # Compare results if at least two engines are selected and file wasn't skipped once
        if len(config.engine) > 1 and file not in timeouts:
            try:
                results: dict[str, list[sp.Expr]] = {}
                for engine in config.engine:
                    # Results for "engine" look like this:
                    #   expr, error_prob
                    results[engine] = [sp.S(times[engine][-1].output[i]) for i in range(2)]
            except Exception as e:
                # Something went wrong while parsing
                print(str(e))
                if config.fail_on_error:
                    raise e
                continue

            fail = False
            # Compare all results
            for engine in config.engine:
                if fail:
                    break
                # Engines the current engine is compared against
                # Technically, we only need one direction of equality
                # but this is more convenient
                other_engines = set(results.keys()) - {engine}
                for other_engine in other_engines:
                    if fail:
                        break
                    # Compare expr and error_prob
                    for i in range(2):
                        try:
                            assert results[engine][i].equals(results[other_engine][i]), \
                                f"""
                                Engine {engine} disagrees with engine {other_engine} on file {file}. 
                                {engine}: {results[engine][i]}
                                {other_engine}: {results[other_engine][i]}
                                """
                        except AssertionError as e:
                            with open("exceptions.txt", "a") as f:
                                # Write file to exception file
                                # We do not need to add file to timeouts,
                                # as the file was already checked
                                f.write(file + "\n")
                            print(str(e))
                            if config.fail_on_error:
                                raise e
                            fail = True
                            break

            # Results are equal, add run to output file (if set)
            if config.output_file is not None and not fail:
                with open(config.output_file, "a") as f:
                    f.write(file)
                    for engine in config.engine:
                        f.write(f",{times[engine][-1].time}")
                    f.write("\n")
            print("Results are equal, continuing...")

    # Write average to output file
    if config.output_file is not None:
        print("Writing output file...")
        with open(config.output_file, "a") as f:
            averages = [str(round(sum(times[engine]) / len(times[engine]), 6)) for engine in engines]
            f.write("Average," + ",".join(averages))

    # If generate markdown is set, create the Markdown table
    if config.generate_markdown:
        print("Generating markdown table...")
        generate_markdown_table(config.output_file)


def setup_outfile(out_file: str):
    # If file exists, rename it to filename_{i}.extension
    if os.path.isfile(out_file):
        i = 1
        new_name = out_file.split(".")[0] + f"_{i}." + out_file.split(".")[1]
        while os.path.isfile(new_name):
            i += 1
            new_name = out_file.split(".")[0] + f"_{i}." + out_file.split(".")[1]
        os.rename(out_file, new_name)

    # Write header
    with open(out_file, "a") as f:
        f.write("file")
        for engine in engines:
            f.write(f",{engine}")
        f.write("\n")


def generate_markdown_table(csv_file: str):
    """
    Parses a given csv file to a Markdown table while highlighting the fastest result

    :param csv_file: The csv file to be parsed.
    """
    out_file = csv_file.split(".")[0] + "_format.md"

    with open(csv_file, "r") as f:
        lines = f.readlines()

    no_fastest = [0] * (len(lines[0].strip().split(",")) - 1)
    with open(out_file, "w") as f:
        for line in lines:
            line = line.strip().split(",")

            # Check if header
            if any([el in line for el in engines]):
                f.write("|" + "|".join(line) + "|\n")
                f.write("|---" * len(line) + "|\n")
            else:
                values = list(map(float, line[1:]))
                min_value = min(values)
                index = line.index(str(min_value))
                no_fastest[index - 1] += 1
                line[index] = f"**{min_value}**"
                f.write("|".join(line) + "|\n")
        f.write("Times fastest run|" + "|".join(map(str, no_fastest)) + "|\n")


if __name__ == '__main__':
    main(sys.argv[1:])
