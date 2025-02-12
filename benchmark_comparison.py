import subprocess
from subprocess import TimeoutExpired
import argparse
import sys
import sympy as sp
import os
from glob import glob
import re

# All files in the pgfexamples folder
# https://stackoverflow.com/a/18394205
all_files: list[str] = [y for x in os.walk("pgfexamples") for y in glob(os.path.join(x[0], '*.pgcl'))] + [
    "example.pgcl"]

# Timeouts / Exception runs
# If files are not present, an empty list is returned
skip_files: list[str] = (list(map(str.strip, open("timeouts.txt", "r").readlines())) \
    if os.path.isfile("timeouts.txt") else []) + \
                                           (list(map(str.strip,
                                                    open("exceptions.txt", "r").readlines())) if os.path.isfile(
    "exceptions.txt") else [])

# All available engines
# (will be executed in this order)
engines: list[str] = [
    "ginac", "symengine", "sympy"
]

# Default CLI args for files that do not contain any additional information
default_instruction: list[str] = ["main"]

# https://stackoverflow.com/a/14693789
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


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
            self.files = [args.input]
        elif os.path.isdir(input_files):
            # A folder is given, only test files in the folder
            self.files = [y for x in os.walk(args.input) for y in glob(os.path.join(x[0], '*.pgcl'))]
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
        metavar="INPUT",
        help="File to be analyzed. If set to \"all\", all files in the pgfexamples folder will be analyzed. " +
             "If a folder is provided, all files with '.pgcl' extensions in the folder or its subfolders are tested. " +
             "If  file does not have a .pgcl extension, it is interpreted to be a file containing files to be tested.",
        type=str
    )

    # --engine
    # Chooses the engine which should be tested
    parser.add_argument(
        "--engine",
        metavar="ENGINE",
        help="The engine that should be tested primarily. Separate multiple engines by ','. " +
             "If unset, all engines are tested. Note: Engines will be executed in the order given. " +
             f"Supported engines: {', '.join(engines)}."
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
    # If set, generates a Markdown table with the results and highlighted entries for fastest entries.
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
    # Create timeouts and exceptions files if not yet present
    if not os.path.isfile("exceptions.txt"):
        with open("exceptions.txt", "w"):
            pass

    if not os.path.isfile("timeouts.txt"):
        with open("timeouts.txt", "w"):
            pass

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

    # If output is wanted, set up the corresponding file
    if config.output_file is not None:
        setup_outfile(config.output_file, config.engine)

    counter = 0
    # For each file, test all engines
    for file in config.files:
        counter += 1
        print(f"Now testing: {file} ({counter}/{len(config.files)})")
        # Check if the current file is in timeouts
        if file in skip_files and config.skip_timeouts:
            print(f"File is in skipped files, skipping...")
            continue
        engine_counter = 0

        instructions = obtain_instructions(file)
        # Check if file is to be skipped
        if not instructions:
            print("File marked to be skipped...")
            continue

        inputs = obtain_inputs(instructions)

        skipped = False
        # For each engine, run the program
        for engine in config.engine:
            if skipped:
                continue

            engine_counter += 1
            print(f"Running {engine} ({engine_counter}/{len(config.engine)})")

            # Check if current file is in timeouts
            # Files can be added dynamically, hence the second check
            if file in skip_files and config.skip_timeouts:
                print(f"File is in skipped files, skipping...")
                skipped = True
                continue

            output = ""

            # Execute the program
            cmd = ["python", "prodigy/cli.py", "--engine", engine, *instructions]

            try:
                output = subprocess.check_output(cmd, timeout=config.timeout, input=inputs).decode()
            except TimeoutExpired as e:
                # Command timed out
                print("Command timed out, writing in timeouts.txt...")
                with open("timeouts.txt", "a") as f:
                    # Write the file in "timeouts.txt" and add to the list
                    f.write(file + "\n")
                    skip_files.append(file)
                    skipped = True
                    if config.fail_on_error:
                        raise e
                    continue
            except subprocess.CalledProcessError as e:
                print("Command threw an exception, writing in exceptions.txt...")
                # Error occurred while running the program (e.g. runtime error)
                with open("exceptions.txt", "a") as f:
                    # Write the file in "timeouts.txt" and add to the list
                    f.write(file + "\n")
                    skip_files.append(file)
                    skipped = True
                    if config.fail_on_error:
                        raise e
                    continue

            if not output:
                raise RuntimeError("No output was captured.")

            # Parse output
            print(output)
            output = output.splitlines()
            run = capture_output(output, instructions, file)
            print(run)

            # Add the result to the dictionary
            times[engine].append(run)
        fail = False

        # Compare results if at least two engines are selected and file wasn't skipped once (if skip_timeouts is set)
        if len(config.engine) > 1 and not skipped:
            fail = compare_output({engine: times[engine][-1] for engine in config.engine}, instructions, file,
                                  config.fail_on_error)
            if not fail:
                print("Results are equal, continuing...")

        # Write results if output file is set (and results are equal or just one engine is tested)
        if config.output_file is not None and not fail and not skipped:
            with open(config.output_file, "a") as f:
                f.write(file)
                for engine in config.engine:
                    f.write(f",{times[engine][-1].time}")
                f.write("\n")
    # If generate markdown is set, create the Markdown table
    if config.generate_markdown:
        print("Generating markdown table...")
        generate_markdown_table(config.output_file, config.engine)


def setup_outfile(out_file: str, engine_list: list[str]) -> None:
    """
    Creates the output csv file and writes the header. If a file of the same name already exists, rename it to <name>_{i},
    where i is the lowest number such that no file with the name <name>_i exists.
    :param out_file: The file to be created.
    :param engine_list: The list of engines which are tested.
    """
    # If file exists, rename it to filename_{i}.extension
    if os.path.isfile(out_file):
        i = 1
        new_name_template = lambda el: out_file.split(".")[0] + f"_{el}." + out_file.split(".")[1]
        new_name = new_name_template(i)
        while os.path.isfile(new_name):
            i += 1
            new_name = new_name_template(i)
        os.rename(out_file, new_name)

    # Write header
    with open(out_file, "a") as f:
        f.write("file")
        for engine in engine_list:
            f.write(f",{engine}")
        f.write("\n")


def obtain_instructions(file_path: str) -> list[str]:
    """
    Reads the first line of given file and checks whether it is instructions for the command
    """

    with open(file_path, "r") as f:
        first_line = f.readlines()[0]

    if first_line.startswith("#"):
        # First line is an instruction
        parts = list(map(str.strip, first_line.split()[1:]))
        # If "skip" is set, the file should be ignored
        if parts == ["skip"]:
            return []
        return parts
    else:
        # No instruction is given -> return the default instruction
        print(f"No instruction found for {file_path}, executing default instruction...")
        return default_instruction + [file_path]

def obtain_inputs(instructions: list[str]) -> bytes:
    if len(instructions) == 2:
        # No input is necessary
        return b""
    method = instructions[0]
    other_file = instructions[2]
    input_cmd = b""
    if method == "check_equality" and ("loopy" in other_file or "template_parameter_synthesis"):
        # Select invariant file for loopy programs
        input_cmd = b"1\n" + str.encode(other_file + "\n")
    return input_cmd

def capture_output(output: list[str], cmd: list[str], file: str) -> Run:
    # Remove ANSI
    output = [ansi_escape.sub("", o) for o in output]
    if "main" in cmd:
        # Sometimes other stuff is logged, which is captured
        # but not interesting for the analysis.
        # The interesting part begins with "Result: [...]"
        while "Result" not in output[0]:
            output = output[1:]
        return Run(
            time=float(output[-1].split()[-2]),
            output=tuple(
                str(output[0].split("\t")[1])
                .removeprefix("(")
                .removesuffix(")")
                .split(",")
            ),
            file=file
        )
    elif "check_equality" in cmd:
        # TODO check parameter for parameter synthesis?
        while "equivalent" not in output[0]:
            output = output[1:]
        return Run(
            time=float(output[-1].split()[-2]),
            output=(output[0].startswith("Program is equivalent to invariant")),
            file=file
        )
    elif "invariant_synthesis" in cmd:
        while "Invariant: " not in output[0]:
            output = output[1:]
        return Run(
            time=float(output[-1].split()[-2]),
            output=output[0].split()[1],
            file=file
        )
    else:
        # todo other methods
        pass


def compare_output(outputs: dict[str, Run], cmd: list[str], file: str, fail_on_error: bool) -> bool:
    fail = False
    if "main" in cmd:
        parsed_results: dict[str, tuple[sp.Expr, sp.Expr]] = {}
        for engine in outputs.keys():
            try:
                parsed_results[engine] = (sp.S(outputs[engine].output[0]), sp.S(outputs[engine].output[0]))
            except Exception as e:
                print(str(e))
                if fail_on_error:
                    raise e

        for engine in outputs.keys():
            other_engines = set(outputs.keys()) - {engine}
            for other_engine in other_engines:
                if fail:
                    break
                for i in range(2):
                    if fail:
                        break
                    try:
                        assert parsed_results[engine][i].equals(parsed_results[other_engine][i]), \
                            f"""
                                Engine {engine} disagrees with {other_engine} on file {file}.
                            """ + "\n".join(f"{e}: {parsed_results[e][i]}" for e in engines)
                    except AssertionError as e:
                        fail = True
                        with open("exceptions.txt", "a") as f:
                            # Write file to exception file
                            # We do not need to add file to timeouts,
                            # as the file was already checked
                            f.write(file + "\n")
                        print(str(e))
                        if fail_on_error:
                            raise e
                        break
    elif "equivalence" in cmd:
        for engine in outputs.keys():
            other_engines = set(outputs.keys()) - {engine}
            for other_engine in other_engines:
                if fail:
                    break
                try:
                    assert outputs[engine].output == outputs[other_engine].output, \
                        f"""
                            Engine {engine} disagrees with {other_engine} on file {file}.
                        """ + "\n".join(f"{e}: {outputs[e].output}" for e in engines)
                except AssertionError as e:
                    fail = True
                    with open("exceptions.txt", "a") as f:
                        # Write file to exception file
                        # We do not need to add file to timeouts,
                        # as the file was already checked
                        f.write(file + "\n")
                    print(str(e))
                    if fail_on_error:
                        raise e
                    break
    elif "invariant_synthesis" in cmd:
        parsed_results: dict[str, sp.Expr] = {}
        for engine in outputs.keys():
            try:
                parsed_results[engine] = sp.S(outputs[engine].output)
            except Exception as e:
                print(str(e))
                if fail_on_error:
                    raise e
        for engine in outputs.keys():
            other_engines = set(outputs.keys()) - {engine}
            for other_engine in other_engines:
                if fail:
                    break
                try:
                    assert parsed_results[engine].equals(parsed_results[other_engine]), \
                        f"""
                            Engine {engine} disagrees with {other_engine} on file {file}.
                        """ + "\n".join(f"{e}: {parsed_results[e]}" for e in engines)
                except AssertionError as e:
                    fail = True
                    with open("exceptions.txt", "a") as f:
                        # Write file to exception file
                        # We do not need to add file to timeouts,
                        # as the file was already checked
                        f.write(file + "\n")
                    print(str(e))
                    if fail_on_error:
                        raise e
                    break
    else:
        # TODO other methods
        pass
    return fail


def generate_markdown_table(csv_file: str, engine_list: list[str]) -> None:
    """
    Parses a given csv file to a Markdown table while highlighting the fastest result.

    :param csv_file: The csv file to be parsed.
    :param engine_list: A list of engines to be compared.
    """

    # TODO should number of skipped runs be logged?

    out_file = csv_file.split(".")[0] + "_format.md"

    # Read the output file
    with open(csv_file, "r") as f:
        lines = f.readlines()

    # Stores the results
    results: list[list[float]] = [[] for _ in range(len(engine_list))]
    # Stores information about the fastest engine
    no_fastest = [0] * (len(engine_list))

    with open(out_file, "w") as f:
        # Parse the results
        for line in lines:
            line = line.strip().split(",")
            for (i, value) in enumerate(line[1:]):
                if any([el in line for el in engine_list]):
                    continue
                results[i].append(float(value.strip()))

            # Check if header
            if any([el in line for el in engine_list]):
                # Write header
                f.write("|" + "|".join(line) + "|\n")
                f.write("|---" * len(line) + "|\n")
            else:
                # Get the values as floats
                values = list(map(float, line[1:]))
                # Observe the minimum (=fastest)
                min_value = min(values)
                # Get its index and increase the counter by one
                index = line.index(str(min_value))
                no_fastest[index - 1] += 1
                # Highlight the fastest run by making it bold
                line[index] = f"**{min_value}**"
                # Write the run
                f.write("|" + "|".join(line) + "|\n")
        # Write the average and summary
        f.write(
            "|Average|" + "|".join([f"{sum(results[i]) / len(results[i])}" for i in range(len(engine_list))]) + "|\n")
        # Only add the time comparison if at least 2 engines are selected
        if len(engine_list) > 1:
            f.write("Times fastest run|" + "|".join(map(str, no_fastest)) + "|\n")


if __name__ == '__main__':
    main(sys.argv[1:])
