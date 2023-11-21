import logging
import sys
from io import StringIO
from typing import Optional

from flask import Flask, jsonify, make_response, render_template, request
from probably import pgcl
from probably.pgcl import check_program, parse_pgcl

from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.equivalence.equivalence_check import check_equivalence
from prodigy.analysis.instruction_handler import compute_discrete_distribution

app = Flask(__name__)


# pylint: disable = broad-except


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/equivalence", methods=["POST"])
def checking_equivalence():
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    try:
        app.logger.info("Equivalence check requested")
        loopy_source = request.files['loop'].read().decode("utf-8")
        if loopy_source == '':
            return make_response(
                jsonify({'message': 'No loop program source selected'}), 500)
        app.logger.debug("Loop-file %s", loopy_source)
        invariant_source = request.files['invariant'].read().decode("utf-8")
        if invariant_source == '':
            return make_response(
                jsonify({'message': 'No invariant source selected'}), 500)
        app.logger.debug("Invariant file %s", invariant_source)
        engine = ForwardAnalysisConfig.Engine.GINAC if request.form[
            'engine'] == 'ginac' else ForwardAnalysisConfig.Engine.SYMPY
        app.logger.debug("Chosen engine %s", engine)

        app.logger.debug("Parse loop-file")
        loopy_prog = pgcl.parse_pgcl(loopy_source)
        app.logger.debug("Parse invariant file")
        invariant_prog = pgcl.parse_pgcl(invariant_source)
        app.logger.debug("Finished parsing")

        app.logger.info("Run equivalence check")
        result, _ = check_equivalence(loopy_prog, invariant_prog,
                                      ForwardAnalysisConfig(engine=engine))
        app.logger.info("Equivalence check finished. Result: %s", result)

        return make_response(
            jsonify({
                'output': my_stdout.getvalue(),
                'equivalent': result
            }), 200)

    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': f'An error occurred: {str(e)}'}), 500)

    finally:
        sys.stdout = old_stdout
        print("Console output:")
        print(my_stdout.getvalue())


@app.route("/analyze", methods=["POST"])
def distribution_transformation():
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    try:
        app.logger.info("Distribution transformation requested")

        app.logger.debug("Collecting necessary data")
        prog_src = request.files["program"].read().decode("utf-8")
        if prog_src == '':
            return make_response(
                jsonify({'message': 'No program source selected'}), 500)
        input_dist_str = request.form["input_dist"]
        if input_dist_str == '':
            return make_response(
                jsonify({'message': 'Please provide an input distribution'}),
                500)
        engine = ForwardAnalysisConfig.Engine.GINAC if request.form[
            'engine'] == 'ginac' else ForwardAnalysisConfig.Engine.SYMPY

        app.logger.debug("Parsing the program source")
        program = parse_pgcl(prog_src)
        app.logger.debug("Parsing done. Continue with type checking")
        if check_program(program):
            app.logger.debug("Typing Errors occured")
            return make_response("Typing Errors occured", 500)
        app.logger.debug("Type check passed.")

        app.logger.debug("Create input distribution")
        config = ForwardAnalysisConfig(engine=engine)
        input_dist = config.factory.from_expr(input_dist_str)
        app.logger.debug("Input distribution created")

        app.logger.info(
            "Analysis task started for %s with input distribution %s", program,
            input_dist)
        result = compute_discrete_distribution(program, input_dist, config)
        app.logger.info("Analysis completed")

        return jsonify({
            "output": my_stdout.getvalue(),
            "distribution": str(result),
            "variables": list(result.get_variables()),
            "parameters": list(result.get_parameters())
        })

    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': f'An error occurred: {str(e)}'}), 500)

    finally:
        sys.stdout = old_stdout
        print("Console output:")
        print(my_stdout.getvalue())


@app.route("/playground", methods=["POST"])
def analyze_raw_code():
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    try:
        app.logger.info("Received a playground request")
        app.logger.debug("Collecting necessary data")
        prog_src = request.form['codearea']
        input_dist_str = request.form["playground_dist"]
        if input_dist_str == '':
            return make_response(
                jsonify({'message': 'Please provide an input distribution'}),
                500)
        engine = ForwardAnalysisConfig.Engine.GINAC if request.form[
            'engine'] == 'ginac' else ForwardAnalysisConfig.Engine.SYMPY

        app.logger.debug("Parsing the program source")
        program = parse_pgcl(prog_src)
        app.logger.debug("Parsing done. Continue with type checking")
        if check_program(program):
            return make_response("Typing Errors occured", 500)
        app.logger.debug("Type check passed.")

        app.logger.debug("Create input distribution")
        config = ForwardAnalysisConfig(engine=engine)
        input_dist = config.factory.from_expr(input_dist_str)
        app.logger.debug("Input distribution created")

        app.logger.info(
            "Analysis task started for %s with input distribution %s", program,
            input_dist)
        result, error_prob = compute_discrete_distribution(program, input_dist, config)
        app.logger.info("Analysis completed")

        return jsonify({
            "output": my_stdout.getvalue(),
            "distribution": str(result),
            "variables": list(result.get_variables()),
            "parameters": list(result.get_parameters())
        })

    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': f'An error occurred: {str(e)}'}), 500)

    finally:
        sys.stdout = old_stdout
        print("Console output:")
        print(my_stdout.getvalue())


def start_server(port: Optional[int] = 8080):
    app.logger.setLevel(logging.DEBUG)
    app.env = 'development'
    app.run("localhost", port)


if __name__ == "__main__":
    start_server()
