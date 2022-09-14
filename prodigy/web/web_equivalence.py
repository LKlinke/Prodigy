import logging
from typing import Optional

from flask import Flask, jsonify, make_response, render_template, request
from probably import pgcl
from probably.pgcl import check_program, parse_pgcl

from prodigy.analysis import (ForwardAnalysisConfig,
                              compute_discrete_distribution)
from prodigy.analysis.equivalence import check_equivalence

app = Flask(__name__)

# pylint: disable = broad-except


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/equivalence", methods=["POST"])
def checking_equivalence():
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
    try:
        loopy_prog = pgcl.parse_pgcl(loopy_source)
    except Exception as e:
        app.logger.exception(e)
        return make_response(jsonify({'message': 'Cannot parse loop file'}),
                             500)
    app.logger.debug("Parse invariant file")
    try:
        invariant_prog = pgcl.parse_pgcl(invariant_source)
    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': 'Cannot parse invariant file'}), 500)
    app.logger.debug("Finished parsing")

    app.logger.info("Run equivalence check")
    try:
        result, _ = check_equivalence(loopy_prog, invariant_prog,
                                      ForwardAnalysisConfig(engine=engine))
    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': 'Error while checking equivalence'}), 500)
    app.logger.info("Equivalence check finished. Result: %s", result)
    return make_response(jsonify({'equivalent': result}), 200)


@app.route("/analyze", methods=["POST"])
def distribution_transformation():
    app.logger.info("Distribution transformation requested")

    app.logger.debug("Collecting necessary data")
    prog_src = request.files["program"].read().decode("utf-8")
    if prog_src == '':
        return make_response(
            jsonify({'message': 'No program source selected'}), 500)
    input_dist_str = request.form["input_dist"]
    if input_dist_str == '':
        return make_response(
            jsonify({'message': 'Please provide an input distribution'}), 500)
    engine = ForwardAnalysisConfig.Engine.GINAC if request.form[
        'engine'] == 'ginac' else ForwardAnalysisConfig.Engine.SYMPY

    app.logger.debug("Parsing the program source")
    try:
        program = parse_pgcl(prog_src)
    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': 'Cannot parse program source'}), 500)
    app.logger.debug("Parsing done. Continue with type checking")
    if check_program(program):
        app.logger.debug("Typing Errors occured")
        return make_response("Typing Errors occured", 500)
    app.logger.debug("Type check passed.")

    app.logger.debug("Create input distribution")
    config = ForwardAnalysisConfig(engine=engine)
    input_dist = config.factory.from_expr(input_dist_str)
    app.logger.debug("Input distribution created")

    app.logger.info("Analysis task started for %s with input distribution %s",
                    program, input_dist)
    try:
        result = compute_discrete_distribution(program, input_dist, config)
    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': 'Error while computing distribution'}), 500)
    app.logger.info("Analysis completed")
    return jsonify({
        "distribution": str(result),
        "variables": list(result.get_variables()),
        "parameters": list(result.get_parameters())
    })


@app.route("/playground", methods=["POST"])
def analyze_raw_code():
    app.logger.info("Received a playground request")
    app.logger.debug("Collecting necessary data")
    prog_src = request.form['codearea']
    input_dist_str = request.form["playground_dist"]
    if input_dist_str == '':
        return make_response(
            jsonify({'message': 'Please provide an input distribution'}), 500)
    engine = ForwardAnalysisConfig.Engine.GINAC if request.form[
        'engine'] == 'ginac' else ForwardAnalysisConfig.Engine.SYMPY

    app.logger.debug("Parsing the program soruce")
    try:
        program = parse_pgcl(prog_src)
    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': 'Cannot parse program source'}), 500)
    app.logger.debug("Parsing done. Continue with type checking")
    if check_program(program):
        make_response("Typing Errors occured", 500)
    app.logger.debug("Type check passed.")

    app.logger.debug("Create input distribution")
    config = ForwardAnalysisConfig(engine=engine)
    input_dist = config.factory.from_expr(input_dist_str)
    app.logger.debug("Input distribution created")

    app.logger.info("Analysis task started for %s with input distribution %s",
                    program, input_dist)
    try:
        result = compute_discrete_distribution(program, input_dist, config)
    except Exception as e:
        app.logger.exception(e)
        return make_response(
            jsonify({'message': 'Error while computing distribution'}), 500)
    app.logger.info("Analysis completed")
    return jsonify({
        "distribution": str(result),
        "variables": list(result.get_variables()),
        "parameters": list(result.get_parameters())
    })


def start_server(port: Optional[int] = 8080):
    app.logger.setLevel(logging.DEBUG)
    app.env = 'development'
    app.run("localhost", port)


if __name__ == "__main__":
    start_server()
