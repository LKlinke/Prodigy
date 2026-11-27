import pytest
from prodigy.language import parse_program

from prodigy.analysis.analyzer import compute_semantics_in
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.context import Frame, ProgramState
from prodigy.analysis.instructionhandler import probchoice_handler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.analysis.session import AnalysisSession
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF

PROGRAM = """x: nat;

{
    x = 3;
} [1/2] {
    x = 5;
}"""


def _choice(dist, error):
    program = parse_program(PROGRAM)
    frame = Frame(AnalysisSession.detached(), ProgramInfo(program), ForwardAnalysisConfig(), compute_semantics_in)
    return probchoice_handler.handle(frame, program.instructions[0], ProgramState(dist, error))


@pytest.mark.parametrize('factory', [SympyPGF, ProdigyPGF])
def test_probabilistic_choice(factory):
    result = _choice(factory.one("x"), factory.zero())
    assert result.dist == factory.from_expr("1/2 * x^3 + 1/2 * x^5", "x") and result.error == factory.zero()


@pytest.mark.parametrize('factory', [SympyPGF, ProdigyPGF])
def test_probabilistic_choice_keeps_the_error_mass(factory):
    error = factory.from_expr("1/2")
    result = _choice(factory.one("x") * "1/2", error)
    assert result.dist == factory.from_expr("1/4 * x^3 + 1/4 * x^5", "x") and result.error == error
