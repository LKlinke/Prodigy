import pytest
from probably.pgcl import parse_pgcl

from prodigy.analysis.analyzer import compute_semantics
from prodigy.analysis.config import ForwardAnalysisConfig
from prodigy.analysis.instructionhandler.probchoice_handler import PChoiceHandler
from prodigy.analysis.instructionhandler.program_info import ProgramInfo
from prodigy.distribution.fast_generating_function import ProdigyPGF
from prodigy.distribution.generating_function import SympyPGF
from prodigy.distribution.symengine_distribution import SymenginePGF


@pytest.mark.parametrize('engine,factory',
                         [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                          (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF),
                          (ForwardAnalysisConfig.Engine.SYMENGINE, SymenginePGF)])
def test_probabilistic_choice(engine, factory):
    pgcl = """
    nat x;
    
    {x := 3} [1/2] { x:= 5}
    """

    program = parse_pgcl(pgcl)
    dist = factory.one(*program.variables)
    error = factory.zero()

    result = PChoiceHandler.compute(program.instructions[0],
                                    ProgramInfo(program),
                                    dist,
                                    error,
                                    ForwardAnalysisConfig(),
                                    compute_semantics)

    assert result[0] == factory.from_expr("1/2 * x^3 + 1/2 * x^5", "x") and result[1] == factory.zero()

    @pytest.mark.parametrize('engine,factory',
                             [(ForwardAnalysisConfig.Engine.SYMPY, SympyPGF),
                              (ForwardAnalysisConfig.Engine.GINAC, ProdigyPGF)])
    def test_probabilistic_choice_with_error(engine, factory):
        pgcl = """
        nat x;

        {x := 3} [1/2] { x:= 5}
        """

        program = parse_pgcl(pgcl)
        dist = factory.one("x", "y") * "1/2"
        error = factory.from_expr("1/2")

        result = PChoiceHandler.compute(program.instructions[0],
                                        ProgramInfo(program),
                                        dist,
                                        error,
                                        ForwardAnalysisConfig(),
                                        compute_semantics)

        assert result[0] == factory.from_expr("1/2 * x^3 + 1/2 * x^5", "x") and result[1] == error
