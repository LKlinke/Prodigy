from enum import Enum

from prodigy.analysis.evtinvariants.heuristics.templates.rational_function import RationalFunctionMaxDeg
from prodigy.analysis.evtinvariants.heuristics.templates.templates import TemplateHeuristic


class TemplateHeuristics(Enum):
    DEFAULT = RationalFunctionMaxDeg

    @staticmethod
    def create(h_type: 'TemplateHeuristics', *args, **kwargs) -> TemplateHeuristic:
        return h_type.value(*args, **kwargs)
