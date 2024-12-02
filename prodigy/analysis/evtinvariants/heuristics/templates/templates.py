from abc import ABC, abstractmethod
from typing import Iterator

from prodigy.distribution import Distribution


class TemplateHeuristic(ABC):
    @abstractmethod
    def generate(self) -> Iterator[Distribution]:
        ...

    def __str__(self):
        return self.__class__.__name__
