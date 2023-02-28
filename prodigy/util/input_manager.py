from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import easygui as gui  # type: ignore


class InputManager(ABC):
    @abstractmethod
    def read_text(self, prompt: str | None = None) -> str:
        pass

    @abstractmethod
    def read_option(self,
                    *options: str | Tuple[str, str],
                    prompt: str | None = None) -> int:
        """Returns the index of the option that was chosen."""

    def read_file(self, prompt: str | None = None, must_exist=True) -> str:
        file = self.read_text(prompt)
        while must_exist and not os.path.exists(file):
            file = self.read_text("File doesn't exist, please try again: ")
        return file

    def read_yn(self, prompt: str | None = None) -> bool:
        prompt_yn = f'{prompt} [Y/n]: ' if prompt is not None else '[Y/n]: '
        chosen = self.read_text(prompt_yn).lower()
        while chosen not in ('y', 'n'):
            chosen = self.read_text(
                'Invalid input, please try again: ').lower()
        return chosen == 'y'

    def read_int(self, prompt: str | None = None) -> int:
        res = self.read_text(prompt)
        while True:
            try:
                int(res)
                break
            except ValueError:
                res = self.read_text('Not an int, please try again: ')
        return int(res)

    def read_float(self, prompt: str | None = None) -> float:
        res = self.read_text(prompt)
        while True:
            try:
                float(res)
                break
            except ValueError:
                res = self.read_text('Not a float, please try again: ')
        return float(res)


class DefaultInputManager(InputManager):
    def read_text(self, prompt: str | None = None) -> str:
        if prompt is not None:
            return input(prompt)
        else:
            return input()

    def read_option(self,
                    *options: str | Tuple[str, str],
                    prompt: str | None = None) -> int:
        if len(options) == 0:
            raise ValueError('Needs at least one option')
        if prompt is not None:
            print(prompt)
        option_to_index: Dict[str, int] = {}
        for index, option in enumerate(options):
            if isinstance(option, str):
                print(f'[{index+1}]: {option}')
                option_to_index[str(index + 1)] = index
            elif isinstance(option, tuple):
                print(f'[{option[0]}]: {option[1]}')
                option_to_index[option[0]] = index
            else:
                raise ValueError(f'Invalid option: {option}')
        chosen = self.read_text('Please choose an option: ')
        while not chosen in option_to_index:
            chosen = self.read_text('Invalid input, please try again: ')
        return option_to_index[chosen]


class GraphicalInputManager(InputManager):
    def read_text(self, prompt: str | None = None) -> str:
        return gui.enterbox(prompt)

    def read_option(self,
                    *options: str | Tuple[str, str],
                    prompt: str | None = None) -> int:
        choices: List[str] = list(
            map(lambda item: item if isinstance(item, str) else item[1],
                list(options)))
        chosen = gui.choicebox(prompt, choices=choices)
        if not isinstance(chosen, str):
            raise ValueError(f"Unknown choice: {chosen}")
        return choices.index(chosen)

    def read_file(self, prompt: str | None = None, must_exist=True) -> str:
        if must_exist:
            return gui.fileopenbox(prompt)
        else:
            return gui.enterbox(prompt)

    def read_yn(self, prompt: str | None = None) -> bool:
        return gui.ynbox(prompt)

    def read_int(self, prompt: str | None = None) -> int:
        return gui.integerbox(prompt, upperbound = 1024)


reader = DefaultInputManager()
