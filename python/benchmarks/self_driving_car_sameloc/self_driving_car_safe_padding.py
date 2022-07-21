import sys
import os
import inspect
from const import XSIZE, YSIZE, GRID

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
benchmarkdir = os.path.dirname(currentdir)
pythondir = os.path.dirname(benchmarkdir)
sys.path.insert(0, pythondir)

from src.shields.safe_padding import SafePadding
from typing import Callable, Dict, Tuple, List, Union
import math

max_output = int(math.ceil(XSIZE / GRID) * (math.ceil(YSIZE / GRID)+1) * 9 + YSIZE *9 + 8)
output_alp = list(range(int(max_output+1)))
output_alp_crash = [l + 10000000 for l in output_alp]
output_alp = output_alp + output_alp_crash

class SelfDrivingCarSafePadding(SafePadding):
    def __init__(self, ltl_formula: Union[str, List[str]]):
        super(SelfDrivingCarSafePadding, self).__init__(ltl_formula=ltl_formula,
                                                   player1_alphabet=[0,1,2],
                                                   player2_alphabet=[],
                                                   output_alphabet=output_alp,
                                                   evaluate_output=self.evaluate_output)

    # @staticmethod
    def alphabet_mapper(action: int) -> Tuple[int, int]:
        return action, 0

    # @staticmethod
    def evaluate_output(self, s: int) -> Callable[[str], bool]:
        if s < 10000000:
            return lambda _: True
        else:
            return lambda _: False

    # @staticmethod
    def reverse_alphabet_mapper(player1_action: int, player2_action: int) -> int:
        return int(player1_action)

    # @staticmethod
    def reverse_output_mapper(mdp_output: int) -> int:
        return mdp_output