import sys
import os
import inspect
import const

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
benchmarkdir = os.path.dirname(currentdir)
pythondir = os.path.dirname(benchmarkdir)
sys.path.insert(0, pythondir)
from src.shields import AdaptiveDynamicShield as DynamicShield
import spot 
from typing import Callable, Dict, Tuple, List, Union

from py4j.java_gateway import JavaGateway, CallbackServerParameters


class SelfDrivingCarDynamicShieldAbs(DynamicShield):
    """
	Dynamic shield for self_driving_car benchmark
	"""
    alphabet_start: int = 0
    alphabet_end: int = 2 #7

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway) -> None:
        DynamicShield.__init__(self, ltl_formula=ltl_formula, gateway=gateway,
                               alphabet_start=self.alphabet_start, alphabet_end=self.alphabet_end,
                               alphabet_mapper= self.alphabet_mapper,
                               evaluate_output= self.evaluate_output,
                               reverse_alphabet_mapper= self.reverse_alphabet_mapper,
                               reverse_output_mapper= self.reverse_output_mapper,
                               max_episode_length = const.MAX_CAR_STEP,
                               concurrent_reconstruction=True,
                               max_shield_life=300)

    @staticmethod
    def alphabet_mapper(action: int) -> Tuple[int, int]:
        return action, 0

    @staticmethod
    def evaluate_output(s: int) -> Callable[[str], bool]:
        # print('eval', s, type(s))
        # print('evaluate output', s, ord(s), '<97?', ord_s < 97)
        if s < 10000000: # 97:  # 25000: #
            return lambda _: True
        else:
            return lambda _: False

    @staticmethod
    def reverse_alphabet_mapper(player1_action: int, player2_action: int) -> int:
        return int(player1_action)

    @staticmethod
    def reverse_output_mapper(mdp_output: int) -> int:
        return mdp_output

