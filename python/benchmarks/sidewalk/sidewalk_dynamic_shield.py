from typing import List, Union, Tuple

from py4j.java_gateway import JavaGateway

from benchmarks.sidewalk.sidewalk_gym_wrapper import evaluate_output
from src.shields import DynamicShield, UpdateShield

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "19 September 2021"


class SidewalkDynamicShield(DynamicShield):
    """
    Dynamic shield for sidewalk benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway,
                 update_shield: UpdateShield = UpdateShield.RESET, min_depth: int = 0,
                 max_shield_life: int = 100, not_use_deviating_shield: bool = False,
                 num_seeds: int = 1) -> None:
        """
        :param ltl_formula: Union[str, List[str]] : the LTL formula for the shielded specification
        """
        alphabet_start: int = 0
        alphabet_end: int = 4 * num_seeds - 1
        DynamicShield.__init__(self, ltl_formula, gateway,
                               alphabet_start, alphabet_end,
                               self.alphabet_mapper,
                               evaluate_output,
                               self.reverse_alphabet_mapper,
                               self.reverse_output_mapper,
                               update_shield=update_shield, min_depth=min_depth, concurrent_reconstruction=True,
                               max_shield_life=max_shield_life, skip_mealy_size=0,
                               not_use_deviating_shield=not_use_deviating_shield)

    @staticmethod
    def alphabet_mapper(mealy_action: int) -> Tuple[int, int]:
        return mealy_action % 4, mealy_action // 4

    @staticmethod
    def reverse_alphabet_mapper(player1_action: int, player2_action) -> int:
        return player1_action + (player2_action * 4)

    @staticmethod
    def reverse_output_mapper(a: int) -> int:
        return a
