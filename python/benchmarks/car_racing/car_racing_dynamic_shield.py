from typing import List, Union, Tuple

from py4j.java_gateway import JavaGateway

from benchmarks.car_racing.discrete_car_racing import AgentActions, evaluate_output, evaluate_output_without_sensor
from src.shields import DynamicShield, UpdateShield

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "3 August 2021"

from src.shields.adaptive_dynamic_shield import AdaptiveDynamicShield


class CarRacingDynamicShield(DynamicShield):
    """
    Dynamic shield for grid_world benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway,
                 update_shield: UpdateShield = UpdateShield.RESET, min_depth: int = 0,
                 max_shield_life: int = 100, not_use_deviating_shield: bool = False, use_sensor=True) -> None:
        """
           The constructor
           Args:
             ltl_formula: Union[str, List[str]] : the LTL formula for the shielded specification
        """
        alphabet_start: int = 0
        alphabet_end: int = len(AgentActions) - 1
        DynamicShield.__init__(self, ltl_formula, gateway,
                               alphabet_start, alphabet_end,
                               self.alphabet_mapper,
                               evaluate_output if use_sensor else evaluate_output_without_sensor,
                               self.reverse_alphabet_mapper,
                               self.reverse_output_mapper,
                               update_shield=update_shield, min_depth=min_depth, concurrent_reconstruction=True,
                               max_shield_life=max_shield_life, skip_mealy_size=0,
                               not_use_deviating_shield=not_use_deviating_shield)

    @staticmethod
    def alphabet_mapper(mealy_action: int) -> Tuple[int, int]:
        return mealy_action, 0

    @staticmethod
    def reverse_alphabet_mapper(player1_action: int, player2_action) -> int:
        return player1_action

    @staticmethod
    def reverse_output_mapper(a: int) -> int:
        return a


class CarRacingAdaptiveDynamicShield(AdaptiveDynamicShield):
    """
    Dynamic shield with adaptive min_depth for CarRacing benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway, max_episode_length: int,
                 update_shield: UpdateShield = UpdateShield.RESET, factor: float = 1.0,
                 max_shield_life: int = 100, not_use_deviating_shield: bool = False, use_sensor=True) -> None:
        """
           The constructor
           Args:
             ltl_formula: Union[str, List[str]] : the LTL formula for the shielded specification
        """
        alphabet_start: int = 0
        alphabet_end: int = len(AgentActions) - 1
        AdaptiveDynamicShield.__init__(self, ltl_formula, gateway,
                                       alphabet_start, alphabet_end,
                                       self.alphabet_mapper,
                                       evaluate_output if use_sensor else evaluate_output_without_sensor,
                                       self.reverse_alphabet_mapper,
                                       self.reverse_output_mapper, max_episode_length=max_episode_length,
                                       update_shield=update_shield, concurrent_reconstruction=True,
                                       max_shield_life=max_shield_life, skip_mealy_size=0,
                                       not_use_deviating_shield=not_use_deviating_shield,
                                       factor=factor)

    @staticmethod
    def alphabet_mapper(mealy_action: int) -> Tuple[int, int]:
        return mealy_action, 0

    @staticmethod
    def reverse_alphabet_mapper(player1_action: int, player2_action) -> int:
        return player1_action

    @staticmethod
    def reverse_output_mapper(a: int) -> int:
        return a
