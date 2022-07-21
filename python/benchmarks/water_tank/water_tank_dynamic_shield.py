__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "12 November 2020"

import sys
from typing import Union, List

sys.path.append('../../')
from src.shields import GenericDynamicShield, GenericAdaptiveDynamicShield, GenericSafePadding
from src.shields.abstract_dynamic_shield import ShieldLifeType
from py4j.java_gateway import JavaGateway
from benchmarks.water_tank.water_tank_arena import WaterTankInputOutputManager


class WaterTankDynamicShield(GenericDynamicShield):
    def __init__(self, ltl_formula: str, gateway: JavaGateway, min_depth: int = 0, max_shield_life=10) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
            gateway: JavaGateway : the java gateway of py4j
        """
        GenericDynamicShield.__init__(self, ltl_formula=ltl_formula,
                                      gateway=gateway,
                                      io_manager=WaterTankInputOutputManager(),
                                      min_depth=min_depth,
                                      shield_life_type=ShieldLifeType.EPISODES,
                                      max_shield_life=max_shield_life)


class WaterTankAdaptiveDynamicShield(GenericAdaptiveDynamicShield):
    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway, max_episode_length,
                 max_shield_life=200) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
            gateway: JavaGateway : the java gateway of py4j
            max_episode_length: int: the maximum number of steps of an episode
            max_shield_life: defines the number of episodes that a shield can be reused
        """
        GenericAdaptiveDynamicShield.__init__(self, ltl_formula=ltl_formula, gateway=gateway,
                                              io_manager=WaterTankInputOutputManager(),
                                              max_episode_length=max_episode_length,
                                              shield_life_type=ShieldLifeType.EPISODES,
                                              max_shield_life=max_shield_life)


class WaterTankSafePadding(GenericSafePadding):
    def __init__(self, ltl_formula: Union[str, List[str]]) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
        """
        GenericSafePadding.__init__(self, ltl_formula=ltl_formula, io_manager=WaterTankInputOutputManager())
