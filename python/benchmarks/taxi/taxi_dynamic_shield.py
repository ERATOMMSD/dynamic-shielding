__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "8 April 2021"

from typing import Union, List
from py4j.java_gateway import JavaGateway
from src.shields import GenericDynamicShield, GenericAdaptiveDynamicShield, GenericSafePadding
from benchmarks.taxi.taxi_alphabet import TaxiInputOutputManager
from src.shields.abstract_dynamic_shield import ShieldLifeType


class TaxiDynamicShield(GenericDynamicShield):
    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway, min_depth: int = 3,
                 max_shield_life=200) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
            gateway: JavaGateway : the java gateway of py4j
            min_depth: int: the minimum depth of the dynamic shield
            max_shield_life: int: defines the number of steps that a shield can be reused
        """
        GenericDynamicShield.__init__(self, ltl_formula=ltl_formula, gateway=gateway,
                                      io_manager=TaxiInputOutputManager(),
                                      min_depth=min_depth,
                                      shield_life_type=ShieldLifeType.EPISODES,
                                      max_shield_life=max_shield_life)


class TaxiAdaptiveDynamicShield(GenericAdaptiveDynamicShield):
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
                                              io_manager=TaxiInputOutputManager(),
                                              max_episode_length=max_episode_length,
                                              shield_life_type=ShieldLifeType.EPISODES,
                                              max_shield_life=max_shield_life)


class TaxiSafePadding(GenericSafePadding):
    def __init__(self, ltl_formula: Union[str, List[str]]) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
        """
        GenericSafePadding.__init__(self, ltl_formula=ltl_formula, io_manager=TaxiInputOutputManager())
