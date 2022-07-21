from benchmarks.common.generic import AbstractInputOutputManager
from src.shields import DynamicShield, AdaptiveDynamicShield, SafePadding
from src.shields.abstract_dynamic_shield import ShieldLifeType
from py4j.java_gateway import JavaGateway
from typing import Tuple


class GenericDynamicShield(DynamicShield):
    def __init__(self, ltl_formula: str, gateway: JavaGateway, io_manager: AbstractInputOutputManager,
                 shield_life_type: ShieldLifeType = ShieldLifeType.EPISODES, max_shield_life: int = 100,
                 min_depth: int = 0, concurrent_reconstruction=True, not_use_deviating_shield=False) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
            gateway: JavaGateway : the java gateway of py4j
            io_manager: AbstractInputOutputManager: defines input/output mappings
            shield_life_type: ShieldLifeType: determines if the shield_life is measured in episodes or steps
            max_shield_life: int: The number of the maximum episodes/steps to refresh the learned shield.
                                  This is used only when concurrent_reconstruction = True
            min_depth : int: the minimum depth we require to merge
        """

        self.io_manager = io_manager

        DynamicShield.__init__(self, ltl_formula=ltl_formula, gateway=gateway,
                               alphabet_start=self.io_manager.get_first_input(),
                               alphabet_end=self.io_manager.get_last_input(),
                               alphabet_mapper=self.io_manager.alphabet_mapper,
                               evaluate_output=self.io_manager.evaluate_output,
                               reverse_alphabet_mapper=self.io_manager.reverse_alphabet_mapper,
                               reverse_output_mapper=self.io_manager.reverse_output_mapper,
                               not_use_deviating_shield=not_use_deviating_shield,
                               concurrent_reconstruction=concurrent_reconstruction,
                               min_depth=min_depth,
                               shield_life_type=shield_life_type,
                               max_shield_life=max_shield_life)


class GenericAdaptiveDynamicShield(AdaptiveDynamicShield):
    def __init__(self, ltl_formula: str, gateway: JavaGateway, io_manager: AbstractInputOutputManager,
                 max_episode_length, shield_life_type: ShieldLifeType = ShieldLifeType.EPISODES,
                 max_shield_life: int = 100, concurrent_reconstruction=True, not_use_deviating_shield=False) -> None:
        """
           The constructor
           Args:
            ltl_formula: str : the LTL formula for the shielded specification
            gateway: JavaGateway : the java gateway of py4j
            io_manager: AbstractInputOutputManager: defines input/output mappings
            max_episode_length: maximum number of steps per episode
            shield_life_type: ShieldLifeType: determines if the shield_life is measured in episodes or steps
            max_shield_life: int: The number of the maximum episodes/steps to refresh the learned shield.
                                  This is used only when concurrent_reconstruction = True
        """

        self.io_manager = io_manager

        AdaptiveDynamicShield.__init__(self, ltl_formula=ltl_formula, gateway=gateway,
                                       alphabet_start=self.io_manager.get_first_input(),
                                       alphabet_end=self.io_manager.get_last_input(),
                                       alphabet_mapper=self.io_manager.alphabet_mapper,
                                       evaluate_output=self.io_manager.evaluate_output,
                                       reverse_alphabet_mapper=self.io_manager.reverse_alphabet_mapper,
                                       reverse_output_mapper=self.io_manager.reverse_output_mapper,
                                       not_use_deviating_shield=not_use_deviating_shield,
                                       concurrent_reconstruction=concurrent_reconstruction,
                                       shield_life_type=shield_life_type,
                                       max_shield_life=max_shield_life,
                                       max_episode_length=max_episode_length)


class GenericSafePadding(SafePadding):
    """
    Generic safe padding
    """

    def __init__(self, ltl_formula: str, io_manager: AbstractInputOutputManager) -> None:
        """
         The constructor
         Args:
          ltl_formula: str : the LTL formula for the shielded specification
        """
        super(GenericSafePadding, self).__init__(ltl_formula=ltl_formula,
                                                 player1_alphabet=io_manager.get_p1_actions(),
                                                 player2_alphabet=io_manager.get_p2_actions(),
                                                 output_alphabet=io_manager.get_outputs(),
                                                 evaluate_output=io_manager.evaluate_output)
        self.io_manager = io_manager

    def alphabet_mapper(self, mealy_action: int) -> Tuple[int, int]:
        return self.io_manager.alphabet_mapper(mealy_action)

    def reverse_alphabet_mapper(self, player1_action: int, player2_action) -> int:
        return self.io_manager.reverse_alphabet_mapper(player1_action, player2_action)

    def reverse_output_mapper(self, a: int) -> int:
        return self.reverse_output_mapper(a)
