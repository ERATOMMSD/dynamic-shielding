import re
from typing import Tuple, Dict, List, Union

from py4j.java_gateway import JavaGateway

import benchmarks.grid_world.grid_world_dynamic_shield
from benchmarks.grid_world.grid_world_arena import AgentActions
from src.shields import DynamicShield, UpdateShield

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "21 October 2020"


class GridWorldDynamicShield(benchmarks.grid_world.grid_world_dynamic_shield.GridWorldDynamicShield):
    """
    Dynamic shield for grid_world benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway,
                 update_shield: UpdateShield = UpdateShield.RESET,
                 min_depth: int = 0, concurrent_reconstruction=False, max_shield_life: int = 100, skip_mealy_size: int = 5000) -> None:
        """
           The constructor
           Args:
             ltl_formula: Union[str, List[str]] : the LTL formula for the shielded specification
             gateway: JavaGateway : the gateway of Py4J
             max_shield_life: int: The number of the maximum episodes to refresh the learned shield. This is used only when concurrent_reconstruction = True
        """
        DynamicShield.__init__(self, ltl_formula, gateway, self.alphabet_start, self.alphabet_end,
                               self.alphabet_mapper, self.evaluate_output,
                               self.reverse_alphabet_mapper, self.reverse_output_mapper,
                               update_shield=update_shield, min_depth=min_depth, skip_mealy_size=skip_mealy_size,
                               concurrent_reconstruction=concurrent_reconstruction, max_shield_life=max_shield_life,
                               not_use_deviating_shield=False)

    """
        Note on the input alphabet
        - We have 5 * 5 = 25 input actions ([@-X])
        
        The encoding is as follows.
        - Player2's AgentActions:
            case (ascii(input action) % 5) of
              0 -> RIGHT
              1 -> UP
              2 -> LEFT
              3 -> DOWN
              4 -> STAY
        - Player1's AgentActions: 
            case ((ascii(input action) / 5) % 5) of
              0 -> RIGHT
              1 -> UP
              2 -> LEFT
              3 -> DOWN
              4 -> STAY
    """
    alphabet_start: int = 0
    alphabet_end: int = 24

    @staticmethod
    def alphabet_mapper(combined_action: int) -> Tuple[int, int]:
        """
            The function maps the alphabet for MealyMachine to that for ReactiveSystem
            Args:
                combined_action: str : An Action of Mealy machine
            Returns:
                the pair of the actions of player 1 and player 2 for ReactiveSystem
        """
        assert combined_action in range(5 * 5), \
            f'The input of `alphabet_mapper` is out of the range {combined_action}'
        to_agent_action: Dict[int, AgentActions] = {
            0: AgentActions.RIGHT,
            1: AgentActions.UP,
            2: AgentActions.LEFT,
            3: AgentActions.DOWN,
            4: AgentActions.STAY
        }

        player1_action: AgentActions = to_agent_action[(combined_action // 5) % 5]
        player2_action: AgentActions = to_agent_action[combined_action % 5]
        return int(player1_action), int(player2_action)

    @staticmethod
    def reverse_alphabet_mapper(player1_action: int, player2_action: int) -> int:
        """
            The function maps the alphabet for ReactiveSystem to that for MealyMachine
            Args:
                player1_action: int : An Action of player1 in ReactiveSystem
                player2_action: int : An Action of player2 in ReactiveSystem
            Returns:
                an actions for MealyMachine
        """

        assert AgentActions(player1_action) in AgentActions, f"illegal player1_action: {player1_action}"
        assert AgentActions(player2_action) in AgentActions, f"illegal player2_action: {player2_action}"

        combined_action: int = player1_action * len(AgentActions) + player2_action
        return combined_action
