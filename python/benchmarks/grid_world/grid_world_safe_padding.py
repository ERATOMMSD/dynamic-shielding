from typing import List, Union, Tuple, Dict

from benchmarks.grid_world.deterministic_position import AgentActions, CrashPropositions
from benchmarks.grid_world.grid_world_arena import evaluate_output
from benchmarks.grid_world.grid_world_arena2 import ArenaPropositions
from src.shields.safe_padding import SafePadding

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "29 September 2021"

player1_alphabet = list(range(len(AgentActions)))
player2_alphabet = list(range(len(AgentActions)))
output_alphabet = list(range(len(ArenaPropositions) * len(CrashPropositions)))


class GridWorldSafePadding(SafePadding):
    """
    Safe pading for grid_world benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]]):
        super(GridWorldSafePadding, self).__init__(ltl_formula=ltl_formula,
                                                   player1_alphabet=player1_alphabet,
                                                   player2_alphabet=player2_alphabet,
                                                   output_alphabet=output_alphabet,
                                                   evaluate_output=evaluate_output)

    def reverse_output_mapper(self, a: int) -> int:
        return a

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
