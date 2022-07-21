from typing import List, Union, Tuple

from benchmarks.sidewalk.sidewalk_gym_wrapper import evaluate_output, ArenaPropositions, ConePropositions, \
    GoalPropositions, WallPropositions
from src.shields.safe_padding import SafePadding

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "28 September 2021"

player1_alphabet = list(range(4))


def player2_alphabet(num_seeds: int):
    return list(range(num_seeds))


output_alphabet = list(
    range(len(ArenaPropositions) * len(ConePropositions) * len(GoalPropositions) * len(WallPropositions)))


class SidewalkSafePadding(SafePadding):
    """
    Safe pading for sidewalk benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]], num_seeds: int = 1):
        super(SidewalkSafePadding, self).__init__(ltl_formula=ltl_formula,
                                                  player1_alphabet=player1_alphabet,
                                                  player2_alphabet=player2_alphabet(num_seeds),
                                                  output_alphabet=output_alphabet,
                                                  evaluate_output=evaluate_output)

    def alphabet_mapper(self, mealy_action: int) -> Tuple[int, int]:
        return mealy_action % 4, mealy_action // 4

    def reverse_alphabet_mapper(self, player1_action: int, player2_action) -> int:
        return player1_action + (player2_action * 4)

    def reverse_output_mapper(self, a: int) -> int:
        return a
