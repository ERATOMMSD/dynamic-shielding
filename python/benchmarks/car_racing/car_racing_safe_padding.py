from typing import List, Union, Tuple

from benchmarks.car_racing.discrete_car_racing import AgentActions, ArenaPropositions, evaluate_output, \
    SensorPropositions
from src.shields.safe_padding import SafePadding

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "29 September 2021"

player1_alphabet = list(range(len(AgentActions)))
player2_alphabet = [0]
output_alphabet = list(range(len(ArenaPropositions) * len(SensorPropositions)))


class CarRacingSafePadding(SafePadding):
    """
    Safe padding for car-racing benchmark
    """

    def __init__(self, ltl_formula: Union[str, List[str]]):
        super(CarRacingSafePadding, self).__init__(ltl_formula=ltl_formula,
                                                   player1_alphabet=player1_alphabet,
                                                   player2_alphabet=player2_alphabet,
                                                   output_alphabet=output_alphabet,
                                                   evaluate_output=evaluate_output)

    def alphabet_mapper(self, mealy_action: int) -> Tuple[int, int]:
        return mealy_action, 1

    def reverse_alphabet_mapper(self, player1_action: int, player2_action) -> int:
        return player1_action

    def reverse_output_mapper(self, a: int) -> int:
        return a
