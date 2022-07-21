from typing import List, Tuple

from src.exceptions.shielding_exceptions import InvalidInputError
from src.model import ReactiveSystem

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "21 December 2020"


class PTA(ReactiveSystem):
    """
    The class for prefix tree acceptor (PTA). A PTA is a tree labeled with the actions and it represents the training data for passive automata learning.
    We implement a PTA as a reactive system.
    """

    def __init__(self, player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int]) -> None:
        super().__init__(player1_alphabet, player2_alphabet, output_alphabet)

    def addSample(self, input_word: List[Tuple[int, int]], output_action: int) -> None:
        state: int = self.initialState
        for player1_action, player2_action in input_word[:-1]:
            if (player1_action, player2_action) in self.transitions[state]:
                state = self.transitions[state][player1_action, player2_action]
            else:
                raise InvalidInputError("The input to the PTA must be prefix closed")
        player1_action, player2_action = input_word[-1]
        if player2_action not in self.player2Alphabet:
            self.player2Alphabet.append(player2_action)
        if state in self.transitions and (player1_action, player2_action) in self.transitions[state]:
            target = self.transitions[state][player1_action, player2_action]
        elif len(self.getStates()) == 0:
            target = 2
        else:
            target = len(self.getStates()) + 1
        self.addTransition(state, player1_action, player2_action, output_action, target)
