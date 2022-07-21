import itertools
from typing import List, Dict, Tuple, Callable

from src.exceptions.shielding_exceptions import UnknownOutputError
from src.model import MealyMachine

__author__ = "Masaki Waga <masakiwaga@gmail.com>, Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.2"
__date__ = "24 September 2020"


class ReactiveSystem:
    """
    The class for finite-state reactive system.
    A finite-state reactive system is basically a Mealy machine with split input alphabets for player 1 and 2.
    """

    def __init__(self, player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int]) -> None:
        self.player1Alphabet: List[int] = player1_alphabet
        self.player2Alphabet: List[int] = player2_alphabet
        self.outputAlphabet: List[int] = output_alphabet
        self.initialState: int = 1
        # self.transition[current_state][(action_player1, action_player2)] is the target_state
        self.transitions: Dict[int, Dict[Tuple[int, int], int]] = {}
        # Note: By the definition of Mealy machine, it is not guaranteed that the output only depends on the action of
        # player 1. However, this is a requirement from the paper.
        # self.output[current_state][action_player1] is the output
        self.output: Dict[int, Dict[Tuple[int, int], int]] = {}

    def setInitialState(self, initial_state: int):
        self.initialState = initial_state

    def addTransition(self, source: int, player1_action: int, player2_action: int, output: int, target: int) -> None:
        """
        Adds a transition
        Args:
            source: int : the source state
            player1_action: str : the action of player 1
            player2_action: str : the action of player 2
            output: the output of the transition
            target: int : the target state
        """
        if source not in self.output:
            self.output[source], self.transitions[source] = {}, {}
        self.output[source][player1_action, player2_action] = output
        self.transitions[source][player1_action, player2_action] = target

    def getStates(self) -> List[int]:
        """
        Returns the states of the Mealy machine
        Returns:
            The list of the states represented by integers
        """
        return list(set(itertools.chain.from_iterable(map(lambda tr: list(tr[1].values()) + [tr[0]],
                                                          self.transitions.items()))))

    def getInitialState(self) -> int:
        """
        Returns the initial state of the Mealy machine
        Returns:
            The initial state represented by an integer
        """
        return self.initialState

    def getTransitions(self) -> Dict[int, Dict[Tuple[int, int], int]]:
        return self.transitions

    def getPlayer1Alphabet(self) -> List[int]:
        """
        Returns the alphabet of player 1
        Returns:
            The list of strings representing the alphabet of player 1
        """
        return self.player1Alphabet

    def getPlayer2Alphabet(self) -> List[int]:
        """
        Returns the alphabet of player 2
        Returns:
            The list of strings representing the alphabet of player 2
        """
        return self.player2Alphabet

    def getOutputAlphabet(self) -> List[int]:
        """
        Returns the output alphabet
        Returns:
            The list of strings representing the output alphabet
        """
        return self.outputAlphabet

    def getSuccessor(self, src: int, player1_action: int, player2_action: int) -> int:
        """
        Returns the next state
        Args:
            src: int : the source state
            player1_action: str : the action of player 1
            player2_action: str : the action of player 2
        Returns:
            The next state after the transition
        """
        return self.transitions[src][player1_action, player2_action]

    def getOutput(self, src: int, player1_action: int, player2_action: int) -> int:
        """
        Returns the output of the transition
        Args:
            src: int : the source state
            player1_action: src : the action of player 1
            player2_action: str : the action of player 2
        Returns:
            The output character
        """
        return self.output[src][player1_action, player2_action]

    def __str__(self):
        r = {}
        for source in self.getStates():
            r[source] = {}
            if source in self.transitions:
                for ((action_1, action_2), target) in self.transitions[source].items():
                    output = self.getOutput(source, action_1, action_2)
                    r[source]['({},{}) / {}'.format(action_1, action_2, output)] = target
        return str(r)

    @classmethod
    def fromMealyMachine(cls, mealy: MealyMachine,
                         alphabet_mapper: Callable[[int], Tuple[int, int]]) -> 'ReactiveSystem':
        """
        Returns the reactive system from a mealy machine
        Args:
            mealy: MealyMachine: the mealy machine to be transformed
            alphabet_mapper: Callable[[str],(str, str)] : a callable that defines how to split the alphabet
        Returns:
            A reactive system
        """
        player1_alphabet, player2_alphabet = list(
            map(list, map(set, list(zip(*map(alphabet_mapper, mealy.getInputAlphabet()))))))
        output_alphabet: List[int] = mealy.getOutputAlphabet()
        reactive_system = ReactiveSystem(player1_alphabet, player2_alphabet, output_alphabet)
        for source in mealy.getStates():
            for action in mealy.getInputAlphabet():
                try:
                    output = mealy.getOutput(source, action)
                    target = mealy.getSuccessor(source, action)
                    action1, action2 = alphabet_mapper(action)
                    reactive_system.addTransition(source, action1, action2, output, target)
                except UnknownOutputError:
                    # This transition was not explored...
                    continue

        reactive_system.setInitialState(mealy.getInitialState())
        return reactive_system
