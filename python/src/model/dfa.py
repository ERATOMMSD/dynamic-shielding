import itertools
from typing import List, Tuple, Dict, FrozenSet, Callable

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "17 August 2020"


class DFA:
    """
    The DFA class to represent a specification. This class is independent of LearnLib.
    Note:
        All the integer encodings are 1-origin.
    """

    def __init__(self, alphabet: List[str]) -> None:
        """
        The constructor
        Example:
            DFA(['a', 'b', 'c', 'd'])
        """
        self.alphabet = alphabet
        self.initialState: int = 1
        self.safeStates: List[int] = []
        # self.transition[current_state][action] is the target_state
        self.transitions: Dict[int, Dict[FrozenSet[Tuple[str, bool]], int]] = {}

    def setInitialState(self, initialState: int):
        self.initialState = initialState

    def setSafeStates(self, safeStates: List[int]):
        self.safeStates = safeStates

    def addSafeState(self, safeState: int):
        self.safeStates.append(safeState)

    def setTransitions(self, transitions: Dict[int, Dict[FrozenSet[Tuple[str, bool]], int]]):
        self.transitions = transitions

    def addTransition(self, source: int, guard: Dict[str, bool], target: int) -> None:
        """
        add a transition
        Args:
            source: int : the source state
            label: str : the label of the transition
            target: int : the target state
        """
        if source not in self.transitions:
            self.transitions[source] = dict()
        self.transitions[source][frozenset(guard.items())] = target

    def getStates(self) -> List[int]:
        """
        Returns the states of the Mealy machine
        Returns:
            The list of the states represented by integers
        """
        return list(set(itertools.chain.from_iterable(
            map(lambda transition: list(map(lambda tr: tr[1], transition[1].items())) + [transition[0]],
                self.transitions.items()))))

    def getInitialState(self) -> int:
        """
        Returns the initial state of the Mealy machine
        Returns:
            The initial state represented by an integer
        """
        return self.initialState

    def getAlphabet(self) -> List[str]:
        """
        Returns the alpabet
        Returns:
            The list of strings representing the alphabet
        """
        return self.alphabet

    def getCharId(self, c: str) -> int:
        """
        Translate the character in the input alphabet to the ID integer.
        Args:
            c: str : string in the input alphabet
        Return:
            ID of LearnLib representing the character.
        """
        return self.alphabet.index(c)

    def getSinkState(self):
        return len(self.getStates()) + 1

    def getSuccessor(self, src: int, valuation: Callable[[str], bool]) -> int:
        """
        Returns the next state
        Args:
            src: int : the source state
            valuation: Callable[[str], bool] : the label of the transition
        Returns:
            The next state after the transition
        """
        if src == self.getSinkState():
            # when we are at the sink state, we stay at the sink state.
            return len(self.getStates()) + 1
        for guard_set, target in self.transitions[src].items():
            if all(map(lambda tpl: valuation(tpl[0]) == tpl[1], guard_set)):
                return target
        # if there is not successor state, go to the sink state
        return self.getSinkState()

    def isSafe(self, state: int) -> bool:
        """
        Returns if the given state is safe or not
        Args:
            state: int : the state to be checked
        Returns:
            True if the given state is an safe state
        """
        return state in self.safeStates

    def getDFA(self) -> str:
        """
        Return:
            the string representation of the DFA
        """
        # len(self.getOutputAlphabet())
        header = f'dfa {len(self.getStates())} {len(self.getAlphabet())} {1} 1 1 {len(self.getStates()) * len(self.getAlphabet())}\n'
        initialState = str(self.getInitialState())
        safeStates = ' '.join(map(str, self.safeStates))
        # TODO: What is the safe states for a Mealy machine?
        return '\n'.join([header, initialState])
