import itertools
from logging import getLogger
from typing import List, Dict, Tuple, Set, Callable

from src.model import DFA, ReactiveSystem

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "07 September 2020"

LOGGER = getLogger(__name__)


class SafetyGame:
    """
    The class for safety game. A finite-state reactive system is basically a Mealy machine but the input alphabet is a pair of the alphabets for player 1 and 2.
    """

    def __init__(self, player1_alphabet: List[int], player2_alphabet: List[int]) -> None:
        """
        The constructor
        Example:
            SafetyGame(["up", "right", "left", "down", "stay"], ["no_perturbe", "left_perturbe", "right_perturbe"])
        """
        self.p1_alphabet: List[int] = player1_alphabet
        self.p2_alphabet: List[int] = player2_alphabet
        self.initial_state: int = 1
        self.unexplored_state: int = 0
        # initial state and unexplored state are assumed to be safe in the beginning
        self.safeStates: List[int] = [self.unexplored_state, self.initial_state]
        # I am not too sure if this is the best data structure for the transitions
        # self.transition[(current_state, action_player1, action_player2)] is the target_state
        self.transitions: Dict[Tuple[int, int, int], int] = {}
        # self.power_transitions[src, a1] = (self.transition[(src, a1, a2)] for a2 in self.p2_alphabet)
        self.power_transitions: Dict[Tuple[int, int], Set[int]] = {}
        # The reverse mapper of the transitions
        self.reverseTransitions: Dict[Tuple[int, int, int], Set[int]] = {}
        self.inverse_state_mapper: Dict[int, Tuple[int, int]] = {}
        # adding transitions from the initial state to the unexplored states
        for (p1_action, p2_action) in itertools.product(self.p1_alphabet, self.p2_alphabet):
            self.add_transition(self.initial_state, p1_action, p2_action, self.unexplored_state)
            self.add_transition(self.unexplored_state, p1_action, p2_action, self.unexplored_state)

    def setInitialState(self, initial_state: int) -> None:
        self.initial_state = initial_state

    def setSafeStates(self, safe_states: List[int]) -> None:
        self.safeStates = safe_states

    def addSafeState(self, safe_state: int) -> None:
        self.safeStates.append(safe_state)

    def setTransitions(self, transitions: Dict[Tuple[int, int, int], int]) -> None:
        self.transitions = transitions
        self.constructReverseTransitions()

    def add_transition(self, source: int,
                       player1_action: int, player2_action: int,
                       target: int) -> None:
        """
        add a transition
        Args:
            source: int : the source state
            player1_action: str : the action of player1
            player2_action: str : the action of player2
            target: int : the target state
        """
        self.appendReverseTransition(source,
                                     player1_action, player2_action, target)
        self.transitions[(source, player1_action, player2_action)] = target
        if (source, player1_action) in self.power_transitions:
            self.power_transitions[source, player1_action].add(target)
        else:
            self.power_transitions[source, player1_action] = {target}

    def getStates(self) -> List[int]:
        """
        Returns the states of the Mealy machine
        Returns:
            The list of the states represented by integers
        """
        return list(set(itertools.chain.from_iterable(
            map(lambda transition: [transition[0][0], transition[1]], self.transitions.items()))))

    def getInitialState(self) -> int:
        """
        Returns the initial state of the Mealy machine
        Returns:
            The initial state represented by an integer
        """
        return self.initial_state

    def isUnexploredState(self, state) -> bool:
        """
        Returns a bool
        Returns:
            true if the state is the unexplored state, false otherwise
        """
        return state == self.unexplored_state

    def getPlayer1Alphabet(self) -> List[int]:
        """
        Returns the alpabet of player1
        Returns:
            The list of strings representing the alphabet of player1
        """
        return self.p1_alphabet

    def getPlayer2Alphabet(self) -> List[int]:
        """
        Returns the alpabet of player2
        Returns:
            The list of strings representing the alphabet of player2
        """
        return self.p2_alphabet

    def getSuccessor(self, src: int,
                     player1_action: int, player2_action: int) -> int:
        """
        Returns the next state
        Args:
            src: int : the source state
            player1_action: str : the action of player1
            player2_action: str : the action of player2
        Returns:
            The next state after the transition
        """
        return self.transitions[(src, player1_action, player2_action)]

    def hasSuccessor(self, src: int,
                     player1_action: int, player2_action: int) -> bool:
        """
        Returns the next state
        Args:
            src: int : the source state
            player1_action: str : the action of player1
            player2_action: str : the action of player2
        Returns:
            The true if the transition exists, false otherwise
        """
        return (src, player1_action, player2_action) in self.transitions

    def appendReverseTransition(self, src: int,
                                p1_action: int, p2_action: int,
                                tgt: int) -> None:
        if (tgt, p1_action, p2_action) in self.reverseTransitions:
            self.reverseTransitions[(tgt, p1_action, p2_action)].add(src)
        else:
            self.reverseTransitions[(tgt, p1_action, p2_action)] = {src}

    def constructReverseTransitions(self) -> None:
        self.reverseTransitions = {}
        for (src, p1_action, p2_action), tgt in self.transitions.items():
            self.appendReverseTransition(src, p1_action, p2_action, tgt)

    def getPredecessors(self, tgt: int,
                        player1_action: int, player2_action: int) -> Set[int]:
        """
        Returns the predecessor state
        Args:
            tgt: int : the target state in the transition
            player1_action: str : the action of player1
            player2_action: str : the action of player2
        Returns:
            The predecessor state before the transition
        """
        if (tgt, player1_action, player2_action) in self.reverseTransitions:
            return self.reverseTransitions[(tgt, player1_action, player2_action)]
        else:
            return set()

    def isSafe(self, state: int) -> bool:
        """
        Returns if the given state is safe or not
        Args:
            state: int : the state to be checked
        Returns:
            True if the given state is an safe state
        """
        return state in self.safeStates

    @classmethod
    def fromReactiveSystemAndDFA(cls, reactive_system: ReactiveSystem,
                                 dfa: DFA,
                                 evaluate_output: Callable[
                                     [int], Callable[[str], bool]]) -> "SafetyGameFromReactiveSystem":
        """
        Construct a safety game from a reactive system and a DFA
        Args:
            reactive_system: ReactiveSystem : the given ReactiveSystem
            dfa: DFA : the given DFA
            evaluate_output: Callable[[str], Callable[[str], bool]] : given a string `output` for an output of the reactive system and an string `AP` representing an atomic proposition, evaluate_output(output)(AP) is returns if `AP` is satisfied in `output`
        """
        safety_game = SafetyGameFromReactiveSystem(reactive_system.getPlayer1Alphabet(),
                                                   reactive_system.getPlayer2Alphabet())
        # maps the pair (reactive_system_state, added_dfa_state) to the corresponding state of safety_game
        safety_game.state_mapper = {}
        safety_game.inverse_state_mapper = {}

        # A function to add new state
        def add_state(reactive_system_state: int, added_dfa_state: int) -> int:
            if (reactive_system_state, added_dfa_state) in safety_game.state_mapper:
                LOGGER.fatal(f'{reactive_system_state} {added_dfa_state} already exists')
                raise ValueError
            added_state = len(safety_game.state_mapper) + 1
            safety_game.state_mapper[reactive_system_state, added_dfa_state] = added_state
            safety_game.inverse_state_mapper[added_state] = (reactive_system_state, added_dfa_state)
            return added_state

        # set the initial state
        safety_game.setInitialState(add_state(reactive_system.getInitialState(), dfa.getInitialState()))

        # set the transitions
        new_states: Set[int] = {safety_game.initial_state}
        while len(new_states) > 0:
            new_state: int = new_states.pop()
            (rs_state, dfa_state) = safety_game.inverse_state_mapper[new_state]
            for (p1_action, p2_action) in itertools.product(safety_game.p1_alphabet, safety_game.p2_alphabet):
                try:
                    rs_output: int = reactive_system.getOutput(rs_state, p1_action, p2_action)
                    rs_next: int = reactive_system.getSuccessor(rs_state, p1_action, p2_action)
                except KeyError:
                    # Transition to sink state (unexplored)
                    safety_game.add_transition(new_state, p1_action, p2_action, safety_game.unexplored_state)
                    continue
                dfa_successor: int = dfa.getSuccessor(dfa_state, evaluate_output(rs_output))
                if (rs_next, dfa_successor) in safety_game.state_mapper:
                    # when the target state is not new
                    next_state = safety_game.state_mapper[(rs_next, dfa_successor)]
                    safety_game.add_transition(new_state, p1_action, p2_action, next_state)
                else:
                    # when the target state is new
                    next_state = add_state(rs_next, dfa_successor)
                    new_states.add(next_state)
                    safety_game.add_transition(new_state, p1_action, p2_action, next_state)

        # set the accepting states
        safe_states = [elem[1] for elem in safety_game.state_mapper.items() if dfa.isSafe(elem[0][1])]
        if len(safe_states) > 0:
            # Do not add unexplored state as winning when the game is already losing
            safe_states.append(safety_game.unexplored_state)
            # Unexplored state has self loop with all transitions (anything may happen)
            for (p1_action, p2_action) in itertools.product(safety_game.p1_alphabet, safety_game.p2_alphabet):
                safety_game.add_transition(safety_game.unexplored_state, p1_action, p2_action,
                                           safety_game.unexplored_state)
        safety_game.setSafeStates(safe_states)
        return safety_game


class SafetyGameFromReactiveSystem(SafetyGame):
    state_mapper: Dict[Tuple[int, int], int]
