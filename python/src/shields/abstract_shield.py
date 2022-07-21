import random
from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import List, Set, Dict

import numpy as np

from src.exceptions.shielding_exceptions import UnsafeStateError, UnknownStateError
from src.model import SafetyGame

LOGGER = getLogger(__name__)

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.2"
__date__ = "15 September 2020"


def monitor_inconsistency(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if args[1] != result:
            LOGGER.debug(f'inconsistent: input: {args[1]}, output: {result}')
        return result

    return wrapper


class AbstractShield(metaclass=ABCMeta):
    """
    The abstract class of a shield.
    """
    win_set: Set[int]  # The winning states
    win_strategy: Dict[int, List[int]]  # win_strategy[s] is the winning actions at state s
    state: int  # the current state in the safety game
    safety_game: SafetyGame

    def __init__(self, safety_game: SafetyGame, win_set: Set[int], win_strategy: Dict[int, List[int]]) -> None:
        """
        The constructor
        """
        self.win_set = win_set
        self.win_strategy = win_strategy
        self.safety_game = safety_game
        self.state = self.safety_game.getInitialState()
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the current execution and restart the Shield state
        """
        pass

    def current_is_winning(self):
        return self.state in self.win_strategy

    def is_unexplored_state(self):
        return self.state == self.safety_game.unexplored_state

    def winning_set_size(self):
        if self.win_set:
            return len(self.win_set)
        else:
            return 0

    def preemptive(self) -> List[int]:
        """
        The method for preemptive shielding
        Returns:
          returns the list of the safe actions at the current state.
        """
        if not self.current_is_winning():
            raise UnsafeStateError("The current state is not safe according to the shield.")
        return self.win_strategy[self.state]

    @monitor_inconsistency
    def postposed(self, player1_action: int) -> int:
        """
        The method for postposed shielding
        Args:
          player1_action: str : the action by player 1
        Returns:
          returns player1_action if player1_action is safe. Otherwise returns one of the safe actions.
        """
        return self.tick(player1_action)

    def tick(self, player1_action:  int) -> int:
        """
        The method for postposed shielding
        Args:
          player1_action: str : the action by player 1
        Returns:
          returns player1_action if player1_action is safe. Otherwise returns one of the safe actions.
        """

        if type(player1_action) == int:
            player1_transformed_action: int = player1_action
        elif type(player1_action) == np.int64:
            player1_transformed_action: int = int(player1_action)
        else:
            raise ValueError(f'An action must be either int. Got {type(player1_action)}')

        if self.state in self.win_set:
            if player1_transformed_action in self.win_strategy[self.state]:
                return player1_transformed_action
            else:
                return random.choice(self.win_strategy[self.state])
        else:
            # returns player1_action when there is no safe actions
            return player1_transformed_action

    def move(self, player1_action: int, player2_action: int, output: int = None) -> None:
        """
        Move the current state of the safety game
        Args:
          player1_action: str : the action by player 1
          player2_action: str : the action by player 2
          output: str : the output of the transition
        """
        if not self.safety_game.hasSuccessor(self.state, player1_action, player2_action):
            raise UnknownStateError
        self.state = self.safety_game.getSuccessor(self.state, player1_action, player2_action)
        LOGGER.debug(f'moved to {self.state}')

        if not self.current_is_winning():
            raise UnsafeStateError("The current state is not safe according to the shield.")
