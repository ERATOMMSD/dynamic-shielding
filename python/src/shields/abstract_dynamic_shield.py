import logging
import pickle
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from logging import getLogger
from typing import List, Union, Callable, Tuple, Optional

from src.exceptions.shielding_exceptions import UnsafeStateError, UnknownStateError
from src.logic import ltl_to_dfa_spot, solve_game
from src.model import SafetyGame, DFA, ReactiveSystem
from src.shields.abstract_shield import AbstractShield

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', level=logging.WARN)
LOGGER = getLogger(__name__)

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.3"
__date__ = "20 July 2021"


class UpdateShield(Enum):
    """
    The enum to specify where to reconstruct the shield
    """
    MOVE = auto()
    RESET = auto()
    NONE = auto()


class ShieldLifeType(Enum):
    EPISODES = auto()
    STEPS = auto()


class AbstractDynamicShield(AbstractShield):
    """
    An abstract class for shields with passive automata learning.
    Usage:
      1. make an instance dynamic_shield of a class inheriting AbstractDynamicShield class giving all the necessary information
      2. when we want to use the preemptive shield, run pta_dynamic_shield.preemptive(), which returns the list of the safe actions
      3. when we want to use the postposed shield, run pta_dynamic_shield.postposed(player1_action), which returns player1_action if player1_action is safe. Otherwise it returns one of the safe actions. If there is no safe action, it returns playr1_action.
      4. when we move to another state by executing the system, run pta_dynamic_shield.move(player1_action, player2_action). We note that player1_action must be the actually executed action not the action before postposed shielding.
      5. at the beginning of each episode (i.e., when we reset the play and go back to the initial state of the arena), run pta_dynamic_shield.reset().
    """

    def __init__(self, ltl_formula: Union[str, List[str]],
                 player1_alphabet: List[int], player2_alphabet: List[int],
                 evaluateOutput: Callable[[int], Callable[[str], bool]],
                 update_shield: UpdateShield = UpdateShield.RESET, concurrent_reconstruction=False,
                 shield_life_type: ShieldLifeType = ShieldLifeType.EPISODES,
                 max_shield_life: int = 100, not_use_deviating_shield=False):

        """
        The constructor
        Args:
          ltl_formula: Union[str, List[str]] : the LTL formula for the shielded specification
          player1_alphabet: List[int] : The actions of player1
          player2_alphabet: List[int] : The actions of player2
          evaluateOutput: Callable[[int], Callable[[str], bool]] : The function to evaluate the output of the reactive system
          update_shield: UpdateShield: specify where to reconstruct the shield
          shield_life_type: ShieldLifeType: determines if the shield_life is measured in episodes or steps
          max_shield_life: int: The number of the maximum episodes/steps to refresh the learned shield. This is used only when concurrent_reconstruction = True
          not_use_deviating_shield: bool: Do not use the shield if the system behavior is not the same as the learned reactive system until `reset` is called.
        """
        if isinstance(ltl_formula, list):
            self.dfa: List[DFA] = [ltl_to_dfa_spot(formula) for formula in ltl_formula]
        else:
            self.dfa: DFA = ltl_to_dfa_spot(ltl_formula)

        self.ltl_formula = ltl_formula
        self.evaluateOutput = evaluateOutput
        self.history: List[Tuple[int, int]] = []
        self.update_shield = update_shield
        self.concurrent_reconstruction = concurrent_reconstruction
        self.future = None
        self.max_shield_life = max_shield_life
        self.current_shield_life = self.max_shield_life
        self.shield_life_type = shield_life_type
        self._debug_double_initialization_flag = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.reactive_system: Optional[ReactiveSystem] = None
        self.reactive_system_state: Optional[int] = None
        self.consistent: bool = True
        self.consistent_from_latest_construction = True
        self.not_use_deviating_shield = not_use_deviating_shield
        safety_game = SafetyGame(player1_alphabet, player2_alphabet)
        win_set, win_strategy = solve_game(safety_game)
        assert win_set == {0, 1}
        assert win_strategy == {0: player1_alphabet, 1: player1_alphabet}
        assert self.update_shield == UpdateShield.RESET
        super().__init__(safety_game, win_set, win_strategy)

    @abstractmethod
    def reconstruct_reactive_system(self) -> ReactiveSystem:
        pass

    def reconstructShield(self) -> None:
        """
        Reconstruct the shield using the current training data
        Warning: Given the same input, the produced shield might be different. This is because the mealy machine given
        to the safety game algorithm might differ due to the parallel execution of computeMealy.
        See more details in passive_learning.py.
        """
        if self.reactive_system is not None and self.consistent_from_latest_construction:
            LOGGER.info('Reactive system is reused')
            return
        if self.concurrent_reconstruction:
            if self.future is None:
                assert self._debug_double_initialization_flag is False, 'self.future should be None only once'
                self.future = self.executor.submit(self.reconstruct_reactive_system)
                self.consistent_from_latest_construction = True
                return
            elif self.future.done() or self.current_shield_life <= 0:
                LOGGER.debug('Retrieve automata learning result')
                self.reactive_system = self.future.result()
                LOGGER.debug('Submit automata reconstruction')
                self.future = self.executor.submit(self.reconstruct_reactive_system)
                self.consistent_from_latest_construction = True
                self.current_shield_life = self.max_shield_life
                self.consistent_from_latest_construction = True
            else:
                if self.shield_life_type == ShieldLifeType.EPISODES:
                    self.current_shield_life -= 1
                return
        else:
            self.reactive_system = self.reconstruct_reactive_system()
        LOGGER.info('Reactive system is updated')
        if isinstance(self.dfa, list):
            index = 0
            for dfa in self.dfa:
                self.safety_game = SafetyGame.fromReactiveSystemAndDFA(self.reactive_system, dfa, self.evaluateOutput)
                self.win_set, self.win_strategy = solve_game(self.safety_game)
                if self.safety_game.getInitialState() in self.win_set:
                    LOGGER.info(f'Enforced formula: {self.ltl_formula[index]}')
                    break
                index += 1
            if index == len(self.dfa):
                LOGGER.error(f'Failed to construct shield!! Use the previous shield')
            LOGGER.debug(f'Size of safety game: {len(self.safety_game.getStates())}')
            # move to the state in the reconstructed safety game
        else:
            self.safety_game = SafetyGame.fromReactiveSystemAndDFA(self.reactive_system, self.dfa, self.evaluateOutput)
            LOGGER.debug(f'Size of safety game: {len(self.safety_game.getStates())}')
            self.win_set, self.win_strategy = solve_game(self.safety_game)
            if self.safety_game.getInitialState() in self.win_set:
                LOGGER.info(f'Enforced formula: {self.ltl_formula}')
            else:
                LOGGER.warning(f'Failed to construct shield!!')
            # move to the state in the reconstructed safety game
        self.state = self.safety_game.getInitialState()
        for (player1Action, player2Action) in self.history:
            self.state = self.safety_game.getSuccessor(self.state, player1Action, player2Action)

    def reset(self) -> None:
        """
        Reset the current execution and restart the Shield state
        """
        LOGGER.debug(f'Latest history: {self.history}')
        self.history.clear()
        self.state = self.safety_game.getInitialState()
        self.consistent = True
        if self.reactive_system is not None:
            self.reactive_system_state = self.reactive_system.getInitialState()
        if self.update_shield == UpdateShield.RESET:
            self.reconstructShield()

    def preemptive(self) -> List[int]:
        """
        The method for preemptive shielding
        Returns:
          returns the list of the safe actions at the current state.
        """
        if (not self.not_use_deviating_shield) or self.consistent:
            return super(AbstractDynamicShield, self).preemptive()
        else:
            return self.reactive_system.getPlayer1Alphabet()

    def postposed(self, player1_action: int) -> int:
        if (not self.not_use_deviating_shield) or self.consistent:
            return super().postposed(player1_action)
        else:
            return player1_action

    def tick(self, player1_action: int) -> int:
        if (not self.not_use_deviating_shield) or self.consistent:
            return super().tick(player1_action)
        else:
            return player1_action

    def move(self, player1_action: int, player2_action: int, output: int) -> None:
        """
        Move the current state of the safety game
        Args:
            player1_action: str : the action by player 1 in MDP
            player2_action: str : the action by player 2 in MDP
            output: str : the output of the transition in MDP
        """
        self.history.append((player1_action, player2_action))
        if self.reactive_system is not None and self.reactive_system_state is not None and self.consistent:
            try:
                if output != self.reactive_system.getOutput(self.reactive_system_state, player1_action, player2_action):
                    self.consistent = False
                    self.consistent_from_latest_construction = False
                self.reactive_system_state = self.reactive_system.getSuccessor(self.reactive_system_state,
                                                                               player1_action,
                                                                               player2_action)
            except KeyError:
                self.consistent = False
                self.consistent_from_latest_construction = False
        try:
            super().move(player1_action, player2_action, output)
            if self.shield_life_type == ShieldLifeType.STEPS:
                self.current_shield_life -= 1
        except (UnsafeStateError, UnknownStateError) as error:
            LOGGER.info(error)
        if self.update_shield == UpdateShield.MOVE:
            self.reconstructShield()

    def save_pickle(self, filename: str = 'simple_reactive_system.pickle') -> None:
        """
        Function to save the ReactiveSystem to a pickle file
        :param filename: the name of the saved pickle file
        """
        with open(filename, mode='wb') as f:
            pickle.dump(self.reactive_system, f)
        LOGGER.debug(f'Reactive system is saved to {filename}')
