from copy import deepcopy
from logging import getLogger
from typing import Callable, Tuple, List, Union

from src.logic import BlueFringeRPNI
from src.model import ReactiveSystem
from src.shields.abstract_dynamic_shield import AbstractDynamicShield, UpdateShield

LOGGER = getLogger(__name__)

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.4"
__date__ = "9 October 2020"


class PTADynamicShield(AbstractDynamicShield):
    """
    The class for a shield with passive automata learning without state merging.
    Usage:
      1. make an instance pta_dynamic_shield of PTADynamicShield class giving all the necessary information
      2. when we have new observation, run pta_dynamic_shield.addSample() or pta_dynamic_shield.addSamples() to add training data
      3. run reconstructShield to construct a shield from the current training data
      4. when we want to use the preemptive shield, run pta_dynamic_shield.preemptive(), which returns the list of the safe actions
      5. when we want to use the postposed shield, run pta_dynamic_shield.postposed(player1_action), which returns player1_action if player1_action is safe. Otherwise it returns one of the safe actions. If there is no safe action, it returns playr1_action.
      6. when we move to another state by executing the system, run pta_dynamic_shield.move(player1_action, player2_action). We note that player1_action must be the actually executed action not the action before postposed shielding.
      7. at the beginning of each episode (i.e., when we reset the play and go back to the initial state of the arena), run pta_dynamic_shield.reset().
    """

    def __init__(self, ltl_formula: Union[str, List[str]],
                 player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int],
                 evaluate_output: Callable[[int], Callable[[str], bool]],
                 update_shield: UpdateShield = UpdateShield.RESET,
                 min_depth: int = 999999999999999, no_merging: bool = True):
        """
        The constructor
        Args:
          ltl_formula: str : the LTL formula for the shielded specification
          player1_alphabet: str : The actions of player1
          player2_alphabet: str : The actions of player2. Some of the actions can be missing, but this must be nonempty.
          output_alphabet: str : The outputs
          evaluate_output: Callable[[str], Callable[[str], bool]] : The function to evaluate the output of the reactive system
          update_shield: UpdateShield: specify where to reconstruct the shield
          min_depth: int : minimum depth of the state merging (by default, we do not merge states)
        """
        self.player1_alphabet: List[int] = player1_alphabet
        self.player2_alphabet: List[int] = player2_alphabet
        self.output_alphabet: List[int] = output_alphabet
        assert len(player2_alphabet) > 0, "player2_alphabet must be nonempty"
        self.learner: BlueFringeRPNI = BlueFringeRPNI(player1_alphabet, player2_alphabet, output_alphabet)
        self.min_depth: int = min_depth
        self.no_merging = no_merging
        super(PTADynamicShield, self).__init__(ltl_formula, player1_alphabet, player2_alphabet,
                                               evaluate_output, update_shield)

    def reconstruct_reactive_system(self) -> ReactiveSystem:
        if self.no_merging:
            return deepcopy(self.learner.pta)
        else:
            return self.learner.compute_model(self.min_depth)

    def addSample(self, input_word: List[Tuple[int, int]], output_action: int) -> None:
        """
        Add a training pair
        Args:
            input_word: str : an input word
            output_action: str : the expected output word for input_word
        """
        self.learner.addSample(input_word, output_action)

    def addSamples(self, training_data: List[Tuple[List[Tuple[int, int]], int]]) -> None:
        """
        Add training data
        Args:
            training_data: List[Tuple[str, str]] : an training data. A list of pairs of input and output words
        """
        for input_word, output_action in training_data:
            self.learner.addSample(input_word, output_action)

    def move(self, player1_action: int, player2_action: int, output: int) -> None:
        """
        Move the current state of the safety game
        Args:
            player1_action: str : the action by player 1 in MDP
            player2_action: str : the action by player 2 in MDP
            output: str : the output of the transition in MDP
        """
        self.addSample(self.history + [(player1_action, player2_action)], output)
        super(PTADynamicShield, self).move(player1_action, player2_action, output)
