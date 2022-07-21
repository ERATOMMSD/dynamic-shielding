import pickle
from logging import getLogger
from typing import Callable, Tuple, List, Union, Optional

from py4j.java_gateway import JavaGateway

from src.logic import PassiveLearning
from src.logic.make_transition_cover import make_transition_cover
from src.logic.reduce_training_data import ReduceTrainingData
from src.model import ReactiveSystem, MealyMachine
from src.shields.abstract_dynamic_shield import AbstractDynamicShield, UpdateShield, ShieldLifeType

LOGGER = getLogger(__name__)

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.4"
__date__ = "9 October 2020"


class DynamicShield(AbstractDynamicShield):
    """
    The class for a shield with passive automata learning.
    Usage:
      1. make an instance dynamic_shield of DynamicShield class giving all the necessary information
      2. when we have new observation, run dynamic_shield.addSample() or dynamic_shield.addSamples() to add training data
      3. run reconstructShield to construct a shield from the current training data
      4. when we want to use the preemptive shield, run dynamic_shield.preemptive(), which returns the list of the safe actions
      5. when we want to use the postposed shield, run dynamic_shield.postposed(player1_action), which returns player1_action if player1_action is safe. Otherwise it returns one of the safe actions. If there is no safe action, it returns playr1_action.
      6. when we move to another state by executing the system, run dynamic_shield.move(player1_action, player2_action). We note that player1_action must be the actually executed action not the action before postposed shielding.
      7. at the beginning of each episode (i.e., when we reset the play and go back to the initial state of the arena), run dynamic_shield.reset().
    """

    def __init__(self, ltl_formula: Union[str, List[str]], gateway: JavaGateway,
                 alphabet_start: int, alphabet_end: int,
                 alphabet_mapper: Callable[[int], Tuple[int, int]],
                 evaluate_output: Callable[[int], Callable[[str], bool]],
                 reverse_alphabet_mapper: Callable[[int, int], int],
                 reverse_output_mapper: Callable[[int], int],
                 update_shield: UpdateShield = UpdateShield.RESET,
                 shield_life_type: ShieldLifeType = ShieldLifeType.EPISODES,
                 min_depth: int = 0, concurrent_reconstruction=False, max_shield_life=100,
                 not_use_deviating_shield=False, skip_mealy_size: int = 0):
        """
        :param ltl_formula: str : the LTL formula for the shielded specification
        :param gateway: JavaGateway : The java gateway of py4j
        :param alphabet_start: int : The beginning character of the input alphabet of the Mealy machine
        :param alphabet_end: int : The end character of the input alphabet of the Mealy machine
        :param alphabet_mapper: Callable[[str],(str, str)] : a callable that defines how to split the alphabet
        :param evaluate_output: Callable[[str], Callable[[str], bool]] : The function to evaluate the output of the reactive system
        :param reverse_alphabet_mapper: Callable[[str, str], str] : a function that combine the split action to reconstruct an action in Mealy machine
        :param reverse_output_mapper: Callable[[str], str] : a function maps the output in ReactiveSystem to the output of MealyMachine
        :param update_shield: UpdateShield: specify where to reconstruct the shield
        :param shield_life_type: ShieldLifeType: determines if the shield_life is measured in episodes or steps
        :param min_depth : int: the minimum depth we require to merge
        :param max_shield_life: int: The number of the maximum episodes/steps to refresh the learned shield. This is used only when concurrent_reconstruction = True
        :param not_use_deviating_shield: bool: Do not use the shield if the system behavior is not the same as the learned reactive system until `reset` is called.
        :param skip_mealy_size: int : We do not merge the states if the Mealy machine is smaller than this

        .. NOTE::
            The constructed alphabet includes both alphabet_start and alphabet_end. For example, if alphabet_start = 1
            and alphabet_end = 3, the constructed alphabet is [1, 2, 3].
        """
        self.learner: PassiveLearning = PassiveLearning(gateway, alphabet_start, alphabet_end,
                                                        min_depth=min_depth, skip_mealy_size=skip_mealy_size)
        self.alphabetMapper = alphabet_mapper
        self.reverse_alphabet_mapper = reverse_alphabet_mapper
        self.reverse_output_mapper = reverse_output_mapper
        player1_alphabet = []
        player2_alphabet = []
        for mealyAction in range(alphabet_start, alphabet_end + 1):
            player1_action, player2_action = self.alphabetMapper(mealyAction)
            player1_alphabet.append(player1_action)
            player2_alphabet.append(player2_action)
        player1_alphabet = list(set(player1_alphabet))
        player2_alphabet = list(set(player2_alphabet))
        super(DynamicShield, self).__init__(ltl_formula, player1_alphabet, player2_alphabet,
                                            evaluate_output, update_shield, concurrent_reconstruction,
                                            shield_life_type, max_shield_life,
                                            not_use_deviating_shield=not_use_deviating_shield)
        self.mealy: Optional[MealyMachine] = None

    def reconstruct_reactive_system(self) -> ReactiveSystem:
        self.mealy = self.learner.computeMealy()
        return ReactiveSystem.fromMealyMachine(self.mealy, self.alphabetMapper)

    def addSample(self, input_word: List[int], output_char: int) -> None:
        """
        Add a training pair
        Args:
            input_word: str : an input word
            output_char: str : the expected output word for inputWord
        """
        self.learner.addSample(input_word, output_char)

    def addSamples(self, training_data: List[Tuple[List[int], int]]) -> None:
        """
        Add training data
        Args:
            training_data: List[Tuple[List[int], int]] : an training data. A list of pairs of input and output words
        """
        self.learner.addSamples(training_data)

    def move(self, player1_action: int, player2_action: int, output: int) -> None:
        """
        Move the current state of the safety game
        Args:
            player1_action: str : the action by player 1 in MDP
            player2_action: str : the action by player 2 in MDP
            output: str : the output of the transition in MDP
        """
        mealy_actions: List[int] = []
        for (player1_history_action, player2_history_action) in self.history + [(player1_action, player2_action)]:
            mealy_actions.append(self.reverse_alphabet_mapper(player1_history_action, player2_history_action))
        self.learner.addSample(mealy_actions, self.reverse_output_mapper(output))
        super(DynamicShield, self).move(player1_action, player2_action, output)

    def return_transition_cover(self) -> List[Tuple[List[int], int]]:
        if self.future is None:
            mealy = self.learner.computeMealy()
        else:
            _ = self.future.result()
            mealy = self.mealy

        return make_transition_cover(mealy)

    def return_nonredundant_training_data(self) -> List[Tuple[List[int], int]]:
        if self.future is None:
            mealy = self.learner.computeMealy()
        else:
            _ = self.future.result()
            mealy = self.mealy
        reducer = ReduceTrainingData(mealy)
        return list(reducer.filter_redundant_samples(self.learner.getSamples()))

    def save_samples(self, pickle_filename: str) -> None:
        with open(pickle_filename, mode='wb') as f:
            pickle.dump(self.return_nonredundant_training_data() + self.return_transition_cover(), f)

    def _save_transition_cover(self, pickle_filename: str) -> None:
        with open(pickle_filename, mode='wb') as f:
            pickle.dump(self.return_transition_cover(), f)

    def save_full_samples(self, pickle_filename: str) -> None:
        with open(pickle_filename, mode='wb') as f:
            pickle.dump(self.learner.getSamples(), f)

    def load_transition_cover(self, pickle_filename: str) -> None:
        with open(pickle_filename, mode='rb') as f:
            transition_cover: List[Tuple[List[int], int]] = pickle.load(f)
            for input_word, output_char in transition_cover:
                if input_word is not None and output_char is not None:
                    self.addSample(input_word, output_char)
            # Construct a shield from the initial samples
            self.consistent_from_latest_construction = False
            tmp_concurrent_reconstruction = self.concurrent_reconstruction
            self.concurrent_reconstruction = False
            self.reconstructShield()
            self.concurrent_reconstruction = tmp_concurrent_reconstruction
            self.consistent_from_latest_construction = True
