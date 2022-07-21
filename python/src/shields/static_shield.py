import pickle
from typing import Callable, Union, List

from src.logic import ltl_to_dfa_spot, solve_game
from src.model import ReactiveSystem, SafetyGame, DFA
from src.shields.abstract_shield import AbstractShield

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "8 October 2020"


class StaticShield(AbstractShield):
    """
    The class for a shield WITHOUT automata learning.

    Usage:
      1. make an instance static_shield of StaticShield class giving all the necessary information
      2. when we want to use the preemptive shield, run static_shield.preemptive(), which returns the list of the safe actions
      3. when we want to use the postposed shield, run static_shield.postposed(player1_action), which returns player1_action if player1_action is safe. Otherwise it returns one of the safe actions. If there is no safe action, it returns playr1_action.
      4. when we move to another state by executing the system, run static_shield.move(player1_action, player2_action). We note that player1_action must be the actually executed action not the action before postposed shielding.
      5. at the beginning of each episode (i.e., when we reset the play and go back to the initial state of the arena), run static_shield.reset().
    """

    def __init__(self, ltl_formula: Union[str, List[str]], reactive_system: ReactiveSystem,
                 evaluate_output: Callable[[int], Callable[[str], bool]]):
        """
        The constructor

        Args:
          ltl_formula: str : the LTL formula for the shielded specification
          reactive_system: ReactiveSystem : the model of the arena
          evaluate_output: Callable[[str], Callable[[str], bool]] :  given a string `output` for an output of the
          reactive system and an string `AP` representing an atomic proposition, evaluate_output(output)(AP) is returns
          if `AP` is satisfied in `output`
        """
        if isinstance(ltl_formula, list):
            for formula in ltl_formula:
                dfa: DFA = ltl_to_dfa_spot(formula)
                safety_game = SafetyGame.fromReactiveSystemAndDFA(reactive_system, dfa, evaluate_output)
                win_set, win_strategy = solve_game(safety_game)
                if safety_game.getInitialState() in win_set:
                    super().__init__(safety_game, win_set, win_strategy)
                    break
        else:
            dfa: DFA = ltl_to_dfa_spot(ltl_formula)
            safety_game = SafetyGame.fromReactiveSystemAndDFA(reactive_system, dfa, evaluate_output)
            win_set, win_strategy = solve_game(safety_game)
            super().__init__(safety_game, win_set, win_strategy)

    def reset(self) -> None:
        self.state = self.safety_game.getInitialState()

    @staticmethod
    def create_from_pickle(ltl_formula: Union[str, List[str]],
                           pickle_filename: str,
                           evaluate_output: Callable[[int], Callable[[str], bool]]) -> "StaticShield":
        """Function to create a static shield from a pickle file of a ReactiveSystem

        :param ltl_formula: the LTL formula for the shielded specification
        :param pickle_filename: The filename of the loaded pickle file
        :param evaluate_output: given a string `output` for an output of the reactive system and an string `AP`
            representing an atomic proposition, evaluate_output(output)(AP) is returns if `AP` is satisfied in `output`
        :return: The created static shield
        """
        with open(pickle_filename, mode='rb') as f:
            reactive_system: ReactiveSystem = pickle.load(f)
        return StaticShield(ltl_formula=ltl_formula, reactive_system=reactive_system, evaluate_output=evaluate_output)
