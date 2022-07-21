import pickle
from logging import getLogger
from typing import List, Union, Callable

from src.model.mdp import MDP
from src.shields.safe_padding import AbstractSafePadding

LOGGER = getLogger(__name__)


class StaticSafePadding(AbstractSafePadding):
    """
    This class implements the safe padding technique in [Hasanbeig+, AAMAS'20].
    """

    def __init__(self, ltl_formula: Union[str, List[str]], mdp: MDP,
                 evaluate_output: Callable[[int], Callable[[str], bool]], horizon: Callable[[int], int] = lambda _: 1,
                 rank: Callable[[int], int] = lambda _: 10000, critical_probability: float = 1.0):
        """
        Args:
          ltl_formula: str : the LTL formula for the shielded specification
          horizon: Callable[[int], int] : function from the state visitation to the horizon of the bounded computation
          rank: Callable[[int], int] : function from the state visitation to the number of the available actions
          critical_probability: float : threshold safety violation probability of critical actions
        """
        super().__init__(ltl_formula, evaluate_output, horizon, rank, critical_probability)
        self.mdp = mdp
        self.current_mdp_state: int = mdp.getInitialState()
        self.reset()

    def reset(self) -> None:
        """
        Reset the current execution and restart the Shield state
        """
        self.current_mdp_state: int = self.mdp.getInitialState()
        if isinstance(self.dfa, list):
            self.current_dfa_state = list(map(lambda dfa: dfa.getInitialState(), self.dfa))
        else:
            self.current_dfa_state = self.dfa.getInitialState()

    def move(self, player1_action: int, player2_action: int, output: int, observation: str) -> None:
        """ Move the current state of the safety game

        Args:
          player1_action: int : the action by player 1
          player2_action: int : the action by player 2
          output: int : the output of the transition
          observation: str : the observation after the transition
        """
        self.current_mdp_state = self.mdp.getSuccessorWithP2Action(self.current_mdp_state,
                                                                   player1_action, player2_action)
        self.current_dfa_state = self.get_successors_dfa(self.current_dfa_state, output)

    def _get_mdp_size(self):
        return len(self.mdp.getStates())

    def _get_player1_alphabet(self):
        return self.mdp.player1Alphabet

    def preemptive(self) -> List[int]:
        return self._compute_safe_actions(self.mdp, self.current_mdp_state)

    @staticmethod
    def create_from_pickle(ltl_formula: Union[str, List[str]],
                           pickle_filename: str,
                           evaluate_output: Callable[[int], Callable[[str], bool]]) -> "StaticSafePadding":
        """Function to create a static shield from a pickle file of a ReactiveSystem

        :param ltl_formula: the LTL formula for the shielded specification
        :param pickle_filename: The filename of the loaded pickle file
        :param evaluate_output: given a string `output` for an output of the reactive system and an string `AP`
            representing an atomic proposition, evaluate_output(output)(AP) is returns if `AP` is satisfied in `output`
        :return: The created static safe padding
        """
        with open(pickle_filename, mode='rb') as f:
            mdp: MDP = pickle.load(f)
        return StaticSafePadding(ltl_formula=ltl_formula, mdp=mdp, evaluate_output=evaluate_output)
