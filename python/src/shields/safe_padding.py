import pickle
from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Union, Callable, Set, Tuple, Dict

from src.logic import ltl_to_dfa_spot
from src.logic.mdp_learner import MDPLearner
from src.model import DFA
from src.model.mdp import MDP

LOGGER = getLogger(__name__)


class AbstractSafePadding(ABC):
    def __init__(self, ltl_formula: Union[str, List[str]],
                 evaluate_output: Callable[[int], Callable[[str], bool]],
                 horizon: Callable[[int], int] = lambda _: 1,
                 rank: Callable[[int], int] = lambda _: 10000,
                 critical_probability: float = 1.0):
        assert horizon(1) > 0, 'Horizon must be positive'
        if isinstance(ltl_formula, list):
            self.dfa: List[DFA] = [ltl_to_dfa_spot(formula) for formula in ltl_formula]
            self.current_dfa_state = [dfa.getInitialState() for dfa in self.dfa]
        else:
            self.dfa: DFA = ltl_to_dfa_spot(ltl_formula)
            self.current_dfa_state = self.dfa.getInitialState()
        self.evaluate_output = evaluate_output
        self.horizon = horizon
        self.rank = rank
        self.critical_probability = critical_probability

    @abstractmethod
    def _get_mdp_size(self):
        pass

    @abstractmethod
    def _get_player1_alphabet(self):
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the current execution and restart the Shield state
        """
        pass

    def is_safe_dfa_state_list(self, dfa_state_list: List[int]) -> List[bool]:
        if not isinstance(self.dfa, list):
            raise RuntimeError
        return [dfa_state_list[i] in self.dfa[i].safeStates for i in range(len(self.dfa))]

    def get_successors_dfa(self, dfa_state_list: Union[List[int], int], output: int) -> Union[List[int], int]:
        if not isinstance(self.dfa, list):
            return self.dfa.getSuccessor(dfa_state_list, self.evaluate_output(output))
        else:
            return [self.dfa[i].getSuccessor(dfa_state_list[i], self.evaluate_output(output))
                    for i in range(len(self.dfa))]

    def compute_bounded_safe_states(self, mdp: MDP, initial_mdp_state: int, initial_dfa_state, horizon: int) -> \
            Set[int]:
        safe_states = set()
        assert horizon > 0, 'Horizon must be positive'
        queue: List[Tuple] = [(initial_mdp_state, initial_dfa_state, 0)]
        while len(queue) > 0:
            mdp_state, dfa_state, distance = queue.pop(0)
            if (isinstance(dfa_state, list) and all(self.is_safe_dfa_state_list(dfa_state))) or \
                    (not isinstance(dfa_state, list) and dfa_state in self.dfa.safeStates):
                safe_states.add(mdp_state)
                if distance < horizon:
                    for player1_action in mdp.getPlayer1Alphabet():
                        for player2_action, _, target in mdp.getProbabilisticSuccessor(mdp_state, player1_action):
                            output: int = mdp.getOutput(mdp_state, player1_action, player2_action)
                            dfa_successors = self.get_successors_dfa(dfa_state, output)
                            queue.append((target, dfa_successors, distance + 1))
        return safe_states

    def bounded_min_safe_probability(self, mdp: MDP, initial_mdp_state: int, initial_dfa_state, horizon: int) \
            -> Dict[int, float]:
        assert horizon > 0, 'Horizon must be positive'

        bounded_safe_states = self.compute_bounded_safe_states(mdp, initial_mdp_state, initial_dfa_state, horizon)
        probability: Dict[int, float] = {safe_state: 1.0 for safe_state in bounded_safe_states}
        # Bellman update
        for _ in range(horizon):
            new_probability = {}
            for player1_action in mdp.getPlayer1Alphabet():
                for source_state in mdp.getStates():
                    sum_probability = 0
                    for player2_action, prob, target in mdp.getProbabilisticSuccessor(source_state, player1_action):
                        if target in probability.keys():
                            sum_probability += probability[target] * prob
                    if source_state in new_probability:
                        new_probability[source_state] = min(new_probability[source_state], sum_probability)
                    else:
                        new_probability[source_state] = sum_probability
            probability = new_probability
        return probability

    def bounded_violation_probability(self, mdp: MDP, initial_mdp_state: int, initial_dfa_state, horizon: int) -> \
            Dict[int, float]:
        assert horizon > 0, 'Horizon must be positive'
        bounded_min_safe_probability = self.bounded_min_safe_probability(mdp, initial_mdp_state, initial_dfa_state,
                                                                         horizon)
        return {player1_action: 1.0 - sum(map(lambda tpl: tpl[1] * bounded_min_safe_probability[tpl[2]],
                                              mdp.getProbabilisticSuccessor(initial_mdp_state, player1_action)))
                for player1_action in self._get_player1_alphabet()}

    def _compute_safe_actions(self, mdp: MDP, mdp_state: int) -> List[int]:
        violation_probability = self.bounded_violation_probability(mdp, mdp_state,
                                                                   self.current_dfa_state,
                                                                   self.horizon(self._get_mdp_size()))
        violation_probability = {key: violation_probability[key] for key in violation_probability.keys() if
                                 violation_probability[key] < self.critical_probability}
        rank: int = self.rank(self._get_mdp_size())
        result = list(map(lambda x: x[0],
                          sorted(violation_probability.items(),
                                 key=lambda x: x[1], reverse=True))[0:rank]) if len(violation_probability) > 0 else []
        if len(result) > 0:
            return result
        else:
            return self._get_player1_alphabet()

    @abstractmethod
    def preemptive(self) -> List[int]:
        pass

    def set_initial_observation(self, observation):
        pass

    @abstractmethod
    def move(self, player1_action: int, player2_action: int, output: int, observation: str) -> None:
        """
        Move the current state of the safety game
        Args:
          player1_action: int : the action by player 1
          player2_action: int : the action by player 2
          output: int : the output of the transition
          observation: str : the observation after the transition
        """
        pass


class SafePadding(AbstractSafePadding):
    """
    This class implements the safe padding technique in [Hasanbeig+, AAMAS'20].
    """
    mdp_learner: MDPLearner
    current_learner_state: str

    def __init__(self, ltl_formula: Union[str, List[str]],
                 player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int],
                 evaluate_output: Callable[[int], Callable[[str], bool]],
                 horizon: Callable[[int], int] = lambda _: 1,
                 rank: Callable[[int], int] = lambda _: 10000,
                 critical_probability: float = 1.0):
        """
        Args:
          ltl_formula: str : the LTL formula for the shielded specification
          horizon: Callable[[int], int] : function from the state visitation to the horizon of the bounded computation
          rank: Callable[[int], int] : function from the state visitation to the number of the available actions
          critical_probability: float : threshold safety violation probability of critical actions
        """
        super().__init__(ltl_formula,
                         evaluate_output=evaluate_output,
                         horizon=horizon, rank=rank, critical_probability=critical_probability)
        self.mdp_learner = MDPLearner(player1_alphabet, player2_alphabet, output_alphabet)
        self.reset()

    def explored_states(self) -> int:
        return len(self.mdp_learner.states)

    def _get_mdp_size(self):
        return len(self.mdp_learner.states)

    def _get_player1_alphabet(self):
        return self.mdp_learner.player1_alphabet

    def reset(self) -> None:
        """
        Reset the current execution and restart the Shield state
        """
        self.current_learner_state = self.mdp_learner.initial_observation
        if isinstance(self.dfa, list):
            self.current_dfa_state = list(map(lambda dfa: dfa.getInitialState(), self.dfa))
        else:
            self.current_dfa_state = self.dfa.getInitialState()

    def move(self, player1_action: int, player2_action: int, output: int, observation: str) -> None:
        """
        Move the current state of the safety game
        Args:
          player1_action: int : the action by player 1
          player2_action: int : the action by player 2
          output: int : the output of the transition
          observation: str : the observation after the transition
        """
        self.mdp_learner.add_sample(source=self.current_learner_state,
                                    player1_action=player1_action, player2_action=player2_action, output=output,
                                    target=observation)
        self.current_learner_state = self.mdp_learner.observation_to_str(observation)
        self.current_dfa_state = self.get_successors_dfa(self.current_dfa_state, output)

    def set_initial_observation(self, observation):
        self.mdp_learner.initial_observation = observation

    def preemptive(self) -> List[int]:
        mdp: MDP = self.mdp_learner.to_mdp()
        current_mdp_state = self.mdp_learner.reverse_states[self.current_learner_state]
        return self._compute_safe_actions(mdp, current_mdp_state)

    def save_pickle(self, pickle_filename: str) -> None:
        """
        Save the MDP as a pickle file.
        :param pickle_filename: The filename of the pickle file to save the samples.
        """
        with open(pickle_filename, mode='wb') as f:
            pickle.dump(self.mdp_learner.to_mdp(), f)
