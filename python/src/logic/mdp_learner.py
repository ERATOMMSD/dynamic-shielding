import itertools
import pickle
from typing import Tuple, Dict, List, Union

import numpy as np

from src.model.mdp import MDP


class MDPLearner:
    # (source, player1_action, player2_action) -> target -> (count, output)
    samples: Dict[Tuple[str, int, int], Dict[str, Tuple[int, int]]]
    __initial_observation: str
    states: List[str]
    reverse_states: Dict[str, int]

    def __init__(self, player1_alphabet: List[int], player2_alphabet: List[int], output_alphabet: List[int]) -> None:
        self.player1_alphabet: List[int] = player1_alphabet
        self.player2_alphabet: List[int] = player2_alphabet
        self.output_alphabet: List[int] = output_alphabet
        self.samples = {}
        self.__initial_observation = ''
        self.states = []
        self.reverse_states = {}
        self.mdp = None

    def _add_to_list(self, observation: str):
        if observation not in self.reverse_states:
            self.reverse_states[observation] = len(self.states)
            self.states.append(observation)

    @property
    def initial_observation(self) -> str:
        return self.__initial_observation

    @staticmethod
    def observation_to_str(observation) -> str:
        if isinstance(observation, np.ndarray):
            return str(observation.tolist())
        elif isinstance(observation, str):
            return observation
        else:
            return str(observation)

    def add_state(self, observation: str) -> int:
        if observation not in self.reverse_states:
            self.reverse_states[observation] = len(self.states)
            self.states.append(observation)
        return self.reverse_states[observation]

    @initial_observation.setter
    def initial_observation(self, initial_observation):
        initial_observation_str: str = self.observation_to_str(initial_observation)
        self.__initial_observation = initial_observation_str
        self._add_to_list(initial_observation_str)
        self.add_state(initial_observation_str)

    def add_sample(self, source: Union[str, np.ndarray], player1_action: int, player2_action: int, output: int,
                   target: Union[str, np.ndarray]):
        self.mdp = None
        source_str: str = self.observation_to_str(source)
        target_str: str = self.observation_to_str(target)
        self.add_state(source_str)
        self.add_state(target_str)

        if (source_str, player1_action, player2_action) in self.samples:
            if target_str in self.samples[source_str, player1_action, player2_action]:
                count, output = self.samples[source_str, player1_action, player2_action][target_str]
                self.samples[source_str, player1_action, player2_action][target_str] = (count + 1, output)
            else:
                self.samples[source_str, player1_action, player2_action][target_str] = (1, output)
        else:
            self.samples[source_str, player1_action, player2_action] = {target_str: (2, output)}

    def to_mdp(self) -> MDP:
        if self.mdp is not None:
            return self.mdp
        self.mdp = MDP(self.player1_alphabet, self.player2_alphabet, self.output_alphabet)
        self.mdp.setInitialState(self.reverse_states[self.initial_observation])
        for source in self.states:
            for player1_action, player2_action in itertools.product(self.player1_alphabet, self.player2_alphabet):
                if (source, player1_action, player2_action) in self.samples:
                    total_count: int = sum(
                        self.samples[source, player1_action, player2_action][label][0] for label in
                        self.samples[source, player1_action, player2_action])
                    for target in self.samples[source, player1_action, player2_action]:
                        count, output = self.samples[source, player1_action, player2_action][target]
                        self.mdp.addProbabilisticTransition(self.reverse_states[source],
                                                       player1_action=player1_action, player2_action=player2_action,
                                                       output=output,
                                                       target=self.reverse_states[target],
                                                       probability=count / total_count)

        return self.mdp

    def save_pickle(self, pickle_filename: str) -> None:
        """
        Save the samples of the learner as a pickle file.
        :param pickle_filename: The filename of the pickle file to save the samples.
        """
        # We make the samples shorter to reduce the size of the saved pickle file
        saved_samples: Dict[Tuple[str, int, int], Dict[str, Tuple[int, int]]] = dict()
        for source, player1_action, player2_action in self.samples.keys():
            source_state = str(self.reverse_states[source])
            if (source_state, player1_action, player2_action) not in saved_samples:
                saved_samples[source_state, player1_action, player2_action] = dict()
            for target in self.samples[source, player1_action, player2_action]:
                target_state = str(self.reverse_states[target])
                count, output = self.samples[source, player1_action, player2_action][target]
                saved_samples[source_state, player1_action, player2_action][target_state] = (count, output)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(saved_samples, f)

    def load_pickle(self, pickle_filename: str) -> None:
        """
        Load the samples from the pickle file.
        :param pickle_filename: The filename of the pickle file to load the samples.
        """
        with open(pickle_filename, 'rb') as f:
            self.samples = pickle.load(f)
            for source_str, player1_action, player2_action in self.samples.keys():
                self.add_state(source_str)
                for target_str in self.samples[source_str, player1_action, player2_action].keys():
                    self.add_state(target_str)
