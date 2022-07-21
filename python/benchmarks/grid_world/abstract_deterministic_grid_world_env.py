import random
from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Tuple, Optional

import gym
import gym.spaces

from benchmarks.grid_world.grid_world_arena_builder import AgentActions
from src.model.mdp import MDP

LOGGER = getLogger(__name__)


class OutputProposition(ABC):
    _crash: bool
    _hit_wall: bool

    def __init__(self, _crash: bool, _hit_wall: bool) -> None:
        self._crash = _crash
        self._hit_wall = _hit_wall

    def crash(self) -> bool:
        return self._crash

    def hit_wall(self) -> bool:
        return self._hit_wall


class AbstractDeterministicGridWorldEnv(gym.Env, ABC):
    """
    Grid World environment for OpenAI Gym
    """
    metadata = {'render.modes': ['human', 'ansi']}
    mdp: MDP

    def __init__(self, mdp: MDP):
        super(AbstractDeterministicGridWorldEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(len(AgentActions))
        self.mdp = mdp
        self.observation_space = gym.spaces.Discrete(len(self.mdp.getStates()) + 1)
        self.reward_range = [-1.0, 1.0]
        self.state: int = 0
        self.task = self.construct_task()
        self.done = False
        self.step_counter: int = 0
        self.MAX_STEP: int = 50
        self.latest_player1_action: Optional[int] = None
        self.latest_player2_action: Optional[int] = None
        self.latest_output: Optional[int] = None
        self.latest_output_proposition: Optional[OutputProposition] = None
        self.reset()

    def reset(self):
        self.state = self.mdp.getInitialState()
        self.task = self.construct_task()
        self.done = False
        self.step_counter = 0
        return self.state

    @abstractmethod
    def construct_task(self):
        pass

    def _seed(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def render(self, mode='human'):
        pass

    @abstractmethod
    def decode_output(self, output: int) -> OutputProposition:
        pass

    def step(self, player1_action: int):
        self.latest_player1_action = player1_action
        successors: List[Tuple[int, float, int]] = self.mdp.getProbabilisticSuccessor(self.state, player1_action)
        random_choice: float = random.random()
        reward = None
        for player2_action, probability, target in successors:
            random_choice -= probability
            if random_choice <= 0:
                self.latest_player2_action = player2_action
                self.latest_output = self.mdp.getOutput(self.state, player1_action, player2_action)
                self.latest_output_proposition = self.decode_output(self.translate_output(self.latest_output))
                if self.latest_output_proposition.hit_wall() or self.latest_output_proposition.crash():
                    penalty = 1
                    self.done = True
                    LOGGER.info('dead')
                else:
                    penalty = 0.5 / self.MAX_STEP

                reward = self.task.get_reward(penalty, self.latest_output)
                if self.task.done:
                    self.done = True
                    LOGGER.info(f'step_counter: {self.step_counter}')
                self.state = target
                break
        assert random_choice <= 0, 'no transition was used'
        LOGGER.debug(f'mdp_state: {self.state}')
        self.step_counter += 1
        if self.step_counter >= self.MAX_STEP:
            LOGGER.debug('Game finished due to max step')
            self.done = True
        info = {'p1_action': self.latest_player1_action,
                'p2_action': self.latest_player2_action,
                'output': self.translate_output(self.latest_output),
                'is_success': self.task.done,
                'is_crash': self.latest_output_proposition.hit_wall() or self.latest_output_proposition.crash()}
        return self.state, reward, self.done, info

    def get_latest_actions(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Returns:
            The tuple of the strings representing latest player1's action, player2's action, and the output
        """
        return self.latest_player1_action, self.latest_player2_action, self.latest_output

    def translate_output(self, mdp_output: int) -> int:
        """
        Translate MDP's output to the Mealy machine's output
        """
        return mdp_output
