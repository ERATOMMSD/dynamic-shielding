import math
import random
import sys
from io import StringIO
from logging import getLogger

import gym
import gym.spaces

from benchmarks.water_tank.water_tank_arena import make_water_tank_mdp, water_level, switch_state, \
    AgentActions, WaterTankInputOutputManager
from src.model.mdp import MDP

LOGGER = getLogger(__name__)


class KeepWaterLevel:
    """
    The task is to survive long enough.
    """

    def __init__(self, capacity):
        self.capacity = capacity

        # The following reward is defined based on the original shielding paper.
        # Now define the reward -- it is only dependent on the state in this scenario (whether it is the state before or
        # after a transition doesn't really matter)
        normlist = []
        for wl in range(capacity):
            normlist.append(-1 * wl * (1 + math.sin(wl * 0.4 + 0.5) * 0.95))
        norm_max = max(normlist)
        norm_min = min(normlist)

        self.reward_function = lambda wl: (2 * ((-1 * wl * (1 + math.sin(wl * 0.4 + 0.5) * 0.95)) - norm_min) / (norm_max - norm_min)) - 1

    def get_reward(self, penalty: float, water_level: int) -> float:
        LOGGER.debug(f'Water Level: {water_level}')
        LOGGER.debug(f'Penalty: {penalty}')
        if penalty > 0:
            return -penalty
        elif 0 < water_level < self.capacity:
            return self.reward_function(water_level)
        raise KeyError


class WaterTankEnv(gym.Env):
    """
    Water Tank environment for OpenAI Gym
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, initial_level=10, capacity=20):
        """
        Construct the MDP with a water tank with capacity X and initial level Y with 0 < Y < X.
        The water inflow/outflow depends on the current water level and has a probabilistic behaviour.
        There is valve that can be opened or closed.
        However, once it's state changes, 3 ticks need to pass to be able to change the state of the valve.
        The role of each player is as follows.
        - Player 1: chooses when to open or close the valve
        - Player 2: defines the outflow/inflow of the water
        Args:
            initial_level: int : the initial water level
            capacity: int : the capacity of the water tank
        Returns: the water tank with initial level X and capacity Y.
        """
        super(WaterTankEnv, self).__init__()
        self.mdp: MDP = make_water_tank_mdp(initial_level, capacity)
        self.action_space = gym.spaces.Discrete(len(AgentActions))
        self.observation_space = gym.spaces.Discrete(len(self.mdp.getStates()))
        self.reward_range = [-1, 1]
        self.state: int = self.mdp.getInitialState()
        self.capacity = capacity
        self.initial_level = initial_level
        self.task = KeepWaterLevel(capacity=capacity)
        self.done = False
        self.output_history = []
        self.actions_history = []
        self.switch_violations = 0
        self.io_manager = WaterTankInputOutputManager()
        self.episodes_count = 0
        self.steps = 0
        self.successful_episodes = 0
        self.empty = 0
        self.full = 0
        self.penalty = 1.0
        self.reset()

    def reset(self):
        LOGGER.debug('Resetting')
        self.state = self.mdp.getInitialState()
        self.task = KeepWaterLevel(capacity=self.capacity)
        self.done = False
        self.steps = 0
        self.episodes_count += 1
        self.output_history.clear()
        self.actions_history.clear()
        return self.state

    def set_penalty(self, penalty: float):
        # -1 is the minimum reward given by the standard reward function
        assert penalty >= 1.0
        self.penalty = penalty

    @staticmethod
    def _seed(seed=None):
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def decode(state):
        return water_level(state), switch_state(state)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        water, switch = self.decode(self.state)
        outfile.write(f'water level {water} - switch state {switch}')
        outfile.write('\n')

    def _change_is_possible(self):
        """"
            A change is possible after keeping the water tank in the same status for three turns.
            The last three ticks were same or there was never a change.
        """
        return ((len(self.output_history) > 2 and
                 len(set(map(lambda x: self.io_manager.is_open(x), self.output_history[-3:]))) == 1) or
                len(set(map(lambda x: self.io_manager.is_open(x), self.output_history))) == 1)

    def valid_move(self, p1_action):
        """"
        Anything is valid in the beginning or
        action is same with previous one or
        we are allowed to change action.
        """
        return (len(self.output_history) == 0 or
                self.actions_history[-1][0] == p1_action or
                self._change_is_possible())

    def step(self, player1_action_int: int):

        p1_action = player1_action_int

        p2_candidates = self.mdp.getProbabilisticSuccessor(self.state, p1_action)
        # Weights encodes the probabilities of the transition
        [(p2_action, prob, successor)] = random.choices(p2_candidates, weights=list(map(lambda x: x[1], p2_candidates)), k=1)

        valid_movement = self.valid_move(p1_action)
        output = self.mdp.getOutput(self.state, p1_action, p2_action)
        self.output_history.append(output)
        self.actions_history.append((p1_action, p2_action))

        _, self.state = self.mdp.getSuccessorWithP2Action(self.state, p1_action, p2_action)

        self.steps += 1

        penalty = 0
        is_success = False
        if not valid_movement:
            self.done = True
            penalty = self.penalty
            LOGGER.info('3 Tick Violation')
            self.switch_violations += 1
        elif self.io_manager.is_full(output):
            self.done = True
            penalty = self.penalty
            self.full += 1
            LOGGER.info('Full Tank Violation')
        elif self.io_manager.is_empty(output):
            self.done = True
            penalty = self.penalty
            self.empty += 1
            LOGGER.info('Empty Tank Violation')
        elif self.spec.max_episode_steps <= self.steps:
            LOGGER.debug('Game finished due to max step')
            self.done = True
            self.successful_episodes += 1
            is_success = True

        LOGGER.debug(f'Player 1: {p1_action}')
        LOGGER.debug(f'Player 2: {p2_action}')
        LOGGER.debug(f'Output: {output}')

        water = water_level(self.state)
        reward = self.task.get_reward(penalty, water)

        LOGGER.debug(f'mdp_state: {self.state}')
        info = {'p1_action': p1_action, 'p2_action': p2_action, 'output': output,
                'is_success': is_success, 'is_crash': self.done and not is_success}
        return self.state, reward, self.done, info


class WaterTankEnvC20I10(WaterTankEnv):
    def __init__(self):
        super().__init__(initial_level=10, capacity=20)


class WaterTankEnvC50I25(WaterTankEnv):
    def __init__(self):
        super().__init__(initial_level=25, capacity=50)


class WaterTankEnvC100I50(WaterTankEnv):

    def __init__(self):
        super().__init__(initial_level=50, capacity=100)
