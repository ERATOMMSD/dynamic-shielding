import random
from abc import abstractmethod, ABC
from logging import getLogger
from typing import List, Tuple, Dict

import gym
import gym.spaces
import gym.spaces

from benchmarks.grid_world.grid_world_arena import make_two_robots_grid_world_mdp, AgentActions, decode_int_output, \
    ArenaPropositions, CrashPropositions
from benchmarks.grid_world.grid_world_tasks import GridWorldTask, VisitAllColorsIgnoreOrder
from benchmarks.grid_world.grid_world_tasks import VisitAllColors, VisitOneColor, SurviveLong
from src.model.mdp import MDP

LOGGER = getLogger(__name__)


class AbstractGridWorldEnv(gym.Env, ABC):
    """
    Grid World environment for OpenAI Gym
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, initial_ego_x: int = 0, initial_ego_y: int = 0,
                 initial_enemy_x: int = 4, initial_enemy_y: int = 3,
                 position_update_noise: float = 0,
                 no_noise_for_wall: bool = True):
        """
        Construct the MDP with two robots.
        The enemy robot takes a uniformly random actions with no noise avoiding the walls and error state.
        The role of each player is as follows.
        - Player 1: plays ego robot
        - Player 2: plays the environment and the enemy robot
        Args:
            initial_ego_x: int : the x element of the initial position of the ego robot
            initial_ego_y: int : the y element of the initial position of the ego robot
            initial_enemy_x: int : the x element of the initial position of the enemy robot
            initial_enemy_y: int : the y element of the initial position of the enemy robot
            position_update_noise: float : the probability of noise
            no_noise_for_wall: bool : If this is True, we do not hit the wall due to the noise
        Returns: the grid world only with one robot. player 1 is the ego robot and player 2 is the noise.
        """
        super(AbstractGridWorldEnv, self).__init__()
        self.mdp: MDP = make_two_robots_grid_world_mdp(initial_ego_x, initial_ego_y, initial_enemy_x, initial_enemy_y,
                                                       position_update_noise, no_noise_for_wall)
        self.action_space = gym.spaces.Discrete(len(AgentActions))
        self.observation_space = gym.spaces.Discrete(len(self.mdp.getStates()))
        self.reward_range = [-1, 1]
        self.state: int = 0
        self.task: GridWorldTask = self.construct_task()
        self.done = False
        self.bomb_counter: int = 0
        self.step_counter: int = 0
        self.MAX_STEP: int = 100000
        self.reset()

    def reset(self):
        self.state = self.mdp.getInitialState()
        self.task = self.construct_task()
        self.done = False
        self.bomb_counter: int = 0
        self.step_counter = 0
        return self.state

    @abstractmethod
    def construct_task(self) -> GridWorldTask:
        pass

    def _seed(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def render(self, mode='human', close=None):
        pass

    def step(self, player1_action: int):
        self.latest_player1_action: int = player1_action
        successors: List[Tuple[int, float, int]] = self.mdp.getProbabilisticSuccessor(self.state, player1_action)
        random_choice: float = random.random()
        reward = None
        for player2_action, probability, target in successors:
            random_choice -= probability
            if random_choice <= 0:
                self.latest_player2_action: int = player2_action
                self.latest_output = self.mdp.getOutput(self.state, player1_action, player2_action)
                latest_output_arena_proposition, latest_output_crash_proposition = decode_int_output(self.latest_output)

                if latest_output_arena_proposition == ArenaPropositions.BOMB:
                    self.bomb_counter += 1
                else:
                    self.bomb_counter = 0
                if self.bomb_counter == 4 or (ArenaPropositions.WALL == latest_output_arena_proposition) or (
                        CrashPropositions.CRASH == latest_output_crash_proposition):
                    penalty = 1
                    self.done = True
                    if self.bomb_counter == 4:
                        LOGGER.info('dead due to BOMB')
                    elif latest_output_arena_proposition == ArenaPropositions.WALL:
                        LOGGER.info('dead due to WALL')
                    elif latest_output_crash_proposition == CrashPropositions.CRASH:
                        LOGGER.info('dead due to CRASH')
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
        if self.done:
            LOGGER.info('done')
        return self.state, reward, self.done, {}

    def get_latest_actions(self) -> Tuple[int, int, int]:
        """
        Returns:
            The tuple of the strings representing latest player1's action, player2's action, and the output
        """
        return self.latest_player1_action, self.latest_player2_action, self.latest_output


def int_to_str(int_action: int) -> str:
    from_int: Dict[int, AgentActions] = {
        0: AgentActions.RIGHT,
        1: AgentActions.UP,
        2: AgentActions.LEFT,
        3: AgentActions.DOWN,
        4: AgentActions.STAY,
    }
    return str(from_int[int_action])


def str_to_int(str_action: str) -> int:
    from_str: Dict[str, int] = {
        "RIGHT": 0,
        "UP": 1,
        "LEFT": 2,
        "DOWN": 3,
        "STAY": 4,
    }
    for key in from_str.keys():
        if key in str_action:
            return from_str[key]
    raise KeyError


class GridWorldEnv(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        return VisitAllColors()


class GridWorldEnvIgnoreOrder(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        return VisitAllColorsIgnoreOrder()


class GridWorldEnvOneColor(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        return VisitOneColor()


class GridWorldEnvBlue(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        task = VisitOneColor()
        task.target_colors = ['BLUE']
        return task


class GridWorldEnvYellow(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        task = VisitOneColor()
        task.target_colors = ['YELLOW']
        return task


class GridWorldEnvGreen(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        task = VisitOneColor()
        task.target_colors = ['GREEN']
        return task


class GridWorldEnvRed(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        task = VisitOneColor()
        task.target_colors = ['RED']
        return task


class GridWorldEnvSurviveLong(AbstractGridWorldEnv):
    def construct_task(self) -> GridWorldTask:
        return SurviveLong()
