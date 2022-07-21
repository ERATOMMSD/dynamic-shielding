import re
from enum import IntEnum
from typing import Tuple, Callable

import cv2
import gym
import numpy as np


class ArenaPropositions(IntEnum):
    NONE = 0
    CRASH = 1
    WALL = 2


class ConePropositions(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3


class GoalPropositions(IntEnum):
    UNOBSERVABLE = 0
    OBSERVABLE = 1
    NEAR = 2


class WallPropositions(IntEnum):
    OTHERWISE = 0
    CENTER = 1


def to_cone_proposition(observation) -> ConePropositions:
    column_size = observation.shape[1]
    left_observation = observation[:, :column_size // 2, :]
    right_observation = observation[:, column_size // 2:, :]
    left_orange = len(np.argwhere(cv2.inRange(left_observation, (140, 70, 10), (220, 130, 50)))) > 0
    right_orange = len(np.argwhere(cv2.inRange(right_observation, (140, 10, 10), (220, 130, 50)))) > 0
    if left_orange and right_orange:
        return ConePropositions.BOTH
    elif not left_orange and not right_orange:
        return ConePropositions.NONE
    elif not left_orange and right_orange:
        return ConePropositions.RIGHT
    else:
        return ConePropositions.LEFT


def to_goal_proposition(observation) -> GoalPropositions:
    assert observation.shape[2] == 3
    red_pixels = np.argwhere(cv2.inRange(observation, (125, 0, 0), (255, 50, 0)))
    if len(red_pixels) == 0:
        return GoalPropositions.UNOBSERVABLE
    elif len(red_pixels) < 50:
        return GoalPropositions.OBSERVABLE
    else:
        return GoalPropositions.NEAR


def to_wall_proposition(observation) -> WallPropositions:
    column_size = observation.shape[1]
    center_observation = observation[:, (2 * column_size // 5): (3 * column_size // 5), :]
    center_wall = len(np.argwhere(cv2.inRange(center_observation, (100, 70, 70), (130, 100, 90)))) > 0
    center_wall |= len(np.argwhere(cv2.inRange(center_observation, (70, 50, 35), (90, 70, 55)))) > 0
    return WallPropositions(center_wall)


class SidewalkWrapper(gym.Wrapper):
    """
    A wrapper of Sidewalk to have an access to the latest actions and the output propositions
    """
    done: bool
    crash: bool
    num_steps: int

    def __init__(self, env: gym.Env):
        self.latest_player1_action: int = 0
        self.latest_player2_action: int = 0
        self.latest_output: int = 0
        self.just_after_reset = True
        super(SidewalkWrapper, self).__init__(env)
        self.reset()

    def reset(self, **kwargs):
        self.num_steps = 0
        self.done = False
        self.crash = False
        self.just_after_reset = True
        obs = super(SidewalkWrapper, self).reset()
        if hasattr(self.env, 'num_seeds'):
            self.latest_player2_action = self.env.default_seed - 1
        else:
            self.latest_player2_action = 0
        return obs

    def step(self, action):
        position_before: np.ndarray = self.env.unwrapped.agent.pos
        dir_before: float = self.env.unwrapped.agent.dir
        obs, reward, self.done, info = self.env.step(action)
        position_after: np.ndarray = self.env.unwrapped.agent.pos
        dir_after: float = self.env.unwrapped.agent.dir
        self.num_steps += 1
        if self.done:
            self.crash = self.num_steps < self.env.max_episode_steps and reward <= 0

        self.latest_player1_action = action
        if not self.just_after_reset:
            self.latest_player2_action = 0
        if self.crash:
            arena_proposition = ArenaPropositions.CRASH
        elif (position_before == position_after).all() and dir_before == dir_after:
            arena_proposition = ArenaPropositions.WALL
        else:
            arena_proposition = ArenaPropositions.NONE
        self.latest_output = int(arena_proposition) + int(to_cone_proposition(obs)) * len(ArenaPropositions) + \
                             int(to_goal_proposition(obs)) * len(ArenaPropositions) * len(ConePropositions) + \
                             int(to_wall_proposition(obs)) * len(ArenaPropositions) * len(ConePropositions) * len(
            GoalPropositions)

        info['p1_action'] = self.latest_player1_action
        info['p2_action'] = self.latest_player2_action
        info['output'] = self.latest_output
        # Return the success information to compute the success rate
        info['is_success'] = self.done and not self.crash and self.num_steps < self.env.max_episode_steps
        info['is_crash'] = self.crash

        self.just_after_reset = False
        return obs, reward, self.done, info

    def get_latest_actions(self) -> Tuple[int, int, int]:
        """
        Returns:
            The tuple of the strings representing latest player1's action, player2's action, and the output
        """
        return self.latest_player1_action, self.latest_player2_action, self.latest_output


def evaluate_output(output_int: int) -> Callable[[str], bool]:
    arena_proposition = str(ArenaPropositions(output_int % len(ArenaPropositions)))
    cone_proposition = str(ConePropositions((output_int // len(ArenaPropositions)) % len(ConePropositions)))
    goal_proposition = str(GoalPropositions(
        ((output_int // len(ArenaPropositions)) // len(ConePropositions)) % len(GoalPropositions)))
    wall_proposition = str(WallPropositions(
        ((output_int // len(ArenaPropositions)) // len(ConePropositions)) // len(GoalPropositions)))

    match_result_arena_propositions = re.match(r'ArenaPropositions\.([A-Z]+)', arena_proposition)
    arena_proposition_no_prefix = match_result_arena_propositions.group(1)
    match_result_cone_propositions = re.match(r'ConePropositions\.([A-Z]+)', cone_proposition)
    cone_proposition_no_prefix = match_result_cone_propositions.group(1)
    match_result_goal_propositions = re.match(r'GoalPropositions\.([A-Z]+)', goal_proposition)
    goal_proposition_no_prefix = match_result_goal_propositions.group(1)
    match_result_wall_propositions = re.match(r'WallPropositions\.([A-Z]+)', wall_proposition)
    wall_proposition_no_prefix = match_result_wall_propositions.group(1)
    return lambda atomic_proposition: atomic_proposition in [arena_proposition, arena_proposition_no_prefix,
                                                             cone_proposition, cone_proposition_no_prefix,
                                                             goal_proposition, goal_proposition_no_prefix,
                                                             wall_proposition, wall_proposition_no_prefix]
