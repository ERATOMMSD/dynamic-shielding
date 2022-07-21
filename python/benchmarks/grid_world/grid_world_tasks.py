from abc import ABC, abstractmethod
from typing import List

import benchmarks.grid_world.grid_world_arena as grid_world_arena
import benchmarks.grid_world.grid_world_arena2 as arena2


class GridWorldTask(ABC):
    done: bool

    @abstractmethod
    def get_reward(self, penalty: float, mdp_output: int) -> float:
        pass


class VisitAllColors(GridWorldTask):
    """
    The task is to visit the colored are in the following order.
        RED -> YELLOW -> GREEN -> BLUE
    """

    def __init__(self):
        self.target_colors: List[grid_world_arena.ArenaPropositions] = [grid_world_arena.ArenaPropositions.RED,
                                                                        grid_world_arena.ArenaPropositions.YELLOW,
                                                                        grid_world_arena.ArenaPropositions.GREEN,
                                                                        grid_world_arena.ArenaPropositions.BOMB]
        self.current_target_color: grid_world_arena.ArenaPropositions = self.target_colors[0]
        self.done = False

    def get_reward(self, penalty: float, mdp_output: int) -> float:
        reward: float = -penalty
        arena_proposition, _ = grid_world_arena.decode_int_output(mdp_output)
        if self.current_target_color == arena_proposition:
            index: int = self.target_colors.index(self.current_target_color)
            reward += 1.0
            if index == len(self.target_colors) - 1:
                self.current_target_color = self.target_colors[0]
                self.done = True
            else:
                self.current_target_color = self.target_colors[index + 1]
        return reward


class VisitAllColorsIgnoreOrder(GridWorldTask):
    """
    The task is to visit the colored areas
      {RED, YELLOW, GREEN, BLUE}
    """

    def __init__(self):
        self.target_colors: List[str] = ['RED', 'YELLOW', 'GREEN', 'BLUE']
        self.current_target_colors: List[str] = self.target_colors
        self.done = False

    def get_reward(self, penalty: float, mdp_output: int) -> float:
        reward: float = -penalty
        for target_color in self.current_target_colors:
            if target_color in mdp_output:
                reward += 1.0
                self.current_target_colors.remove(target_color)
                break
        self.done = len(self.current_target_colors) == 0
        return reward


class VisitOneColor(GridWorldTask):
    """
    The task is to visit one of the following color: RED, YELLOW, GREEN, and BLUE.
    """

    def __init__(self):
        self.target_colors: List[grid_world_arena.ArenaPropositions] = [grid_world_arena.ArenaPropositions.RED,
                                                                        grid_world_arena.ArenaPropositions.YELLOW,
                                                                        grid_world_arena.ArenaPropositions.GREEN,
                                                                        grid_world_arena.ArenaPropositions.BLUE]
        self.done = False

    def get_reward(self, penalty: float, mdp_output: int) -> float:
        reward: float = -penalty

        arena_proposition, _ = grid_world_arena.decode_int_output(mdp_output)
        if arena_proposition in self.target_colors:
            reward += 1.0
            self.done = True
        return reward


class SurviveLong(GridWorldTask):
    """
    The task is to survive long enough.
    """

    def __init__(self, length: int = 10000):
        self.length: int = length
        self.alive_turn: int = 0
        self.done = False

    def get_reward(self, penalty: float, mdp_output: int) -> float:
        reward: float = -penalty
        self.alive_turn += 1
        if self.alive_turn >= self.length:
            reward += 1.0
            self.done = True
        return reward


class VisitGoal(GridWorldTask):
    """
    The task is to visit the goal in GridWorld2
    """

    def __init__(self, huge_penalty: bool = False):
        self.huge_penalty = huge_penalty
        self.done = False

    def get_reward(self, penalty: float, mdp_output: int) -> float:
        reward: float = -penalty
        arena_proposition, crash_propsition = arena2.decode_mdp_int_output(mdp_output)
        if arena_proposition == arena2.ArenaPropositions.GOAL:
            reward += 1.0
            self.done = True
        if self.huge_penalty:
            if arena_proposition == arena2.ArenaPropositions.WALL or crash_propsition == arena2.CrashPropositions.CRASH:
                reward -= 0.1
                self.done = True
        return reward
