from logging import getLogger
from typing import Dict

import benchmarks.grid_world.grid_world_arena2 as arena2
from benchmarks.grid_world.abstract_deterministic_grid_world_env import AbstractDeterministicGridWorldEnv, \
    OutputProposition
from benchmarks.grid_world.deterministic_position import CrashPropositions
from benchmarks.grid_world.grid_world_tasks import VisitGoal

LOGGER = getLogger(__name__)


class GridWorldEnv2(AbstractDeterministicGridWorldEnv):
    """
    Grid World2 environment for OpenAI Gym
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, huge_penalty: bool = False, goal_x: int = 2, goal_y: int = 2):
        self.huge_penalty: bool = huge_penalty
        super(GridWorldEnv2, self).__init__(arena2.make_grid_world_mdp(goal_x=goal_x, goal_y=goal_y))
        self.reward_range = [-1.0, 1.0]

    def construct_task(self):
        return VisitGoal(self.huge_penalty)

    def decode_output(self, output: int) -> OutputProposition:
        arena_proposition, crash_proposition = arena2.decode_int_output(output)
        return OutputProposition(arena_proposition == arena2.ArenaMealyPropositions.WALL,
                                 crash_proposition == CrashPropositions.CRASH)

    def translate_output(self, mdp_output: int) -> int:
        translate_arena_proposition: Dict[arena2.ArenaPropositions, arena2.ArenaMealyPropositions] = {
            arena2.ArenaPropositions.NOTHING: arena2.ArenaMealyPropositions.NOTHING,
            arena2.ArenaPropositions.GOAL: arena2.ArenaMealyPropositions.NOTHING,
            arena2.ArenaPropositions.WALL: arena2.ArenaMealyPropositions.WALL
        }

        mdp_arena_proposition = arena2.ArenaPropositions(mdp_output // len(CrashPropositions))
        crash_proposition = CrashPropositions(mdp_output % len(CrashPropositions))
        mealy_arena_proposition = translate_arena_proposition[mdp_arena_proposition]
        return int(mealy_arena_proposition) * len(CrashPropositions) + int(crash_proposition)
