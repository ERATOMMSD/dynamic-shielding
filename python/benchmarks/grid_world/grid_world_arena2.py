import itertools
from enum import auto, IntEnum
from typing import Tuple, Dict

from benchmarks.grid_world.deterministic_position import AbstractPosition, AgentActions, CrashPropositions
from benchmarks.grid_world.grid_world_arena_builder import GridWorldArenaBuilder
from src.model.mdp import MDP

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.2"
__date__ = "1 March 2021"


class ArenaPropositions(IntEnum):
    NOTHING = 0
    WALL = auto()
    GOAL = auto()


class ArenaMealyPropositions(IntEnum):
    NOTHING = 0
    WALL = auto()


class Position(AbstractPosition):
    error_output = ArenaPropositions.WALL

    def __init__(self, x_size: int, y_size: int, x: int = 0, y: int = 0):
        super(Position, self).__init__(x_size, y_size, x, y)
        self.output_dict = {(x, y): ArenaPropositions.NOTHING for (x, y)
                            in itertools.product(range(0, self.x_size), range(0, self.y_size))}


def make_grid_world_mdp(initial_player1_x: int = 0, initial_player1_y: int = 0,
                        initial_player2_x: int = 4, initial_player2_y: int = 4,
                        goal_x: int = 2, goal_y: int = 2) -> MDP:
    """
    Args:
        initial_player1_x: int : the _x element of the initial position of the ego agent
        initial_player1_y: int : the _y element of the initial position of the ego agent
        initial_player2_x: int : the _x element of the initial position of the other agent
        initial_player2_y: int : the _y element of the initial position of the other agent
        goal_x: int : the _x element of the goal
        goal_y: int : the _y element of the goal
    Returns: the grid world with two agents.

    The map is as follows
     xxxxxxx
     x    Ex
     x xxx x
     x xG  x
     x xxx x
     xS    x
     xxxxxxx

     - x :: wall
     - S :: ego agent's starting point
     - G :: Goal
     - E :: enemy's starting point
    """  # Size of the grid world
    x_size = 5
    y_size = 5
    builder = GridWorldArenaBuilder(Position, x_size, y_size, initial_player1_x, initial_player1_y,
                                    initial_player2_x, initial_player2_y)
    builder.player1_alphabet([int(action) for action in AgentActions])
    builder.player2_alphabet([int(action) for action in AgentActions])

    for arena_proposition in ArenaPropositions:
        for crash_proposition in CrashPropositions:
            builder.append_output_alphabet(int(arena_proposition) * len(CrashPropositions) + int(crash_proposition))

    for x in range(1, 4):
        for y in [1, 3]:
            builder.set_output(x, y, ArenaPropositions.WALL)
    builder.set_output(1, 2, ArenaPropositions.WALL)
    builder.set_output(goal_x, goal_y, ArenaPropositions.GOAL)

    return builder.build()


def decode_mdp_int_output(combined_output: int) -> Tuple[ArenaPropositions, CrashPropositions]:
    assert combined_output in range(len(ArenaPropositions) * len(CrashPropositions)), \
        f'The input of `decode_int_output` is out of the range {combined_output}'
    return (ArenaPropositions(combined_output // len(CrashPropositions)),
            CrashPropositions(combined_output % len(CrashPropositions)))


def decode_int_output(combined_output: int) -> Tuple[ArenaMealyPropositions, CrashPropositions]:
    assert combined_output in range(len(ArenaMealyPropositions) * len(CrashPropositions)), \
        f'The input of `decode_int_output` is out of the range {combined_output}'
    to_arena_propositions: Dict[int, ArenaMealyPropositions] = {
        0: ArenaMealyPropositions.NOTHING,
        1: ArenaMealyPropositions.WALL,
    }
    to_crash_propositions: Dict[int, CrashPropositions] = {
        0: CrashPropositions.NO_CRASH,
        1: CrashPropositions.CRASH
    }
    return (to_arena_propositions[combined_output // len(CrashPropositions)],
            to_crash_propositions[combined_output % len(CrashPropositions)])
