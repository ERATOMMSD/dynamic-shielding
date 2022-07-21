import itertools
import re
from enum import auto, IntEnum
from logging import getLogger
from typing import List, Tuple, Dict, Callable

LOGGER = getLogger(__name__)

from src.model.mdp import MDP

__author__ = "Masaki Waga <masakiwaga@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.2"
__date__ = "12 October 2020"


class AgentActions(IntEnum):
    RIGHT = 0
    UP = auto()
    LEFT = auto()
    DOWN = auto()
    STAY = auto()


class NoiseActions(IntEnum):
    NO_NOISE = 0
    NOISE = auto()


class ArenaPropositions(IntEnum):
    NOTHING = 0
    WALL = auto()
    BOMB = auto()
    RED = auto()
    BLUE = auto()
    GREEN = auto()
    YELLOW = auto()


class CrashPropositions(IntEnum):
    NO_CRASH = 0
    CRASH = auto()


def make_grid_world_mdp(initial_x: int = 0, initial_y: int = 0, position_update_noise: float = 0,
                        no_noise_for_wall: bool = False) -> MDP:
    """
    Args:
        initial_x: int : the x element of the initial position
        initial_y: int : the y element of the initial position
        position_update_noise: float : the probability of noise
        no_noise_for_wall: bool : If this is true, we do not hit the wall due to the noise
    Returns: the grid world only with one robot. player 1 is the ego robot and player 2 is the noise.
    """
    # Size of the grid world
    xsize = 9
    ysize = 9

    # ==================================
    # First, a function that computes the possible/likely
    # transitions when going from a (x,y)-cell into some
    # direction. It computes the image of the complete cell
    # and then performs probability-weighting according to
    # the areas of overlap
    def compute_succs(x: int, y: int, direction: AgentActions) -> List[Tuple[int, int, NoiseActions, float]]:
        """
        Args:
            x: int : The x coordinate of the current position
            y: int : The y coordinate of the current position
            direction: int : The action showing the moving direction
        Returns:
            The list of tuples (xpos_next, ypos_next, noise_action, prob) of the next position (xpos_next, ypos_next) with the action of the noise and the probability to go to the position.
        """

        if x < 0 or y < 0:
            # error state
            return [(-1, -1, NoiseActions.NO_NOISE, 1.0)]

        final_successors: List[Tuple[int, int, NoiseActions, float]] = []

        if direction == AgentActions.RIGHT:
            successor_normal = (x + 1, y, NoiseActions.NO_NOISE)
            successor_noise = (x + 1, y + 1, NoiseActions.NOISE)
        elif direction == AgentActions.UP:
            successor_normal = (x, y + 1, NoiseActions.NO_NOISE)
            successor_noise = (x - 1, y + 1, NoiseActions.NOISE)
        elif direction == AgentActions.LEFT:
            successor_normal = (x - 1, y, NoiseActions.NO_NOISE)
            successor_noise = (x - 1, y - 1, NoiseActions.NOISE)
        elif direction == AgentActions.DOWN:
            successor_normal = (x, y - 1, NoiseActions.NO_NOISE)
            successor_noise = (x + 1, y - 1, NoiseActions.NOISE)
        else:
            # No move
            return [(x, y, NoiseActions.NO_NOISE, 1.0)]

        noise_probability: float = position_update_noise

        # Move with noise
        if noise_probability > 0:
            if no_noise_for_wall:
                if (successor_noise[0] < 0 or successor_noise[0] >= xsize or
                    successor_noise[1] < 0 or successor_noise[1] >= ysize) or \
                        compute_output[successor_noise[0], successor_noise[1]] == ArenaPropositions.WALL:
                    noise_probability = 0
                else:
                    final_successors.append(successor_noise + (noise_probability,))
            else:
                if (successor_noise[0] < 0 or successor_noise[0] >= xsize or
                        successor_noise[1] < 0 or successor_noise[1] >= ysize):
                    final_successors.append((-1, -1, NoiseActions.NOISE, noise_probability))
                else:
                    final_successors.append(successor_noise + (noise_probability,))

        # Move without noise
        if successor_normal[0] < 0 or successor_normal[0] >= xsize or \
                successor_normal[1] < 0 or successor_normal[1] >= ysize:
            final_successors.append((-1, -1, NoiseActions.NO_NOISE, 1 - noise_probability))
        else:
            final_successors.append(successor_normal + (1 - noise_probability,))

        return final_successors

    player1_alphabet: List[int] = [int(action) for action in AgentActions]
    player2_alphabet: List[int] = [int(action) for action in NoiseActions]
    output_alphabet: List[int] = [int(action) for action in ArenaPropositions]
    mdp = MDP(player1_alphabet, player2_alphabet, output_alphabet)

    def state_mapper(x: int, y: int) -> int:
        # error state
        if x < 0 or y < 0:
            return xsize * ysize + 1
        else:
            return x + y * xsize + 1

    compute_output: Dict[Tuple[int, int], ArenaPropositions] = {(x, y): ArenaPropositions.NOTHING for (x, y) in
                                                                itertools.product(range(0, xsize), range(0, ysize))}
    for (x, y) in itertools.product(range(0, 2), range(6, 8)):
        compute_output[x, y] = ArenaPropositions.BLUE
    compute_output[8, 8] = ArenaPropositions.RED
    compute_output[2, 1] = ArenaPropositions.GREEN
    compute_output[7, 0] = ArenaPropositions.YELLOW
    compute_output[7, 1] = ArenaPropositions.YELLOW
    for x in range(2, 5):
        compute_output[x, 7] = ArenaPropositions.WALL
    for x in range(7, 9):
        compute_output[x, 7] = ArenaPropositions.WALL
    for x in range(2, 4):
        compute_output[x, 5] = ArenaPropositions.WALL
    for y in range(2, 4):
        compute_output[5, y] = ArenaPropositions.WALL
    compute_output[6, 1] = ArenaPropositions.WALL
    for x in range(7, 9):
        compute_output[x, 3] = ArenaPropositions.WALL
    for x in range(1, 4):
        compute_output[x, 2] = ArenaPropositions.WALL
    compute_output[1, 1] = ArenaPropositions.WALL
    compute_output[3, 1] = ArenaPropositions.WALL
    compute_output[0, 5] = ArenaPropositions.BOMB
    compute_output[3, 3] = ArenaPropositions.BOMB
    compute_output[4, 6] = ArenaPropositions.BOMB
    compute_output[7, 4] = ArenaPropositions.BOMB
    # Error state
    compute_output[-1, -1] = ArenaPropositions.WALL

    # Set the transitions
    for (x, y) in itertools.product(range(0, xsize), range(0, ysize)):
        for player1_action in AgentActions:
            for next_x, next_y, player2_action, probability in compute_succs(x, y, player1_action):
                if probability > 0:
                    output = compute_output[next_x, next_y]
                    mdp.addProbabilisticTransition(state_mapper(x, y),
                                                   int(player1_action),
                                                   int(player2_action),
                                                   int(output),
                                                   probability,
                                                   state_mapper(next_x, next_y))
    for player1_action in AgentActions:
        # the error state is WALL
        mdp.addProbabilisticTransition(xsize * ysize + 1, int(player1_action),
                                       int(NoiseActions.NO_NOISE),
                                       int(ArenaPropositions.WALL), 1.0, xsize * ysize + 1)

    # Set the initial state
    mdp.setInitialState(state_mapper(initial_x, initial_y))
    return mdp


def make_two_robots_grid_world_mdp(initial_ego_x: int = 0, initial_ego_y: int = 0,
                                   initial_enemy_x: int = 4, initial_enemy_y: int = 3,
                                   position_update_noise: float = 0,
                                   no_noise_for_wall: bool = False) -> MDP:
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
    ego_mdp: MDP = make_grid_world_mdp(initial_ego_x, initial_ego_y, position_update_noise, no_noise_for_wall)
    enemy_mdp = make_grid_world_mdp(initial_enemy_x, initial_enemy_y, 0)

    player1_alphabet: List[int] = ego_mdp.getPlayer1Alphabet()
    player2_alphabet: List[int] = list(range(len(AgentActions) * len(NoiseActions)))
    crash_alphabet: List[int] = [int(action) for action in CrashPropositions]
    output_alphabet: List[int] = list(
        map(lambda tpl: encode_int_output(ArenaPropositions(tpl[0]), CrashPropositions(tpl[1])),
            itertools.product(ego_mdp.getOutputAlphabet(), crash_alphabet)))
    mdp = MDP(player1_alphabet, player2_alphabet, output_alphabet)

    original_arena_size: int = len(ego_mdp.getStates())

    def convert_state(ego_position: int, enemy_position: int):
        return ego_position + (enemy_position - 1) * original_arena_size

    # define the initial state
    mdp.setInitialState(convert_state(ego_mdp.getInitialState(), enemy_mdp.getInitialState()))

    for ego_position in ego_mdp.getStates():
        for enemy_position in enemy_mdp.getStates():
            crash_proposition = CrashPropositions.CRASH if ego_position == enemy_position else CrashPropositions.NO_CRASH
            for player1_action in AgentActions:
                ego_proposition = ego_mdp.getOutput(ego_position, int(player1_action), int(NoiseActions.NO_NOISE))
                player1_successors = ego_mdp.getProbabilisticSuccessor(ego_position, int(player1_action))
                player2_successors: List[Tuple[AgentActions, int]] = []
                for player2_action in AgentActions:
                    for _, _, enemy_next_state in enemy_mdp.getProbabilisticSuccessor(enemy_position,
                                                                                      int(player2_action)):
                        # Note: in grid world benchmark, the output does not depend on the action
                        enemy_next_proposition = enemy_mdp.getOutput(enemy_next_state, int(AgentActions.STAY),
                                                                     int(NoiseActions.NO_NOISE))
                        if enemy_next_state != original_arena_size and \
                                enemy_next_proposition != ArenaPropositions.WALL and \
                                enemy_next_proposition != ArenaPropositions.BOMB:
                            # The last state is the unsafe state
                            player2_successors.append((player2_action, enemy_next_state))
                if len(player2_successors) > 0:
                    player2_probability = 1 / len(player2_successors)
                    for noise_action, player1_probability, ego_next_state in player1_successors:
                        if player1_probability > 0:
                            for player2_action, enemy_next_state in player2_successors:
                                mdp.addProbabilisticTransition(convert_state(ego_position, enemy_position),
                                                               int(player1_action),
                                                               encode_environment_action(player2_action,
                                                                                         NoiseActions(noise_action)),
                                                               encode_int_output(ArenaPropositions(ego_proposition),
                                                                                 crash_proposition),
                                                               player1_probability * player2_probability,
                                                               convert_state(ego_next_state, enemy_next_state))
    return mdp


def encode_environment_action(player2_action: AgentActions, noise_action: NoiseActions):
    return int(noise_action) + int(player2_action) * len(NoiseActions)


def encode_int_output(arena_proposition: ArenaPropositions, crash_proposition: CrashPropositions) -> int:
    return int(arena_proposition) * len(CrashPropositions) + int(crash_proposition)


def decode_int_output(combined_output: int) -> Tuple[ArenaPropositions, CrashPropositions]:
    assert combined_output in range(len(ArenaPropositions) * len(CrashPropositions)), \
        f'The input of `decode_int_output` is out of the range {combined_output}'
    to_arena_propositions: Dict[int, ArenaPropositions] = {
        0: ArenaPropositions.NOTHING,
        1: ArenaPropositions.WALL,
        2: ArenaPropositions.BOMB,
        3: ArenaPropositions.RED,
        4: ArenaPropositions.BLUE,
        5: ArenaPropositions.GREEN,
        6: ArenaPropositions.YELLOW
    }
    to_crash_propositions: Dict[int, CrashPropositions] = {
        0: CrashPropositions.NO_CRASH,
        1: CrashPropositions.CRASH
    }
    return (to_arena_propositions[combined_output // len(CrashPropositions)],
            to_crash_propositions[combined_output % len(CrashPropositions)])


def evaluate_output(output: int, _decode_int_output=decode_int_output) -> Callable[[str], bool]:
    arena_proposition, crash_proposition = _decode_int_output(output)

    arena_str = str(arena_proposition)
    crash_str = str(crash_proposition)
    match_result_arena = re.match(r'Arena[^P]*Propositions\.([A-Z]+)', arena_str)
    arena_omit_str = match_result_arena.group(1)
    match_result_crash = re.match(r'CrashPropositions\.(.+)', crash_str)
    crash_omit_str = match_result_crash.group(1)
    return lambda atomic_proposition: atomic_proposition in [arena_str, crash_str, arena_omit_str, crash_omit_str]
