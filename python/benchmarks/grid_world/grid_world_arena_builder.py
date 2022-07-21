import itertools
from copy import deepcopy
from typing import List, Dict, Tuple, Any, Type

from benchmarks.grid_world.deterministic_position import AgentActions, AbstractPosition, CrashPropositions
from src.model.mdp import MDP


class GridWorldArenaBuilder:
    _player1_alphabet: List[int] = [int(action) for action in AgentActions]
    _player2_alphabet: List[int] = [int(action) for action in AgentActions]
    output_alphabet: List[int]
    output_dict: Dict[Tuple[int, int], Any]
    position_type: Type

    x_size: int
    y_size: int
    initial_player1_x: int
    initial_player1_y: int
    initial_player2_x: int
    initial_player2_y: int

    def __init__(self, position_type: Type[AbstractPosition], x_size: int, y_size: int,
                 initial_player1_x: int, initial_player1_y: int, initial_player2_x: int, initial_player2_y: int):
        self.position_type = position_type
        self.x_size = x_size
        self.y_size = y_size
        self.initial_player1_x = initial_player1_x
        self.initial_player1_y = initial_player1_y
        self.initial_player2_x = initial_player2_x
        self.initial_player2_y = initial_player2_y
        self.output_alphabet = []
        self.output_dict = {}

    def player1_alphabet(self, value: List[int]) -> "GridWorldArenaBuilder":
        self._player1_alphabet = value
        return self

    def player2_alphabet(self, value: List[int]) -> "GridWorldArenaBuilder":
        self._player2_alphabet = value
        return self

    def append_output_alphabet(self, value: int) -> "GridWorldArenaBuilder":
        self.output_alphabet.append(value)
        return self

    def set_output(self, x: int, y: int, output) -> "GridWorldArenaBuilder":
        self.output_dict[x, y] = output
        return self

    @classmethod
    def state_mapper(cls, player1: AbstractPosition, player2: AbstractPosition) -> int:
        return int(player1) + int(player2) * player1.size() + 1  # state in MDP is 1-origin

    def build(self) -> MDP:
        mdp = MDP(self._player1_alphabet, self._player2_alphabet, self.output_alphabet)
        position = self.position_type(self.x_size, self.y_size)

        for (x, y), output in self.output_dict.items():
            position.set_output(x, y, output)

        # Set the transitions
        for (player1_x, player1_y, player2_x, player2_y) in itertools.product(range(-1, self.x_size),
                                                                              range(-1, self.y_size),
                                                                              range(-1, self.x_size),
                                                                              range(-1, self.y_size)):
            current_player1_pos = deepcopy(position)
            current_player1_pos.set_position(player1_x, player1_y)
            current_player2_pos = deepcopy(position)
            current_player2_pos.set_position(player2_x, player2_y)
            next_player2_list = current_player2_pos.safe_next()
            if len(next_player2_list) == 0:
                continue
            probability: float = 1.0 / len(next_player2_list)
            for player1_action in AgentActions:
                next_player1_pos = current_player1_pos.move(player1_action)
                for next_player2_pos, player2_action in next_player2_list:
                    crash_proposition: CrashPropositions = next_player1_pos.check_crash(next_player2_pos)
                    output: int = int(next_player1_pos.output()) * len(CrashPropositions) + int(crash_proposition)
                    assert output in self.output_alphabet
                    mdp.addProbabilisticTransition(self.state_mapper(current_player1_pos, current_player2_pos),
                                                   int(player1_action),
                                                   int(player2_action),
                                                   output,
                                                   probability,
                                                   self.state_mapper(next_player1_pos, next_player2_pos))

        # Set the initial state
        player1_init = deepcopy(position)
        player1_init.set_position(self.initial_player1_x, self.initial_player1_y)
        player2_init = deepcopy(position)
        player2_init.set_position(self.initial_player2_x, self.initial_player2_y)
        mdp.setInitialState(self.state_mapper(player1_init, player2_init))
        return mdp
