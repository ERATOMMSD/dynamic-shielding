from copy import deepcopy
from enum import IntEnum, auto
from typing import Tuple, Dict, Any, List


class AgentActions(IntEnum):
    RIGHT = 0
    UP = auto()
    LEFT = auto()
    DOWN = auto()
    STAY = auto()


class CrashPropositions(IntEnum):
    NO_CRASH = 0
    CRASH = auto()


class AbstractPosition:
    _x: int
    _y: int
    x_size: int
    y_size: int
    output_dict: Dict[Tuple[int, int], Any]
    error_output: Any

    def __init__(self, x_size: int, y_size: int, x: int = 0, y: int = 0):
        self.x_size = x_size
        self.y_size = y_size
        if x >= x_size or y >= y_size:
            self._x = -1
            self._y = -1
        else:
            self._x = x
            self._y = y

    @classmethod
    def error_position(cls, x_size: int, y_size: int) -> "AbstractPosition":
        pos = AbstractPosition(x_size, y_size, -1, -1)
        return pos

    def in_correct_range(self, x: int, y: int):
        return 0 <= x < self.x_size and 0 <= y < self.y_size

    def is_error(self):
        return self._x < 0 or self._y < 0 or self._x >= self.x_size or self._y >= self.y_size

    def move(self, action: AgentActions) -> "AbstractPosition":
        """
        Args:
            action: AgentActions : The action showing the moving direction
        Returns:
            The next position of the agent
        """
        position = deepcopy(self)
        position.move_assign(action)
        return position

    def safe_next(self) -> "List[Tuple[AbstractPosition, AgentActions]]":
        return list(filter(lambda position: not position[0].is_error(),
                           ((self.move(action), action) for action in AgentActions)))

    def size(self) -> int:
        return self.x_size * self.y_size + 1

    def __int__(self) -> int:
        """
        Note: the integer encoding is 0-origin
        """
        if self.is_error():
            return self.x_size * self.y_size
        else:
            return self._x + self.x_size * self._y

    def set_position(self, x: int, y: int):
        if self.in_correct_range(x, y):
            self._x = x
            self._y = y
        else:
            self._x = -1
            self._y = -1

    def set_output(self, x: int, y: int, output):
        if self.in_correct_range(x, y):
            self.output_dict[x, y] = output
        else:
            raise IndexError(f'x: {x}, y: {y}')

    def get_xy(self) -> Tuple[int, int]:
        return self._x, self._y

    def check_crash(self, another_agent: "AbstractPosition") -> CrashPropositions:
        if self.get_xy() == another_agent.get_xy():
            return CrashPropositions.CRASH
        else:
            return CrashPropositions.NO_CRASH

    def move_assign(self, action: AgentActions) -> None:
        if self.is_error():
            return
        if action == AgentActions.UP:
            self._y += 1
        elif action == AgentActions.DOWN:
            self._y -= 1
        elif action == AgentActions.LEFT:
            self._x -= 1
        elif action == AgentActions.RIGHT:
            self._x += 1
        if self.output() == self.error_output:
            self._x = -1
            self._y = -1

    def output(self):
        if self.is_error():
            return self.error_output
        else:
            return self.output_dict[self._x, self._y]
