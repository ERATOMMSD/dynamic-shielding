from enum import Enum
from benchmarks.common.generic import SimpleIOManager


class Observations(Enum):
    NORMAL = 0
    CLIFF = 1
    GOAL = 2


class AgentActions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class EnvironmentActions(Enum):
    IDLE = 0


class CliffWalkingInputOutputManager(SimpleIOManager):

    def __init__(self):
        super().__init__(AgentActions, EnvironmentActions, Observations)

    @staticmethod
    def is_cliff(output: int) -> bool:
        return Observations.CLIFF.value == output
