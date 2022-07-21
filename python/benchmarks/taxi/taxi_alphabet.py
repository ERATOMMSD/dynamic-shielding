from enum import Enum
from typing import Callable

from benchmarks.common.generic import SimpleIOManager


class Observations(Enum):
    # observations when doing moving action (up, down, left, right)
    MOVED_EMPTY = 0
    SCRATCH = 1
    BROKEN = 2
    ARRIVED_DROPOFF = 3
    ARRIVED_PICKUP = 4
    # observations after doing pick-up/drop-off action
    SUCCESSFUL_PICKUP = 5
    SUCCESSFUL_DROPOFF = 6
    WRONG_PICKUP = 7
    WRONG_DROPOFF = 8
    MOVED_WITH_PASSENGER = 9


class AgentActions(Enum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5


class EnvironmentActions(Enum):
    WAIT = 0
    PICKME = 1
    DROPME = 2
    BRAKE_CAR = 3


class TaxiInputOutputManager(SimpleIOManager):

    def __init__(self):
        super().__init__(AgentActions, EnvironmentActions, Observations)

    def evaluate_output(self, output: int) -> Callable[[str], bool]:
        """
        Only one observation is enabled at a time.
        Args:
            output: int: output value
        Returns:
            a callable mapping observations to boolean values
        """
        return lambda out: self.observations(output % len(Observations)).name == out

    def reverse_output_mapper(self, output: int) -> int:
        """
        Args:
            output: int: output value
        Returns:
            same output value (assuming that the output values of the environment are same)
        """
        return output

    def is_wrong_dropoff(self, output: int) -> bool:
        return output % len(Observations) == Observations.WRONG_DROPOFF.value

    def is_wrong_pickup(self, output: int) -> bool:
        return output % len(Observations) == Observations.WRONG_PICKUP.value

    def is_broken(self, output: int) -> bool:
        return output % len(Observations) == Observations.BROKEN.value

    def is_scratch(self, output: int) -> bool:
        return output % len(Observations) == Observations.SCRATCH.value

    def is_error(self, output: int) -> bool:
        return self.is_scratch(output) or self.is_broken(output) or \
               self.is_wrong_pickup(output) or self.is_wrong_dropoff(output)
