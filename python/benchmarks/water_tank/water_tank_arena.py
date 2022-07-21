from enum import Enum
import math

from src.model.mdp import MDP
from benchmarks.common.generic import BooleanIOManager

__author__ = "Ezequiel Castellano <ezequiel.castellano@gmail.com>"
__status__ = "experimental"
__version__ = "0.0.1"
__date__ = "02 November 2020"


class AgentActions(Enum):
    CLOSE = 0
    OPEN = 1


class EnvironmentActions(Enum):
    NONE = 0
    NORMAL = 1
    HIGH = 2


class Observations(Enum):
    OPEN = 0    # valve is open
    EMPTY = 1   # water level <= 0
    FULL = 2    # water level >= capacity


class WaterTankInputOutputManager(BooleanIOManager):

    def __init__(self):
        super().__init__(AgentActions, EnvironmentActions, Observations)

    def is_open(self, output: int) -> bool:
        return self.evaluate_output(output)(Observations.OPEN.name)

    def is_full(self, output: int) -> bool:
        return self.evaluate_output(output)(Observations.FULL.name)

    def is_empty(self, output: int) -> bool:
        return self.evaluate_output(output)(Observations.EMPTY.name)

    def is_error(self, output: int) -> bool:
        return self.is_empty(output) or self.is_full(output)


class WaterTankPropBuilder(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.io_manager = WaterTankInputOutputManager()

    def build(self, level: int, last_action: AgentActions) -> int:
        # valuation = [OPEN, EMPTY, FULL]
        valuation = [last_action == AgentActions.OPEN, level <= 0, level >= self.capacity]
        return self.io_manager.get_output_id_from_valuation(valuation)


def water_level(state):
    return state // 7


def switch_state(state):
    return (state % 7) - 3


def make_water_tank_mdp(initial_level: int = 50, capacity: int = 100) -> MDP:
    """
    Args:
        initial_level: int : the initial water level of the tank.
        capacity: int : the capacity of the tank.
    Returns: a water tank with the specified initial water level.
    """
    water_levels = range(0, capacity + 1)
    io_manager = WaterTankInputOutputManager()
    mdp = MDP(io_manager.get_p1_actions(), io_manager.get_p2_actions(), io_manager.get_outputs())

    def state_mapper(level, switch):
        return min(max(0, level), capacity) * 7 + switch + 3

    def outflow_probability(level):
        return (math.sin(level / 12.345) + 1.0) * 0.35

    def inflow_probability(level):
        return (math.sin(level / 18 + 1.2345) + 1.0) * 0.45

    def add_openning_transitions(mdp, next_switch, op, ip, prop_builder, source, state_mapper, water_level):
        mdp.addProbabilisticTransition(source, AgentActions.OPEN.value, EnvironmentActions.HIGH.value,
                                       prop_builder.build(water_level + 2, AgentActions.OPEN),
                                       op * (1.0 - ip), state_mapper(water_level + 2, next_switch))
        mdp.addProbabilisticTransition(source, AgentActions.OPEN.value, EnvironmentActions.NORMAL.value,
                                       prop_builder.build(water_level + 1, AgentActions.OPEN),
                                       op * ip + (1.0 - op) * (1.0 - ip), state_mapper(water_level + 1, next_switch))
        mdp.addProbabilisticTransition(source, AgentActions.OPEN.value, EnvironmentActions.NONE.value,
                                       prop_builder.build(water_level, AgentActions.OPEN),
                                       (1.0 - op) * ip, state_mapper(water_level, next_switch))

    def add_closing_transitions(mdp, next_switch, op, prop_builder, source, state_mapper, water_level):
        mdp.addProbabilisticTransition(source, AgentActions.CLOSE.value, EnvironmentActions.NONE.value,
                                       prop_builder.build(water_level, AgentActions.CLOSE), op,
                                       state_mapper(water_level, next_switch))
        mdp.addProbabilisticTransition(source, AgentActions.CLOSE.value, EnvironmentActions.NORMAL.value,
                                       prop_builder.build(water_level - 1, AgentActions.CLOSE), 1.0 - op,
                                       state_mapper(water_level - 1, next_switch))

    prop_builder = WaterTankPropBuilder(capacity)

    for water_level in water_levels:
        for switch_state in [-3, -2, -1, 0, 1, 2, 3]:
            # Inflow and outflow probabilities are the ones specified in envs/water_tank/watertank.py

            op = outflow_probability(water_level)
            ip = inflow_probability(water_level)
            source = state_mapper(water_level, switch_state)

            # Switch states:
            # -3 Close, Wait 2 more turns
            # -2 Close, Wait 1 more turn
            # -1 Close, Change allowed
            # 0 Open/Close, Change allowed
            # 1 Open, Change allowed
            # 2 Open, Wait 1 more turn
            # 3 Open, Wait 2 more turns

            # Closed valve transitions
            if switch_state >= 1:
                # Open, change allowed (1) and Open, change not allowed (2, 3) mapped to arbitrary state
                next_switch = -3
                add_closing_transitions(mdp, next_switch, op, prop_builder, source, state_mapper, water_level)
            else:
                # Closed, switch state increases until change allowed (-1)
                next_switch = min(-1, switch_state + 1)
                add_closing_transitions(mdp, next_switch, op, prop_builder, source, state_mapper, water_level)

            # Open valve transitions
            if switch_state <= -1:
                # Closed, change allowed (-1) and Open, change not allowed (-2, -3) mapped to arbitrary state
                next_switch = 3
                add_openning_transitions(mdp, next_switch, op, ip, prop_builder, source, state_mapper, water_level)
            else:
                # Open, switch state decreases until change allowed (1)
                next_switch = max(1, switch_state - 1)
                add_openning_transitions(mdp, next_switch, op, ip, prop_builder, source, state_mapper, water_level)

    # Set the initial state
    mdp.setInitialState(state_mapper(initial_level, 0))

    return mdp
