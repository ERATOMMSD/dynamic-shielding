import random
from typing import Dict

from gym.envs.toy_text.taxi import TaxiEnv

from benchmarks.taxi.taxi_alphabet import AgentActions, EnvironmentActions, Observations
from benchmarks.taxi.taxi_alphabet import TaxiInputOutputManager


class TaxiExt(TaxiEnv):
    """
     - map has 25 states (5 rows x 5 cols)
     - there are 4 fix locations for pickup/dropoff [(0, 0), (0, 4), (4, 0), (4, 3)]
     - a passenger index corresponds to the index of this locations (0..3) or 4 when inside the taxi
     - (render) blue passenger, magenta destination, yellow taxi
     See more info in gym.envs.toy_text.taxi.TaxiEnv
    """

    def __init__(self):
        super().__init__()
        self.crashes = 0
        self.crash_stamina = 100
        self.delivered = 0
        self.taxi_io_manager = TaxiInputOutputManager()
        self.penalty = 0.0
        # counters
        self.wrong_dropoff_count = 0
        self.wrong_pickup_count = 0
        self.broken_car_count = 0
        self.crash_count = 0
        self.episodes_count = 0
        self.reset()

    def update_counters(self, output):
        if output == Observations.SCRATCH.value:
            self.crash_count += 1
        elif output == Observations.BROKEN.value:
            self.broken_car_count += 1
            self.crash_count += 1
        elif output == Observations.WRONG_DROPOFF.value:
            self.wrong_dropoff_count += 1
        elif output == Observations.WRONG_PICKUP.value:
            self.wrong_pickup_count += 1

    def set_penalty(self, penalty: float):
        assert penalty >= 0.0
        self.penalty = penalty

    @staticmethod
    def action_name(action):
        return {0: 'south', 1: 'north', 2: 'east', 3: 'west', 4: 'pick', 5: 'drop'}[action]

    def step(self, action: int) -> (int, float, bool, Dict):

        source = self.s

        next_state, reward, done, info = super().step(action)

        info['p1_action'] = action
        info['output'] = self.output(source, action, next_state)
        info['p2_action'] = self.get_p2_action(source).value
        info['is_success'] = done
        info['is_crash'] = (info['output'] == Observations.BROKEN.value)

        self.update_counters(info['output'])

        if done:
            # +20 reward is already given by default environment
            self.delivered += 1
        elif info['output'] == Observations.BROKEN.value:
            done = True
            # Environment action (passenger) is override when there is a fatal crash
            info['p2_action'] = EnvironmentActions.BRAKE_CAR.value
            reward = -100
        elif info['output'] == Observations.SCRATCH.value:
            reward = - self.penalty

        # Add some positional information
        r, c, _, _ = self.decode(next_state)
        # info['output'] += len(Observations) * ((r % 2) * 2 + (c % 2))
        # add raw position. This is feasible because the arena is small
        info['output'] += len(Observations) * (r * 5 + c)

        return next_state, reward, done, info

    def get_p2_action(self, source):
        if self.dropoff_location(source):
            return EnvironmentActions.DROPME
        elif self.pickup_location(source):
            return EnvironmentActions.PICKME
        else:
            return EnvironmentActions.WAIT

    def passenger_in(self, state):
        _, _, p_idx, _ = self.decode(state)
        return p_idx == 4

    def pickup_location(self, state):
        row, col, p_idx, _ = self.decode(state)
        return not self.passenger_in(state) and self.locs[p_idx] == (row, col)

    def dropoff_location(self, state):
        row, col, _, d_idx = self.decode(state)
        return self.passenger_in(state) and self.locs[d_idx] == (row, col)

    def wrong_dropoff(self, source, action) -> bool:
        return action == AgentActions.DROPOFF.value and not self.dropoff_location(source)

    def wrong_pickup(self, source, action) -> bool:
        return action == AgentActions.PICKUP.value and not self.pickup_location(source)

    def successful_dropoff(self, source, action) -> bool:
        return action == AgentActions.DROPOFF.value and self.dropoff_location(source)

    def successful_pickup(self, source, action) -> bool:
        return action == AgentActions.PICKUP.value and self.pickup_location(source)

    def crash(self, source, agent_action, target):
        src_row, src_col, _, _ = self.decode(source)
        tg_row, tg_col, tg_p_idx, tg_d_idx = self.decode(target)
        return agent_action < AgentActions.PICKUP.value and src_row == tg_row and src_col == tg_col

    def output(self, source: int, action: int, target: int) -> int:

        if self.crash(source, action, target):
            self.crashes += 1
            if random.random() < self.crashes / self.crash_stamina:
                return Observations.BROKEN.value
            else:
                return Observations.SCRATCH.value

        # observations after doing pick-up/drop-off action
        if self.wrong_dropoff(source, action):
            return Observations.WRONG_DROPOFF.value
        elif self.wrong_pickup(source, action):
            return Observations.WRONG_PICKUP.value
        elif self.successful_dropoff(source, action):
            return Observations.SUCCESSFUL_DROPOFF.value
        elif self.dropoff_location(target):
            return Observations.ARRIVED_DROPOFF.value
        elif self.pickup_location(target):
            return Observations.ARRIVED_PICKUP.value
        elif self.successful_pickup(source, action):
            return Observations.SUCCESSFUL_PICKUP.value
        else:
            if self.passenger_in(target):
                return Observations.MOVED_WITH_PASSENGER.value
            else:
                return Observations.MOVED_EMPTY.value

    def reset(self):
        self.crashes = 0
        self.episodes_count += 1
        return super().reset()


class TaxiFixStart(TaxiExt):
    """
    Taxi always starts in the center, and passenger location and destination are fixed.
    "+---------+",
    "| : | : :G|",
    "| : | : : |",
    "| : :X: : |",
    "| | : | : |",
    "|Y| : | : |",
    "+---------+",
    """
    def reset(self, **kwargs):
        _ = super().reset()
        passenger_location = 2
        destination = 3
        self.s = self.encode(2, 2, passenger_location, destination)
        return self.s


class TaxiStartCenter(TaxiExt):
    """
    Taxi always starts in the center, but passenger can be in any positions.
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : :X: : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
    """

    def reset(self, **kwargs):
        _ = super().reset()
        passenger_location = random.choice(range(4))
        destination = (passenger_location + random.choice(range(1, 3))) % 4
        self.s = self.encode(2, 2, passenger_location, destination)
        return self.s
