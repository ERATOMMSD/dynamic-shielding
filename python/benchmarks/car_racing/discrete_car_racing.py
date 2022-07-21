import re
from enum import IntEnum, Enum, auto
from typing import List, Callable, Tuple

import gym.spaces as spaces
import numpy as np

from benchmarks.car_racing.car_racing import CarRacing


class AgentActions(IntEnum):
    NONE = 0
    ACCEL = 1
    RIGHT = 2
    LEFT = 3
    BRAKE = 4


class ArenaPropositions(IntEnum):
    NORMAL = 0
    GRASS = 1
    CRASH = 2


class SensorPropositions(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


def toSensorPropositions(state):
    class Area(Enum):
        Road = auto()
        Grass = auto()
        Car = auto()
        Unknown = auto()

    def toArea(pixel: List[int]):
        if pixel[0] == pixel[1] and pixel[1] == pixel[2]:
            return Area.Road
        elif pixel[1] > 200:
            return Area.Grass
        elif pixel[0] > 200:
            return Area.Car
        else:
            return Area.Unknown

    if Area.Road in map(toArea, sum(map(lambda a: a.tolist(), (ls[0:32] for ls in state[35:45])), [])):
        return SensorPropositions.LEFT
    elif Area.Road in map(toArea, sum(map(lambda a: a.tolist(), (ls[64:96] for ls in state[35:45])), [])):
        return SensorPropositions.RIGHT
    else:
        return SensorPropositions.STRAIGHT


class DiscreteCarRacing(CarRacing):
    # Discretization is taken from this blog article https://notanymike.github.io/Solving-CarRacing/
    ACTION_MAP = {
        AgentActions.NONE: np.array([0.0, 0.0, 0.0]),
        AgentActions.ACCEL: np.array([0.0, 1.0, 0.0]),
        AgentActions.RIGHT: np.array([1.0, 0.0, 0.0]),
        AgentActions.LEFT: np.array([-1.0, 0.0, 0.0]),
        AgentActions.BRAKE: np.array([0.0, 0.0, 0.8])
    }
    new_tile_count: int

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(len(AgentActions))
        self.grass_history: List[int] = []
        self.crash_history: List[int] = []
        # Counters to detect the spin in the end of the episode
        self.left_spin_count: int = 0
        self.right_spin_count: int = 0
        self.total_spin_episodes: int = 0

    def is_spin(self):
        return self.right_spin_count > 20 or self.left_spin_count > 20 or (self.car is not None and self.car.is_spin)

    def reset(self):
        self.crash_history = [0] + self.crash_history
        self.grass_history = [0] + self.grass_history
        if self.is_spin():
            self.total_spin_episodes += 1
        self.left_spin_count = 0
        self.right_spin_count = 0
        self.new_tile_count = 0
        return super().reset()

    def step(self, action: int):
        if action is None:
            return super(DiscreteCarRacing, self).step(action)
        previous_tile_count = self.tile_visited_count
        total_reward = 0
        for _ in range(4):
            state, reward, done, info = super(DiscreteCarRacing, self).step(self.ACTION_MAP[AgentActions(action)])

            if self.grass is True:
                arena_proposition = ArenaPropositions.GRASS
                self.grass_history[0] += 1
            elif self.crash is True:
                arena_proposition = ArenaPropositions.CRASH
                self.crash_history[0] += 1
            else:
                arena_proposition = ArenaPropositions.NORMAL
            sensor_proposition = toSensorPropositions(state)
            total_reward += reward

        if AgentActions(action) == AgentActions.LEFT:
            self.left_spin_count += 1
        else:
            self.left_spin_count = 0
        if AgentActions(action) == AgentActions.RIGHT:
            self.right_spin_count += 1
        else:
            self.right_spin_count = 0

        info['p1_action'] = action
        info['p2_action'] = 0
        info['output'] = encode_output(arena_proposition, sensor_proposition)

        # Return the success information to compute the success rate
        info['is_success'] = self.tile_visited_count >= len(self.track)
        info['is_crash'] = self.is_spin()

        self.new_tile_count = self.tile_visited_count - previous_tile_count

        return state, total_reward, done, info


class DiscreteCarRacingWithoutSensor(CarRacing):
    # Discretization is taken from this blog article https://notanymike.github.io/Solving-CarRacing/
    ACTION_MAP = {
        AgentActions.NONE: np.array([0.0, 0.0, 0.0]),
        AgentActions.ACCEL: np.array([0.0, 1.0, 0.0]),
        AgentActions.RIGHT: np.array([1.0, 0.0, 0.0]),
        AgentActions.LEFT: np.array([-1.0, 0.0, 0.0]),
        AgentActions.BRAKE: np.array([0.0, 0.0, 0.8])
    }

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(len(AgentActions))
        self.crash_history: List[int] = []

    def reset(self):
        self.crash_history = [0] + self.crash_history
        return super().reset()

    def step(self, action: int):
        if action is None:
            return super(DiscreteCarRacingWithoutSensor, self).step(action)
        state, reward, done, info = super(DiscreteCarRacingWithoutSensor, self).step(
            self.ACTION_MAP[AgentActions(action)])

        if self.grass is True:
            arena_proposition = ArenaPropositions.GRASS
            self.crash_history[0] += 1
        elif self.crash is True:
            arena_proposition = ArenaPropositions.CRASH
            self.crash_history[0] += 1
        else:
            arena_proposition = ArenaPropositions.NORMAL

        info['p1_action'] = action
        info['p2_action'] = 0
        info['output'] = int(arena_proposition)

        return state, reward, done, info


def encode_output(arena_proposition: ArenaPropositions, sensor_proposition: SensorPropositions) -> int:
    return int(arena_proposition) + int(sensor_proposition) * len(ArenaPropositions)


def decode_output(output: int) -> Tuple[ArenaPropositions, SensorPropositions]:
    return ArenaPropositions(output % len(ArenaPropositions)), SensorPropositions(output // len(ArenaPropositions))


def evaluate_output(output_int: int) -> Callable[[str], bool]:
    arena_proposition, sensor_proposition = decode_output(output_int)
    arena_proposition_str = str(arena_proposition)
    sensor_propositions_str = str(sensor_proposition)

    match_result_arena = re.match(r'ArenaPropositions\.([A-Z]+)', arena_proposition_str)
    prop_omit_str = match_result_arena.group(1)
    match_result_sensor = re.match(r'SensorPropositions\.([A-Z]+)', sensor_propositions_str)
    sensor_omit_str = match_result_sensor.group(1)
    return lambda atomic_proposition: atomic_proposition in [arena_proposition_str, prop_omit_str,
                                                             sensor_propositions_str, sensor_omit_str]


def evaluate_output_without_sensor(output_int: int) -> Callable[[str], bool]:
    if output_int == int(ArenaPropositions.NORMAL):
        arena_proposition_str = str(ArenaPropositions.NORMAL)
    else:
        arena_proposition_str = str(ArenaPropositions.GRASS)

    match_result_arena = re.match(r'ArenaPropositions\.([A-Z]+)', arena_proposition_str)
    prop_omit_str = match_result_arena.group(1)
    return lambda atomic_proposition: atomic_proposition in [arena_proposition_str, prop_omit_str]


if __name__ == "__main__":
    from pyglet.window import key

    action = AgentActions.NONE


    def key_press(k, _mod):
        global restart
        global action
        if k == 0xFF0D:
            restart = True
        elif k == key.LEFT:
            action = AgentActions.LEFT
        elif k == key.RIGHT:
            action = AgentActions.RIGHT
        elif k == key.UP:
            action = AgentActions.ACCEL
        elif k == key.DOWN:
            action = AgentActions.BRAKE


    def key_release(_k, _mod):
        global action
        action = AgentActions.NONE


    env = DiscreteCarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    is_open = True
    while is_open:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(int(action))
            total_reward += r
            if steps % 200 == 0 or done:
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print(action)
            steps += 1
            isopen = env.render()
            if done or restart or is_open is False:
                break
    env.close()
