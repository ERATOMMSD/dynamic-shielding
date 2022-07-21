import random
import time
from enum import IntEnum

import gym

from benchmarks.sidewalk.deterministic_side_walk import DeterministicSideWalk
from benchmarks.sidewalk.sidewalk_gym_wrapper import SidewalkWrapper


class RangedDeterministicSidewalk(DeterministicSideWalk):
    def __init__(self, num_seeds: int = 1, sidewalk_length: int = 12, **kwargs):
        self.sidewalk_length = sidewalk_length
        self.num_seeds: int = num_seeds
        self.default_seed = 0
        super().__init__(1, sidewalk_length, **kwargs)

    def reset(self):
        self.default_seed = (self.default_seed % self.num_seeds) + 1
        # self.default_seed = random.randrange(self.num_seeds) + 1
        return super(RangedDeterministicSidewalk, self).reset()


if __name__ == "__main__":
    from pyglet.window import key


    class AgentActions(IntEnum):
        LEFT = 0
        RIGHT = 1
        FORWARD = 2
        NONE = 3


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
            action = AgentActions.FORWARD


    def key_release(_k, _mod):
        global action
        action = AgentActions.NONE


    env = SidewalkWrapper(RangedDeterministicSidewalk(num_seeds=5, sidewalk_length=12, domain_rand=True))
    env.render()
    env.unwrapped.window.on_key_press = key_press
    env.unwrapped.window.on_key_release = key_release
    is_open = True
    record = False
    if record:
        env = gym.wrappers.Monitor(env, '/tmp', force=True)
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
            time.sleep(0.1)
    env.close()
