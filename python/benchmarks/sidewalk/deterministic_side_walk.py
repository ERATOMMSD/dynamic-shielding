import math
import time
from enum import IntEnum

import gym
import numpy as np
from gym_miniworld.entity import MeshEnt, Box
from gym_miniworld.envs.sidewalk import Sidewalk

from benchmarks.sidewalk.sidewalk_gym_wrapper import SidewalkWrapper


class DeterministicSideWalk(Sidewalk):
    """
    The sidewalk benchmark with deterministic arena construction
    """

    def __init__(self, default_seed: int = 1, sidewalk_length: int = 12, max_episode_steps: int = 150, **kwargs):
        self.sidewalk_length = sidewalk_length
        self.default_seed: int = default_seed
        super().__init__(**kwargs)
        self.max_episode_steps = max_episode_steps

    def reset(self):
        self.seed(self.default_seed)
        return super(DeterministicSideWalk, self).reset()

    def _gen_world(self):
        sidewalk = self.add_rect_room(
            min_x=-3, max_x=0,
            min_z=0, max_z=self.sidewalk_length,
            wall_tex='brick_wall',
            floor_tex='concrete_tiles',
            no_ceiling=True
        )

        self.street = self.add_rect_room(
            min_x=0, max_x=6,
            min_z=-80, max_z=80,
            floor_tex='asphalt',
            no_ceiling=True
        )

        self.connect_rooms(sidewalk, self.street, min_z=0, max_z=self.sidewalk_length)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(
                mesh_name='building',
                height=30
            ),
            pos=np.array([30, 0, 30]),
            dir=-math.pi
        )

        for i in range(1, sidewalk.max_z // 2):
            self.place_entity(
                MeshEnt(
                    mesh_name='cone',
                    height=0.75
                ),
                pos=np.array([1, 0, 2 * i])
            )

        self.box = self.place_entity(
            Box(color='red'),
            room=sidewalk,
            min_z=sidewalk.max_z - 2,
            max_z=sidewalk.max_z
        )

        self.place_agent(
            room=sidewalk,
            min_z=0,
            max_z=1.5
        )


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


    env = SidewalkWrapper(DeterministicSideWalk(default_seed=5, sidewalk_length=20))
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
