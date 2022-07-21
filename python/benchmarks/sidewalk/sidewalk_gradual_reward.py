from typing import List

import gym


class SidewalkGradualReward(gym.Wrapper):
    """
    Add gradual reward to ease the learning
    """
    reward_zs: List[float]

    def __init__(self, env, num_grades: int = 2):
        self.num_grades = num_grades
        super().__init__(env)

    def reset(self, **kwargs):
        self.reward_zs = [i * self.env.sidewalk_length / self.num_grades for i in
                          range(1, self.num_grades)]
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done and len(self.reward_zs) > 0 and self.reward_zs[0] <= self.env.agent.pos[2]:
            self.reward_zs.pop(0)
            reward += self.env.unwrapped._reward() / self.num_grades
        return obs, reward, done, info
