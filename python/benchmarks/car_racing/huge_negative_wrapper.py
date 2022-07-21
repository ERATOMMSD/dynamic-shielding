import gym


class HugeNegativeWrapper(gym.Wrapper):
    """
        The wrapper to give huge negative reward for grass entrance
    """
    HUGE_NEGATIVE_REWARD: int = -100

    def step(self, action):
        # Execute Action
        state, reward, done, info = self.env.step(action)

        # Give penalty to grass entrance
        if self.env.grass is True:
            reward += self.HUGE_NEGATIVE_REWARD

        return state, reward, done, info
