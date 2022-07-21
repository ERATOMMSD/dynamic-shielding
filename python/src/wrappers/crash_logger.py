from abc import ABC, abstractmethod
from typing import List

import gym
from stable_baselines3.common.callbacks import BaseCallback


class AbstractCrashLoggingWrapper(gym.Wrapper, ABC):
    """
    Custom Wrapper to maintain the history of the episodes.
    """
    crash_history: List[bool]

    def __init__(self, verbose=0):
        super(AbstractCrashLoggingWrapper, self).__init__(verbose)
        self.crash_history = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            self.crash_history = [self.is_crash()] + self.crash_history
        return obs, reward, done, info

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def is_crash(self):
        pass


class CrashLoggingCallback(BaseCallback):
    """
    Custom callback for plotting the number of crash in tensorboard.
    """

    def __init__(self, verbose=0, mean_duration: int = 100):
        self.mean_duration = mean_duration
        super(CrashLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for idx, env in enumerate(self.locals["env"].envs):
            if len(env.crash_history) > 0:
                self.logger.record('episodes', len(env.crash_history))
                self.logger.record('crash_episodes', sum(env.crash_history))
                self.logger.record(f'{self.mean_duration} ep mean failure rate',
                                   sum(env.crash_history[0:self.mean_duration]) /
                                   len(env.crash_history[0:self.mean_duration]))
        return True
