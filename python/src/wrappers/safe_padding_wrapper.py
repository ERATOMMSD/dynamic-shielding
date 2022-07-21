from logging import getLogger
from typing import List

import gym

from src.exceptions.shielding_exceptions import UnsafeStateError
from src.shields.safe_padding import AbstractSafePadding

LOGGER = getLogger(__name__)


class SafePaddingWrapper(gym.Wrapper):
    disabled_actions: List[int]

    def __init__(self, env, safe_padding: AbstractSafePadding, debug=False):
        super().__init__(env)
        self.env = env
        self.safe_padding = safe_padding
        self.debug = debug

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.safe_padding.set_initial_observation(observation)
        self.safe_padding.reset()

        return observation

    def step(self, action):
        # Execute Action
        next_state, reward, done, info = self.env.step(action)

        # Update safe padding
        self.safe_padding.move(info['p1_action'], info['p2_action'], info['output'], next_state)

        return next_state, reward, done, info

    def get_shield_disabled_actions(self):
        """
        This method returns the actions that are not allowed by the safe padding
        It assumes that the actions space is Discrete(n).
        """
        try:
            allowed = self.safe_padding.preemptive()
            self.disabled_actions = [action for action in range(self.env.action_space.n) if action not in allowed]
        except UnsafeStateError as e:
            LOGGER.fatal(f'We are in an unsafe state according to the shield, which should not happen...')
            if not self.no_pdb:
                import pdb
                pdb.set_trace()
                raise e
            else:
                return []
        return self.disabled_actions
