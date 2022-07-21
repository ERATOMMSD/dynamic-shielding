from abc import ABC
from logging import getLogger
from typing import Optional, List

import gym

from src.exceptions.shielding_exceptions import UnsafeStateError
from src.shields import DynamicShield
from src.shields.abstract_dynamic_shield import AbstractDynamicShield

LOGGER = getLogger(__name__)


class AbstractShieldWrapper(gym.Wrapper, ABC):
    def __init__(self, env, shield, punish=False, debug=False):
        super().__init__(env)
        self.env = env
        self.punish = punish
        # specify shield
        self.shield = shield
        self.shield.reset()
        self.debug = debug
        # counts the number of times the episode finishes in unexplored state
        self.unexplored_count = 0
        self.winning_at_start_count = 0
        self.losing_at_start_count = 0
        self.reactive_system_size = 0
        self.safety_game_size = 0
        # information on the DynamicShield
        if isinstance(self.shield, DynamicShield):
            self.min_depth = self.shield.learner.min_depth
            self.max_shield_life = self.shield.max_shield_life
            self.skip_mealy_size = self.shield.learner.skip_mealy_size
        else:
            self.min_depth = 0
            self.max_shield_life = 0
            self.skip_mealy_size = 0

    def step(self, action):
        # Execute Action
        next_state, reward, done, info = self.env.step(action)

        # Update Shield
        self.update_shield(info['p1_action'], info['p2_action'], info['output'], done)

        # TODO Punishment goes here

        return next_state, reward, done, info

    def update_shield(self, p1_action, p2_action, output, done):
        old_state = None
        if not self.shield.is_unexplored_state() and \
                hasattr(self.shield.safety_game, 'inverse_state_mapper') and \
                self.shield.state in self.shield.safety_game.inverse_state_mapper:
            # The unexplored state is not in inverse_state_mapper of safety game
            old_state = self.shield.safety_game.inverse_state_mapper[self.shield.state]
        try:
            self.shield.move(p1_action, p2_action, output)
            if done:
                self.reset_shield()
        except Exception as e:
            LOGGER.error(f'Moving Shield failed. Cannot make move {p1_action} || {p2_action} in state {old_state}')
            import pdb
            pdb.set_trace()
            raise e
        if isinstance(self.shield, DynamicShield):
            self.min_depth = self.shield.learner.min_depth
            self.max_shield_life = self.shield.max_shield_life
            self.skip_mealy_size = self.shield.learner.skip_mealy_size
        else:
            self.min_depth = 0
            self.max_shield_life = 0
            self.skip_mealy_size = 0

    def reset(self, **kwargs):
        self.shield.reset()
        return super(AbstractShieldWrapper, self).reset(**kwargs)

    def reset_shield(self):
        # Adding information for statistics
        if self.shield.is_unexplored_state():
            self.unexplored_count += 1
        # Reset the shield
        # Note that env is also reset when done is true before base callback
        self.shield.reset()
        # Adding information for statistics
        if self.shield.current_is_winning():
            self.winning_at_start_count += 1
        else:
            self.losing_at_start_count += 1
        if isinstance(self.shield, DynamicShield):
            if self.shield.mealy is not None:
                self.reactive_system_size = len(self.shield.mealy.getStates())
        self.safety_game_size = len(self.shield.safety_game.getStates())


class PostposedShieldWrapper(AbstractShieldWrapper):

    def __init__(self, env, shield, punish=False, debug=False):
        # last action needs to be remembered for callback
        self.p1_post_action = None
        super().__init__(env, shield, punish, debug)

    def step(self, action: int):
        self.p1_post_action = self.shield.postposed(action)
        if self.debug and action != self.p1_post_action:
            print(f'Shielded Observes: {action}')
            print(f'Shielded Proposes: {self.p1_post_action}')
            self.env.render()

        return super().step(self.p1_post_action)


class PreemptiveShieldWrapper(AbstractShieldWrapper):
    def __init__(self, env, shield, punish=False, debug=False, no_pdb=False):
        # actions that will by disabled by preemptive shield
        self.disabled_actions = None
        self.no_pdb = no_pdb
        super().__init__(env, shield, punish, debug)

    def get_shield_disabled_actions(self):
        """
        This method returns the actions that are not allowed by the shield
        It assumes that the actions space is Discrete(n).
        """
        try:
            allowed = self.shield.preemptive()
            assert len(allowed) > 0, 'No actions are allowed by the shield'
            self.disabled_actions = [action for action in range(self.env.action_space.n) if action not in allowed]
            # The following assertion may fail when the alphabet in the arena and the dynamic shield are inconsistent.
            assert len(self.disabled_actions) < self.env.action_space.n, \
                'All actions are disabled by the shield. ' \
                'Please check that the alphabet in the arena and the dynamic shield are consistent'
        except UnsafeStateError as e:
            LOGGER.fatal(f'We are in an unsafe state according to the shield, which should not happen...')
            if not self.no_pdb:
                import pdb
                pdb.set_trace()
                raise e
            else:
                return []
        return self.disabled_actions

    def step(self, action):
        """ This method makes sure that we actually do not receive an unwanted action."""
        assert len(self.disabled_actions) < self.env.action_space.n, 'All actions are disabled by the shield'
        if action in self.disabled_actions:
            print(self.disabled_actions)
            raise Exception(
                "We wanted to block action {}, but it was still injected. How could this happen?".format(action))
        return super().step(action)


class BlockingShieldWrapper(AbstractShieldWrapper):
    """ Postposed shield blocking the execution of unsafe actions

    **********
    Motivation
    **********

    By using a preemptive shield, we can make the training safer and we can train a good controller with the
    corresponding shield. However, from time to time, the synthesized controller does not performs well without the
    shield. The motivation of blocking shield is to transfer the knowledge of the shield to the synthesized controller.

    **********
    Idea
    **********

    The idea of blocking shielding is much like the preemptive shield but it punishes an unsafe action. To do so, we allow
    the learner to choose an unsafe action but we return a negative reward without executing the unsafe action if an unsafe
    action is chosen. This means that, unlike preemptive shield, the set of safe actions is not given to the learner.

    From the implementation viewpoint, blocking shield is closer to post-posed shield but it blocks the action (i.e. does
    not execute the environment) and return a negative reward if the given action is unsafe.
    """

    def __init__(self, env, shield, punish=True, debug=False, punish_reward: Optional[float] = None,
                 punish_reward_factor: float = 0.01, block_and_reset: bool = False):
        """ The constructor
        :param env:
        :param shield:
        :param punish:
         When this is False, we only block the execution of unsafe actions and the reward is 0.
        :param debug:
        :param punish_reward:
         The amount of the reward given to unsafe actions. If punish_reward is None, we use an adaptive reward.
        :param punish_reward_factor:
         The factor of the punished in adaptive reward.
         The adaptive reward is -abs(punish_reward_factor * average_reward).
        :param block_and_reset:
         If this is True, we finish the episode when an unsafe action is given.
        """
        self.block_and_reset = block_and_reset
        self.punish_reward = punish_reward
        self.reward_history: List[float] = []
        self.step_reward: List[float] = []
        self.state = None
        self.punish_reward_factor = punish_reward_factor
        super().__init__(env, shield, punish, debug)

    def step(self, action):
        """ Execute the system only if the action is safe. Otherwise, it returns a negative reward.
        """
        # Get the set of safe actions
        safe_actions = self.shield.preemptive()
        if action in safe_actions:
            # Execute the system with a safe action
            self.state, reward, done, info = super().step(action)
            self.step_reward.append(reward)
            return self.state, reward, done, info
        else:
            # Do not execute the system if the given action is unsafe
            if self.punish and self.punish_reward is not None:
                reward = self.punish_reward
            elif self.punish and self.punish_reward is None and len(self.reward_history) > 0:
                # We define the reward based on the previous reward.
                reward = -abs(self.punish_reward_factor * sum(self.reward_history) / len(self.reward_history))
            else:
                reward = 0
            self.step_reward.append(reward)
            if self.block_and_reset:
                self.reset_shield()
            return self.state, reward, self.block_and_reset, dict()

    def reset(self, **kwargs):
        self.state = super(BlockingShieldWrapper, self).reset(**kwargs)
        if len(self.step_reward) > 0:
            self.reward_history.append(sum(self.step_reward))
        self.step_reward = []
        if len(self.reward_history) > 50:
            self.reward_history.pop(0)
        return self.state


class SaveBestShieldWrapper(gym.Wrapper):
    """
    Wrapper for evaluation shield. We need this shield for evaluation shield but the usual wrapper is also necessary.
    """

    def __init__(self, env, shield: AbstractDynamicShield,
                 best_shield_pickle_name: str = 'best_shield.pickle'):
        super().__init__(env)
        self.shield = shield
        self.best_shield_pickle_name = best_shield_pickle_name
        self.best_reward = -float('inf')
        self.episode_reward = 0

    def reset(self, **kwargs):
        self.best_reward = max(self.best_reward, self.episode_reward)
        self.episode_reward = 0
        return super(SaveBestShieldWrapper, self).reset(**kwargs)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        self.episode_reward += reward
        if done and self.best_reward < self.episode_reward:
            self.shield.save_pickle(self.best_shield_pickle_name)

        return next_state, reward, done, info
