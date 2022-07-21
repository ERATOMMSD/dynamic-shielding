import os
from logging import getLogger

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.shields import SafePadding, AdaptiveDynamicShield
from src.shields.evaluation_shield import EvaluationShield
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import AbstractShieldWrapper

LOGGER = getLogger(__name__)

SHIELD_PICKLE_NAME: str = "best_shield.pickle"
SAFE_PADDING_PICKLE_NAME: str = "best_safe_padding.pickle"


class ShieldingCallback(BaseCallback):
    def __init__(self, verbose=0):

        super().__init__(verbose)

    def _on_step(self) -> bool:
        for idx, env in enumerate(self.locals["env"].envs):
            if hasattr(env, 'p1_post_action'):
                # make sure that the learning agent knows when we rewrote the action
                self.locals["actions"][idx] = env.p1_post_action
            while not isinstance(env, AbstractShieldWrapper) and hasattr(env, 'env'):
                env = env.env
            if isinstance(env, AbstractShieldWrapper):
                self.logger.record('shield/winning_at_start', env.winning_at_start_count)
                self.logger.record('shield/losing_at_start', env.losing_at_start_count)
                self.logger.record('shield/winning_strategy_size', env.shield.winning_set_size())
                self.logger.record('shield/unexplored_at_end', env.unexplored_count)
                self.logger.record('shield/min_depth', env.min_depth)
                self.logger.record('shield/skip_mealy_size', env.skip_mealy_size)
                self.logger.record('shield/max_shield_life', env.max_shield_life)
                self.logger.record('shield/reactive_system_size', env.reactive_system_size)
                self.logger.record('shield/safety_game_size', env.safety_game_size)
                if isinstance(env.shield, AdaptiveDynamicShield):
                    self.logger.record('shield/factor', env.shield.factor)

        return True


class SaveBestShieldCallback(BaseCallback):
    def __init__(self):
        super(SaveBestShieldCallback, self).__init__()

    def _on_step(self) -> bool:
        """
        The function called when the evaluated model is the best one so far.
        """
        assert isinstance(self.parent, EvalCallback), "SaveBestShieldCallback should only be used with EvalCallback"
        if self.parent.best_model_save_path is not None:
            shield = None
            safe_padding = None
            if isinstance(self.parent.eval_env, AbstractShieldWrapper) and \
                    isinstance(self.parent.eval_env.shield, EvaluationShield):
                shield = self.parent.eval_env.shield
            elif hasattr(self.parent.eval_env, 'unwrapped') and \
                    hasattr(self.parent.eval_env.unwrapped, 'envs') and \
                    isinstance(self.parent.eval_env.unwrapped.envs[0], AbstractShieldWrapper) and \
                    isinstance(self.parent.eval_env.unwrapped.envs[0].shield, EvaluationShield):
                shield = self.parent.eval_env.unwrapped.envs[0].shield
            elif isinstance(self.parent.eval_env, SafePaddingWrapper) and \
                    isinstance(self.parent.eval_env.safe_padding, SafePadding):
                safe_padding = self.parent.eval_env.safe_padding
            elif hasattr(self.parent.eval_env, 'unwrapped') and \
                    hasattr(self.parent.eval_env.unwrapped, 'envs') and \
                    isinstance(self.parent.eval_env.unwrapped.envs[0], SafePaddingWrapper) and \
                    isinstance(self.parent.eval_env.unwrapped.envs[0].safe_padding, SafePadding):
                safe_padding = self.parent.eval_env.unwrapped.envs[0].safe_padding
            if shield is not None:
                save_path = os.path.join(self.parent.best_model_save_path, SHIELD_PICKLE_NAME)
                LOGGER.info(f"Saving best shield to {save_path}")
                shield.save_pickle(save_path)
            elif safe_padding is not None:
                save_path = os.path.join(self.parent.best_model_save_path, SAFE_PADDING_PICKLE_NAME)
                LOGGER.info(f"Saving best safe padding to {save_path}")
                safe_padding.save_pickle(save_path)
            else:
                LOGGER.info('No shield or safe padding to save')

        return True
