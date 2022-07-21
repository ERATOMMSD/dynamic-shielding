from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.wrappers.safe_padding_wrapper import SafePaddingWrapper


class SafePaddingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for idx, env in enumerate(self.locals["env"].envs):
            if isinstance(env, Monitor):
                env = env.env
            if isinstance(env, SafePaddingWrapper):
                self.logger.record('safe_padding/explored_states', env.safe_padding.explored_states())

        return True
