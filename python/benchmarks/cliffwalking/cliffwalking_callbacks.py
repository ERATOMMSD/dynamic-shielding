from src.wrappers.shield_callbacks import ShieldingCallback
import numpy as np


class CliffWalkingShieldingCallback(ShieldingCallback):

    def _on_step(self):
        for idx, env in enumerate(self.locals["env"].envs):
            self.logger.record('stats/cliff', env.cliff)
            self.logger.record('stats/goal', env.goal)
            self.logger.record('stats/explored', np.mean(env.recent_exploration[-100:]) if env.recent_exploration else 0.0)
            self.logger.record('stats/episodes', env.episodes_count)
            self.logger.record('stats/penalty', env.penalty)
        return super(CliffWalkingShieldingCallback, self)._on_step()
