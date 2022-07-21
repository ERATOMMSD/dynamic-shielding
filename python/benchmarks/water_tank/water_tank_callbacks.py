from src.wrappers.shield_callbacks import ShieldingCallback


class WaterTankShieldingCallback(ShieldingCallback):

    def _on_step(self):
        for idx, env in enumerate(self.locals["env"].envs):
            self.logger.record('stats/full', env.full)
            self.logger.record('stats/empty', env.empty)
            self.logger.record('stats/switch_violations', env.switch_violations)
            self.logger.record('stats/episodes', env.episodes_count)
            self.logger.record('stats/successful_episodes', env.successful_episodes)
            self.logger.record('stats/penalty', env.penalty)
            self.logger.record('stats/capacity', env.capacity)
            self.logger.record('stats/initial_level', env.initial_level)
        return super(WaterTankShieldingCallback, self)._on_step()
