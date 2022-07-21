from src.wrappers.shield_callbacks import ShieldingCallback


class TaxiShieldingCallback(ShieldingCallback):

    def _on_step(self):
        for idx, env in enumerate(self.locals["env"].envs):
            self.logger.record('stats/wrong_dropoff_count', env.wrong_dropoff_count)
            self.logger.record('stats/wrong_pickup_count', env.wrong_pickup_count)
            self.logger.record('stats/crash_count', env.crash_count)
            self.logger.record('stats/broken_count', env.broken_car_count)
            self.logger.record('stats/episodes', env.episodes_count)
            self.logger.record('stats/delivery', env.delivered)
        return super(TaxiShieldingCallback, self)._on_step()
