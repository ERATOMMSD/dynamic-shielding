from stable_baselines3.common.callbacks import BaseCallback


class CarRacingLoggingCallback(BaseCallback):
    """
    Custom callback for logging of car-racing
    """
    mean_duration: int = 10

    def __init__(self, verbose=0):
        super(CarRacingLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        for idx, env in enumerate(self.locals["env"].envs):
            if hasattr(env, 'tile_visited_count_history'):
                self.logger.record('tile_visited_count', env.tile_visited_count)
                if len(env.tile_visited_count_history) > 0:
                    self.logger.record(f'{self.mean_duration} ep mean # of visited tiles',
                                       sum(env.tile_visited_count_history[-1 - self.mean_duration:]) /
                                       len(env.tile_visited_count_history[-1 - self.mean_duration:]))
            if hasattr(env, 'grass_history'):
                self.logger.record('grass_count', sum(env.grass_history))
                if len(env.grass_history) > 0:
                    self.logger.record(f'{self.mean_duration} ep mean # of steps in grass',
                                       sum(env.grass_history[0:self.mean_duration]) /
                                       len(env.grass_history[0:self.mean_duration]))
            if hasattr(env, 'total_spin_episodes'):
                self.logger.record('total_spin_episodes', env.total_spin_episodes)
        return True
