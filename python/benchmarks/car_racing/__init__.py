from gym.envs.registration import register

register(
    id='discrete_car_racing-v4',
    entry_point='benchmarks.car_racing.discrete_car_racing:DiscreteCarRacing',
    max_episode_steps=250,
    reward_threshold=900,
)
