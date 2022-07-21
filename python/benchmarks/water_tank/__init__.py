from gym.envs.registration import register

register(
    id='WaterTank-c100-i50-v0',
    max_episode_steps=1000,
    entry_point='benchmarks.water_tank.water_tank_env:WaterTankEnvC100I50'
)

register(
    id='WaterTank-c50-i25-v0',
    max_episode_steps=500,
    entry_point='benchmarks.water_tank.water_tank_env:WaterTankEnvC50I25'
)

register(
    id='WaterTank-c20-i10-v0',
    max_episode_steps=200,
    entry_point='benchmarks.water_tank.water_tank_env:WaterTankEnvC20I10'
)

