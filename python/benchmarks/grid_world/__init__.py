from gym.envs.registration import register

register(
    id='grid_world-v2',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnv'
)

register(
    id='grid_world_ignore_order-v2',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvIgnoreOrder'
)

register(
    id='grid_world_one_color-v1',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvOneColor'
)
register(
    id='grid_world_red-v1',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvRed'
)
register(
    id='grid_world_yellow-v1',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvYellow'
)
register(
    id='grid_world_green-v1',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvGreen'
)
register(
    id='grid_world_blue-v1',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvBlue'
)
register(
    id='grid_world_survive_long-v1',
    entry_point='benchmarks.grid_world.grid_world_env:GridWorldEnvSurviveLong'
)
register(
    id='grid_world2-v1',
    entry_point='benchmarks.grid_world.deterministic_grid_world_env:GridWorldEnv2'
)

register(
    id='grid_world2-huge_penalty2-v1',
    entry_point='benchmarks.grid_world.deterministic_grid_world_env:GridWorldEnv2',
    kwargs={'huge_penalty': True}
)

register(
    id='grid_world2-v2',
    entry_point='benchmarks.grid_world.deterministic_grid_world_env:GridWorldEnv2',
    kwargs={'goal_x': 0, 'goal_y': 4}
)

register(
    id='grid_world3-v2',
    entry_point='benchmarks.grid_world.deterministic_grid_world_env:GridWorldEnv3'
)
