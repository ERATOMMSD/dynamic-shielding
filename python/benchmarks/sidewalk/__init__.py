from gym.envs.registration import register

register(
    id='deterministic_side_walk-v0',
    entry_point='benchmarks.sidewalk.deterministic_side_walk:DeterministicSideWalk',
    kwargs={'default_seed': 1}
)

register(
    id='deterministic_sidewalk-v1',
    entry_point='benchmarks.sidewalk.deterministic_side_walk:DeterministicSideWalk',
    kwargs={'default_seed': 1, 'sidewalk_length': 20, 'max_episode_steps': 185}
)

register(
    id='RangedDeterministicSidewalk-DomainRand-v6',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 3, 'domain_rand': True}
)

register(
    id='ranged_deterministic_sidewalk-v0',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 5}
)

register(
    id='RangedDeterministicSidewalk-DomainRand-v0',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 5, 'domain_rand': True}
)

register(
    id='ranged_deterministic_sidewalk-v1',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 10, 'domain_rand': True}
)
register(
    id='ranged_deterministic_sidewalk-v2',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 50, 'domain_rand': True}
)

register(
    id='ranged_deterministic_sidewalk-v5',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 75, 'domain_rand': True}
)

register(
    id='ranged_deterministic_sidewalk-v4',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 100, 'domain_rand': True}
)

register(
    id='ranged_deterministic_sidewalk-v3',
    entry_point='benchmarks.sidewalk.ranged_deterministic_sidewalk:RangedDeterministicSidewalk',
    kwargs={'num_seeds': 5, 'sidewalk_length': 20}
)

register(
    id='sidewalk-v1',
    entry_point='gym_miniworld.envs.sidewalk:Sidewalk',
    kwargs={'domain_rand': True}
)
