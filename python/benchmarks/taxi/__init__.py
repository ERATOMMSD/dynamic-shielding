import sys
sys.path.append('../../')

from gym import register

register(
    id='TaxiFixStart-v3',
    entry_point='benchmarks.taxi.taxi_ext:TaxiFixStart',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=100,
)

register(
    id='TaxiStartCenter-v3',
    entry_point='benchmarks.taxi.taxi_ext:TaxiStartCenter',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)
