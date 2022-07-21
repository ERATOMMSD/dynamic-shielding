import sys
sys.path.append('../../')

from gym.envs.registration import register

register(
    id='CliffWalkingExt-v0',
    max_episode_steps=100,
    entry_point='benchmarks.cliffwalking.cliffwalking_env:CliffWalkingExt'
)

