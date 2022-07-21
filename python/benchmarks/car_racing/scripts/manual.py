import os.path as osp
import sys

import gym
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.car_racing.scripts.helper import TASK_STRING
from benchmarks.car_racing.discrete_car_racing import AgentActions

experiment_string = 'PPO'
algorithm = 'PPO'
# configure the training environment
env = Monitor(gym.make(f'{TASK_STRING}'))

actions = [AgentActions.NONE]
actions = actions * 1000
actions = [AgentActions.ACCEL] * 50 + [AgentActions.NONE] * 30 + [AgentActions.LEFT] * 15 + \
          [AgentActions.ACCEL] * 90 + [AgentActions.BRAKE] * 10 + [AgentActions.RIGHT] * 5 + [AgentActions.NONE] * 35 + \
          [AgentActions.LEFT] * 10 + [AgentActions.NONE] * 75 + [AgentActions.LEFT] * 10 + [AgentActions.NONE] * 50 + \
          [AgentActions.LEFT] * 15 + actions

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    obs, rewards, done, info = env.step(actions[i])
    env.render()
    if done:
        env.reset()
        break
