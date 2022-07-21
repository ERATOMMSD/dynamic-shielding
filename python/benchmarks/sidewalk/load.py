import os.path as osp
import pathlib
import sys

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.sidewalk.helper import EVAL_TASK_STRING

# configure the training environment
env = Monitor(gym.make(f'{EVAL_TASK_STRING}'))

# load the model
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print('Usage: load.py <path>')
    exit(0)
model = PPO.load(path)
record = False
if record:
    env = gym.wrappers.Monitor(env, pathlib.PurePath(path).parent)

# Enjoy trained agent
obs = env.reset()
for i in range(env.unwrapped.max_episode_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done and i + 1 < env.unwrapped.max_episode_steps:
        print(f'Completed in {i + 1} steps!!')
        print('completed' if rewards > 0 else 'crashed')
        break
env.reset()
