import os.path as osp
import sys

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.grid_world.scripts.helper import TASK_STRING

# configure the training environment
env = Monitor(gym.make(f'{TASK_STRING}'))

# load the model
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print('Usage: load.py <path>')
    exit(0)
model = PPO.load(path)

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(env.unwrapped.MAX_STEP):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones and i + 1 < env.unwrapped.MAX_STEP:
        print(f'Completed in {i + 1} steps!!')
        print('completed' if rewards > 0 else 'crashed')
        break
env.reset()
