import os
import os.path as osp
import pathlib
import sys

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)

from benchmarks.car_racing.discrete_car_racing import evaluate_output
from benchmarks.car_racing.car_racing_specifications import no_consecutive_grass
from src.shields import StaticShield
from src.wrappers.shield_callbacks import SHIELD_PICKLE_NAME
from src.wrappers.shield_policy import shield_policy
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper
from benchmarks.car_racing.scripts.helper import TASK_STRING

experiment_string = 'PPO'
algorithm = 'PPO'
# configure the training environment
env = Monitor(gym.make(f'{TASK_STRING}'))

# load the model
# model = PPO.load('../model-unshielded.zip')
# model = PPO.load('../model.zip'
# model = PPO.load('../no_shield-1500000.zip')
# model = PPO.load('../logs/discrete_car_racing-v0_PPO_dynamic_shield-1000000-0/best_model/best_model')
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = osp.join(osp.dirname(__file__),
                    '..',
                    'logs/aloha/discrete_car_racing-v4/Nov23_07:12:11/discrete_car_racing-v4_adaptive_dynamic_shield_no_consecutive_use_deviating_shield/best_model/best_model.zip')
model = PPO.load(path)
record = False
if record:
    env = gym.wrappers.Monitor(env, pathlib.PurePath(path).parent)
# model = PPO.load('../logs/discrete_car_racing-v0_PPO_dynamic_shield-150000-0/model.zip')

# Use shield
shield_path = osp.join(pathlib.PurePosixPath(path).parent, SHIELD_PICKLE_NAME)
max_episode_steps = env.spec.max_episode_steps
if os.path.exists(shield_path):
    print('use shield')
    shield = StaticShield.create_from_pickle(ltl_formula=no_consecutive_grass,
                                             pickle_filename=shield_path,
                                             evaluate_output=evaluate_output)
    shield.reset()
    eval_env = PreemptiveShieldWrapper(env, shield)
    env = model._wrap_env(eval_env)
    model.env = env
    shield_policy(model, eval_env=eval_env, force_eval=True)
else:
    eval_env = env

# Enjoy trained agent
obs = eval_env.reset()
total_reward = 0
states = None
for i in range(max_episode_steps):
    obs = obs.copy()
    action, states = model.predict(obs, state=states, deterministic=True)
    obs, rewards, done, info = eval_env.step(action)
    total_reward += rewards
    eval_env.render()
    if done and i + 1 < max_episode_steps:
        print(f'Completed in {i + 1} steps!!')
        break
print(f'Total reward: {total_reward}')
eval_env.reset()
