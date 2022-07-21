import sys
import os.path as osp
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../')
sys.path.append(PROJECT_ROOT)
from taxi_ext import TaxiFixStart, TaxiStartCenter

# Constants
experiment_string = 'PPO'
algorithm = 'PPO'

if len(sys.argv) <= 1:
    print('Usage: load.py [model path]')
    exit(1)

total_success = 0

# configure the training environment
env = Monitor(TaxiStartCenter())

for i in range(1, len(sys.argv)):
    # load the model
    path = sys.argv[i]
    model = PPO.load(path)

    # Enjoy trained agent
    obs = env.reset()
    if env.spec is not None and env.spec.max_episode_steps is not None:
        max_step = env.spec.max_episode_steps
    elif hasattr(env.unwrapped, 'MAX_STEP') and  env.unwrapped.MAX_STEP is not None:
        max_step = env.unwrapped.MAX_STEP
    elif hasattr(env.unwrapped, 'MAX_STEPS') and env.unwrapped.MAX_STEPS is not None:
        max_step = env.unwrapped.MAX_STEPS
    else:
        raise RuntimeError()
    for j in range(max_step):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break
    env.reset()

print(f'total_trial: {len(sys.argv) - 1}')
print(f'total_success: {env.unwrapped.delivered}')
