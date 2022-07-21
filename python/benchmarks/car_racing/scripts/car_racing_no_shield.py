import os.path as osp
import sys

import gym
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from src.wrappers.crash_logger import CrashLoggingCallback
from benchmarks.car_racing.scripts.helper import get_default_parser
from benchmarks.car_racing.car_racing_logger import CarRacingLoggingCallback
from benchmarks.car_racing.huge_negative_wrapper import HugeNegativeWrapper
from benchmarks.common.train import train

LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')
NUM_THREADS = 16


def car_racing_no_shield(task_string: str, total_steps: int, num: int, lr: float, huge_negative_reward=False):
    # string constants for this experiment setting
    label = f'{task_string}_no_shield'
    algorithm = 'PPO'
    # configure the training environment
    root_path = f'{osp.dirname(__file__)}/..'
    env = Monitor(gym.make(f'{task_string}'))
    print(f'Training {algorithm}')
    if huge_negative_reward:
        env = HugeNegativeWrapper(env)

    # train the model
    callback_list = [CrashLoggingCallback(mean_duration=10), CarRacingLoggingCallback()]
    train(env, label, task_string, callback_list, total_steps=total_steps, learning_rate=lr,
          policy='CnnPolicy', root_dir=root_path)


if __name__ == "__main__":
    parser = get_default_parser()

    args = parser.parse_args()
    car_racing_no_shield(args.task_string, args.num_timesteps, args.num, args.lr)
