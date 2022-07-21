import os.path as osp
import sys
from typing import List

import gym
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from src.wrappers.crash_logger import CrashLoggingCallback
from src.wrappers.safe_padding_callbacks import SafePaddingCallback
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from benchmarks.car_racing.scripts.helper import get_default_parser
from benchmarks.car_racing.car_racing_logger import CarRacingLoggingCallback
from benchmarks.car_racing.car_racing_safe_padding import CarRacingSafePadding
from benchmarks.car_racing.scripts.car_racing_dynamic_shield import NO_CONSECUTIVE_GRASS_SPECIFICATIONS
from benchmarks.common.train import train

LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')
NUM_THREADS = 16


def car_racing_safe_padding(task_string: str, total_steps: int, num: int, lr: float):
    # string constants for this experiment setting
    algorithm = 'PPO'
    # configure the training environment
    root_path = f'{osp.dirname(__file__)}/..'
    label = f'{task_string}_safe_padding'
    specification: List[str] = NO_CONSECUTIVE_GRASS_SPECIFICATIONS
    env = Monitor(gym.make(f'{task_string}'))
    env = SafePaddingWrapper(env=env, safe_padding=CarRacingSafePadding(ltl_formula=specification))
    print(f'Training {algorithm}')

    # train the model
    callback_list = [SafePaddingCallback(), CrashLoggingCallback(mean_duration=10), CarRacingLoggingCallback()]
    train(env, label, task_string, callback_list, total_steps=total_steps, learning_rate=lr,
          policy='CnnPolicy', root_dir=root_path)


if __name__ == "__main__":
    parser = get_default_parser()

    args = parser.parse_args()
    car_racing_safe_padding(args.task_string, args.num_timesteps, args.num, args.lr)
