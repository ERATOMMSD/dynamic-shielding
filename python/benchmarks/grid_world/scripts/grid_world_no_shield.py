import os.path as osp
import sys

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
from benchmarks.grid_world.grid_world_crash_logger import GridWorldCrashLogger
from src.wrappers.crash_logger import CrashLoggingCallback

PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.grid_world.scripts.helper import configure_logger, get_default_parser

LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')
NUM_THREADS = 16


def grid_world_no_shield(task_string: str, num_timesteps: int, num: int, lr: float, gamma: float,
                     tb_log_name='tensorboard.log', vector=False):
    # string constants for this experiment setting
    experiment_string = 'PPO_no_shield'
    algorithm = 'PPO'
    # configure the training environment
    log_dir = osp.join(LOG_ROOT, f'{task_string}_{experiment_string}-{num_timesteps}-{num}/')
    configure_logger(log_dir)
    if vector:
        env = make_vec_env(f'{task_string}', n_envs=NUM_THREADS)
    else:
        env = Monitor(GridWorldCrashLogger(gym.make(f'{task_string}')))
    print(f'Training {algorithm}')

    # train the model
    model = PPO('MlpPolicy', env, verbose=1, gamma=gamma, learning_rate=lr, tensorboard_log=log_dir)
    eval_callback = EvalCallback(eval_env=model.get_env(), best_model_save_path=osp.join(log_dir, 'best_model'),
                                 n_eval_episodes=5,
                                 eval_freq=50000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=num_timesteps, log_interval=1, tb_log_name=tb_log_name,
                callback=[CrashLoggingCallback(), eval_callback])
    # Save the trained model
    model.save(osp.join(log_dir, 'final_model'))


if __name__ == "__main__":
    parser = get_default_parser()

    args = parser.parse_args()
    grid_world_no_shield(args.task_string, args.num_timesteps, args.num, args.lr, args.gamma)
