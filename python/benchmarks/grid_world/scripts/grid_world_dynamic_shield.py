import os.path as osp
import sys

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.common.generic import launch_gateway
from src.wrappers.shield_callbacks import ShieldingCallback
from benchmarks.grid_world.grid_world_crash_logger import GridWorldCrashLogger
from benchmarks.grid_world.grid_world_wrappers import GridWorldPostDynamicShieldWrapper
from src.wrappers.crash_logger import CrashLoggingCallback
from benchmarks.grid_world.scripts.helper import configure_logger, get_default_parser

LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')
NUM_THREADS = 16


def grid_world_dynamic_shield(task_string: str, num_timesteps: int, num: int, lr: float, gamma: float,
                              min_depth: int, tb_log_name='tensorboard.log', concurrent_reconstruction=True,
                              max_shield_life: int = 100):
    # string constants for this experiment setting
    experiment_string = 'PPO_dynamic_shielding'
    algorithm = 'PPO'
    # configure the training environment
    log_dir = osp.join(LOG_ROOT, f'{task_string}_{experiment_string}-{num_timesteps}-{num}/')
    configure_logger(log_dir)
    env = Monitor(GridWorldCrashLogger(gym.make(f'{task_string}')))
    print(f'Training {algorithm}')
    # Set up py4j gateway
    gateway = launch_gateway()
    shielded_env = GridWorldPostDynamicShieldWrapper(env, gateway, min_depth=min_depth,
                                                     concurrent_reconstruction=concurrent_reconstruction,
                                                     max_shield_life=max_shield_life)

    # train the model
    model = PPO('MlpPolicy', shielded_env, verbose=1, gamma=gamma, learning_rate=lr, tensorboard_log=log_dir)
    eval_callback = EvalCallback(eval_env=model.get_env(), best_model_save_path=osp.join(log_dir, 'best_model'),
                                 n_eval_episodes=5,
                                 eval_freq=50000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=num_timesteps, log_interval=1, tb_log_name=tb_log_name,
                callback=[CrashLoggingCallback(), eval_callback, ShieldingCallback()])
    # Save the trained model
    model.save(osp.join(log_dir, 'final_model'))
    gateway.close()


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument('--min_depth', type=int, default=0,
                        help='the minimum depth to be merged in the strong RPNI-style algorithm')
    parser.add_argument('--concurrent_reconstruction', type=bool, default=True,
                        help='reconstruct shielding concurrently')
    parser.add_argument('--max_shield_life', type=int, default=100,
                        help='maximum episodes to refresh the learned shield')

    args = parser.parse_args()
    grid_world_dynamic_shield(args.task_string, args.total_steps, args.num, args.lr, args.gamma,
                              min_depth=args.min_depth, concurrent_reconstruction=args.concurrent_reconstruction,
                              max_shield_life=args.max_shield_life)
