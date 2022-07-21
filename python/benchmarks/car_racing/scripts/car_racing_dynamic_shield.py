import os.path as osp
import sys
from typing import List, Optional

import gym
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)

from benchmarks.common.generic import launch_gateway
from benchmarks.car_racing.car_racing_specifications import no_grass, no_grass_duration, \
    no_consecutive_grass_duration, no_consecutive_grass
from benchmarks.car_racing.huge_negative_wrapper import HugeNegativeWrapper
from benchmarks.car_racing.scripts.helper import get_default_parser, FACTOR
from benchmarks.common.train import train
from src.shields.evaluation_shield import EvaluationShield
from src.wrappers.crash_logger import CrashLoggingCallback
from src.wrappers.shield_callbacks import ShieldingCallback
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper
from benchmarks.car_racing.car_racing_dynamic_shield import CarRacingDynamicShield, CarRacingAdaptiveDynamicShield
from benchmarks.car_racing.car_racing_logger import CarRacingLoggingCallback

LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')

NO_GRASS_SPECIFICATIONS = [no_grass, no_grass_duration(5), no_grass_duration(3), no_grass_duration(1)]
NO_CONSECUTIVE_GRASS_SPECIFICATIONS = [no_consecutive_grass, no_consecutive_grass_duration(5),
                                       no_consecutive_grass_duration(3), no_consecutive_grass_duration(1)]


def car_racing_dynamic_shield(task_string: str, total_steps: int, learning_rate: float, min_depth: Optional[int],
                              max_shield_life: int = 100, not_use_deviating_shield: bool = False,
                              strict_specification: bool = False, huge_negative_reward=False):
    # string constants for this experiment setting
    if min_depth is None:
        shield_string = 'adaptive_dynamic'
    else:
        shield_string = 'dynamic'
    label = f'{task_string}_{shield_string}_shield'
    if not strict_specification:
        label += '_no_consecutive'
    if not not_use_deviating_shield:
        label += '_use_deviating_shield'
    algorithm = 'PPO'
    root_path = f'{osp.dirname(__file__)}/..'
    # Set up py4j gateway
    gateway = launch_gateway()
    if strict_specification:
        specification: List[str] = NO_GRASS_SPECIFICATIONS
    else:
        specification: List[str] = NO_CONSECUTIVE_GRASS_SPECIFICATIONS

    env = Monitor(gym.make(f'{task_string}'))
    eval_env = Monitor(gym.make(f'{task_string}'))
    if huge_negative_reward:
        env = HugeNegativeWrapper(env)
        eval_env = HugeNegativeWrapper(eval_env)
    if min_depth is not None:
        # dynamic shield with static min_depth
        shield = CarRacingDynamicShield(specification, gateway, min_depth=min_depth,
                                        not_use_deviating_shield=not_use_deviating_shield, use_sensor=True,
                                        max_shield_life=max_shield_life)
    else:
        # dynamic shield with adaptive min_depth
        shield = CarRacingAdaptiveDynamicShield(specification, gateway, max_episode_length=250, factor=FACTOR,
                                                not_use_deviating_shield=not_use_deviating_shield, use_sensor=True,
                                                max_shield_life=max_shield_life)
    env = PreemptiveShieldWrapper(env=env, shield=shield)
    eval_shield = EvaluationShield(training_env=env)
    eval_env = PreemptiveShieldWrapper(env=eval_env, shield=eval_shield)
    print(f'Training {algorithm}')
    callback_list = [ShieldingCallback(), CrashLoggingCallback(mean_duration=10), CarRacingLoggingCallback()]
    train(env=env,
          label=label,
          game=task_string,
          callback=callback_list,
          total_steps=total_steps,
          learning_rate=learning_rate,
          policy='CnnPolicy',
          eval_env=eval_env,
          root_dir=root_path)
    gateway.close()


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument('--min_depth', type=int, default=None,
                        help='the minimum depth to be merged in the strong RPNI-style algorithm')
    parser.add_argument('--max_shield_life', type=int, default=100,
                        help='maximum episodes to refresh the learned shield')
    args = parser.parse_args()
    car_racing_dynamic_shield(args.task_string, args.total_steps, args.lr, args.min_depth,
                              args.max_shield_life)
