"""
This is a script to evaluate dynamic shielding with side_walk benchmarks

Usage: python train.py [OPTIONS]
"""
import os.path as osp
import sys
from enum import Flag, auto

import gym
from stable_baselines3.common.monitor import Monitor

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../')
sys.path.append(PROJECT_ROOT)
from src.shields import UpdateShield
from src.shields.adaptive_dynamic_shield import AdaptiveDynamicShield
from src.shields.evaluation_shield import EvaluationShield
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper
from src.wrappers.crash_logger import CrashLoggingCallback
from src.wrappers.shield_callbacks import ShieldingCallback
from src.wrappers.safe_padding_callbacks import SafePaddingCallback
from benchmarks.common.train import train
from benchmarks.common.generic import launch_gateway
from benchmarks.sidewalk.sidewalk_wrappers import SidewalkPreDynamicShieldWrapper, SidewalkSafePaddingWrapper
from benchmarks.sidewalk.sidewalk_dynamic_shield import SidewalkDynamicShield
from benchmarks.sidewalk.sidewalk_crash_logger import SideWalkCrashLogger
from benchmarks.sidewalk.sidewalk_gym_wrapper import SidewalkWrapper, evaluate_output
from benchmarks.sidewalk.helper import get_default_parser, TASK_STRING, NUM_TIMESTEP, LEARNING_RATE, MIN_DEPTH, \
    EVAL_TASK_STRING, MAX_SHIELD_LIFE, FACTOR
from benchmarks.sidewalk.sidewalk_gradual_reward import SidewalkGradualReward
from benchmarks.sidewalk.sidewalk_specifications import SPECIFICATIONS


class Shields(Flag):
    NoShield = auto()
    DynamicShield = auto()
    SafePadding = auto()
    AdaptiveDynamic = auto()


def get_shields(args) -> Shields:
    if args.shield == 'all':
        return Shields.AdaptiveDynamic | Shields.NoShield | Shields.SafePadding
    elif args.shield == 'dynamic':
        return Shields.DynamicShield
    elif args.shield == 'no':
        return Shields.NoShield
    elif args.shield == 'safe_padding':
        return Shields.SafePadding
    elif args.shield == 'adaptive':
        return Shields.AdaptiveDynamic
    else:
        raise RuntimeError(f'Unknown shields {args.shield}')


def run(shield: Shields):
    env = SideWalkCrashLogger(SidewalkWrapper(gym.make(f'{TASK_STRING}')))
    eval_env = SidewalkWrapper(gym.make(f'{EVAL_TASK_STRING}'))
    num_grades: int = env.sidewalk_length // 5
    env = Monitor(SidewalkGradualReward(env, num_grades=num_grades))
    eval_env = Monitor(SidewalkGradualReward(eval_env, num_grades=num_grades))
    root_path = osp.dirname(__file__)
    gateway = launch_gateway()
    if shield == Shields.NoShield:
        shield_string = 'no'
        label = f'{TASK_STRING}_{shield_string}_shield'
        callback = [CrashLoggingCallback()]
        train(env=env,
              label=label,
              game=TASK_STRING,
              total_steps=NUM_TIMESTEP,
              learning_rate=LEARNING_RATE,
              callback=callback,
              policy='CnnPolicy',
              root_dir=root_path)
    elif shield == Shields.SafePadding:
        shield_string = 'safe_padding'
        label = f'{TASK_STRING}_{shield_string}'
        env = SidewalkSafePaddingWrapper(env=env)
        callback = [SafePaddingCallback(), CrashLoggingCallback()]
        train(env=env,
              label=label,
              game=TASK_STRING,
              total_steps=NUM_TIMESTEP,
              learning_rate=LEARNING_RATE,
              callback=callback,
              policy='CnnPolicy',
              root_dir=root_path)
    elif shield == Shields.AdaptiveDynamic:
        shield_string = 'adaptive_dynamic'
        label = f'{TASK_STRING}_{shield_string}_shield'
        if hasattr(env, 'num_seeds'):
            num_seeds = env.num_seeds
        else:
            num_seeds = 1
        shield = AdaptiveDynamicShield(ltl_formula=SPECIFICATIONS, gateway=gateway,
                                       alphabet_start=0, alphabet_end=4 * num_seeds,
                                       max_episode_length=env.unwrapped.max_episode_steps,
                                       alphabet_mapper=SidewalkDynamicShield.alphabet_mapper,
                                       evaluate_output=evaluate_output,
                                       reverse_alphabet_mapper=SidewalkDynamicShield.reverse_alphabet_mapper,
                                       reverse_output_mapper=SidewalkDynamicShield.reverse_output_mapper,
                                       update_shield=UpdateShield.RESET,
                                       concurrent_reconstruction=True,
                                       max_shield_life=MAX_SHIELD_LIFE,
                                       not_use_deviating_shield=False,
                                       factor=FACTOR)
        callback = [ShieldingCallback(), CrashLoggingCallback()]
        env = PreemptiveShieldWrapper(env=env, shield=shield)
        eval_shield = EvaluationShield(training_env=env)
        eval_env = PreemptiveShieldWrapper(env=eval_env, shield=eval_shield)
        train(env=env,
              label=label,
              game=TASK_STRING,
              total_steps=NUM_TIMESTEP,
              learning_rate=LEARNING_RATE,
              callback=callback,
              policy='CnnPolicy',
              root_dir=root_path,
              eval_env=eval_env)
    else:  # shield == Shields.DynamicShield
        shield_string = 'dynamic'
        label = f'{TASK_STRING}_{shield_string}_shield-depth{MIN_DEPTH}'
        env = SidewalkPreDynamicShieldWrapper(env=env, gateway=gateway, min_depth=MIN_DEPTH,
                                              max_shield_life=MAX_SHIELD_LIFE)
        callback = [ShieldingCallback(), CrashLoggingCallback()]
        eval_shield = EvaluationShield(training_env=env)
        eval_env = PreemptiveShieldWrapper(env=eval_env, shield=eval_shield)
        train(env=env,
              label=label,
              game=TASK_STRING,
              total_steps=NUM_TIMESTEP,
              learning_rate=LEARNING_RATE,
              callback=callback,
              policy='CnnPolicy',
              root_dir=root_path,
              eval_env=eval_env)
    gateway.close()


if __name__ == "__main__":
    parser = get_default_parser()

    args = parser.parse_args()
    shields = get_shields(args)
    for shield in Shields:
        if shield in shields:
            run(shield)
