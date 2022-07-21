"""
This is a script to evaluate dynamic shielding with grid_world benchmarks

Usage: python run.py [OPTIONS]

Parameters:
    benchmarks: all (default), grid_world2, grid_world3
    shield: all (default), static, dynamic, no
"""
import argparse
import itertools
import os.path as osp
import sys
from enum import Flag, auto
from typing import Dict, Tuple

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)

from benchmarks.grid_world.scripts.grid_world_runner import Algorithms, GridWorldRunner, \
    DynamicShieldGridWorldRunner, ShieldKind, SafePaddingGridWorldRunner, AdaptiveDynamicShieldGridWorldRunner
from benchmarks.grid_world.scripts.helper import MAX_SHIELD_LIFE


class Benchmarks(Flag):
    GridWorld2 = auto()

class Shields(Flag):
    DynamicShield = auto()
    NoShield = auto()
    HugeNegativeReward = auto()
    DynamicShieldWithHugeNegativeReward = auto()
    SafePadding = auto()
    AdaptiveDynamic = auto()


# Constants
NUM_TIMESTEPS: Dict[Benchmarks, int] = {Benchmarks.GridWorld2: 100 * 1000}
LEARNING_RATES: Dict[Tuple[Benchmarks, Algorithms], float] = {(Benchmarks.GridWorld2, Algorithms.PPO): 3e-4}
GAMMAS: Dict[Benchmarks, float] = {Benchmarks.GridWorld2: 0.99}
TASK_STRINGS: Dict[Benchmarks, str] = {Benchmarks.GridWorld2: 'grid_world2-v1'}
MIN_DEPTH: Dict[Benchmarks, int] = {Benchmarks.GridWorld2: 5}


def setup_parser() -> argparse.ArgumentParser:
    """
        Define the parser of the arguments
    """
    parser = argparse.ArgumentParser(description='evaluate dynamic shielding with grid_world benchmarks')
    parser.add_argument('--benchmarks', type=str, default='grid_world2',
                        help='the benchmarks to use [all | grid_world2 (default) ]')
    parser.add_argument('--shield', type=str, default='all',
                        help='the shields to use [all (default) | dynamic | no | negative | negative_dynamic | safe_padding | adaptive]')
    parser.add_argument('--shield_kind', type=str, default='preemptive',
                        help='the shields to use [all | preemptive (default) | postposed]')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        help='the RL algorithm to use [all | PPO (default) | DQN]')
    return parser


def get_benchmarks(args) -> Benchmarks:
    if args.benchmarks == 'all':
        return Benchmarks.GridWorld2 | Benchmarks.GridWorld3
    elif args.benchmarks == 'grid_world2':
        return Benchmarks.GridWorld2
    elif args.benchmarks == 'grid_world2_2':
        return Benchmarks.GridWorld2_2
    elif args.benchmarks == 'grid_world3':
        return Benchmarks.GridWorld3
    else:
        raise RuntimeError(f'Unknown benchmarks {args.benchmarks}')


def get_shields(args) -> Shields:
    if args.shield == 'all':
        return Shields.AdaptiveDynamic | Shields.NoShield | Shields.SafePadding
    elif args.shield == 'dynamic':
        return Shields.DynamicShield
    elif args.shield == 'no':
        return Shields.NoShield
    elif args.shield == 'negative':
        return Shields.HugeNegativeReward
    elif args.shield == 'negative_dynamic':
        return Shields.DynamicShieldWithHugeNegativeReward
    elif args.shield == 'safe_padding':
        return Shields.SafePadding
    elif args.shield == 'adaptive':
        return Shields.AdaptiveDynamic
    else:
        raise RuntimeError(f'Unknown shields {args.shield}')


def get_shield_kind(args) -> ShieldKind:
    if args.shield_kind == 'all':
        return ShieldKind.Preemptive | ShieldKind.Postposed
    elif args.shield_kind == 'preemptive':
        return ShieldKind.Preemptive
    elif args.shield_kind == 'postposed':
        return ShieldKind.Postposed
    else:
        raise RuntimeError(f'Unknown shield kinds {args.shield_kind}')


def get_algorithm(args) -> Algorithms:
    if args.algorithm == 'all':
        return Algorithms.DQN | Algorithms.PPO
    elif args.algorithm == 'PPO':
        return Algorithms.PPO
    elif args.algorithm == 'DQN':
        return Algorithms.DQN
    else:
        raise RuntimeError(f'Unknown algorithm {args.algorithm}')


def run(benchmark: Benchmarks, shield: Shields, algorithm: Algorithms, shield_kind: ShieldKind):
    task_string: str = TASK_STRINGS[benchmark]
    if shield == Shields.NoShield:
        runner = GridWorldRunner(task_string, algorithm)
        shield_string = 'no'
    elif shield == Shields.HugeNegativeReward:
        task_string = task_string.replace('-', '-huge_penalty2-')
        runner = GridWorldRunner(task_string, algorithm)
        shield_string = 'no'
    elif shield == Shields.DynamicShieldWithHugeNegativeReward:
        task_string = task_string.replace('-', '-huge_penalty2-')
        runner = DynamicShieldGridWorldRunner(task_string, algorithm, shield_kind, MIN_DEPTH[benchmark],
                                              skip_mealy_size=0,
                                              max_shield_life=MAX_SHIELD_LIFE, concurrent_reconstruction=True)
        shield_string = 'dynamic'
    elif shield == Shields.SafePadding:
        runner = SafePaddingGridWorldRunner(task_string, algorithm, shield_kind)
        shield_string = 'safe_padding'
    elif shield == Shields.DynamicShield:
        runner = DynamicShieldGridWorldRunner(task_string, algorithm, shield_kind, MIN_DEPTH[benchmark],
                                              skip_mealy_size=0,  # save_full_pickle_filename='training_data.pickle',
                                              max_shield_life=MAX_SHIELD_LIFE, concurrent_reconstruction=True)
        shield_string = 'dynamic'
    elif shield == Shields.AdaptiveDynamic:
        runner = AdaptiveDynamicShieldGridWorldRunner(task_string, algorithm, shield_kind, skip_mealy_size=0,
                                                      max_shield_life=MAX_SHIELD_LIFE, concurrent_reconstruction=True)
        shield_string = 'adaptive_dynamic'
    else:
        raise RuntimeError(f'Unknown shield kind {shield}')
    runner.train(NUM_TIMESTEPS[benchmark], num=0, lr=LEARNING_RATES[benchmark, algorithm], gamma=GAMMAS[benchmark],
                 tb_log_name=f'{task_string}_{shield_string}_shield_{str(algorithm)}')


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    benchmarks = get_benchmarks(args)
    shields = get_shields(args)
    shield_kinds = get_shield_kind(args)
    algorithms = get_algorithm(args)
    for benchmark, shield, algorithm, shield_kind in itertools.product(Benchmarks, Shields, Algorithms, ShieldKind):
        if benchmark in benchmarks and shield in shields and algorithm in algorithms and shield_kind in shield_kinds:
            run(benchmark, shield, algorithm, shield_kind)
