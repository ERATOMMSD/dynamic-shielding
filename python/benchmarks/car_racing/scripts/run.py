"""
This is a script to evaluate dynamic shielding with car-racing

Usage: python run.py [OPTIONS]

Parameters:
    shield: all (default), dynamic, no
"""
import argparse
import os.path as osp
import sys
from enum import Flag, auto

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
from benchmarks.car_racing.scripts.car_racing_safe_padding import car_racing_safe_padding
from benchmarks.car_racing.scripts.car_racing_dynamic_shield import car_racing_dynamic_shield
from benchmarks.car_racing.scripts.car_racing_no_shield import car_racing_no_shield
from benchmarks.car_racing.scripts.helper import TASK_STRING, LEARNING_RATE, MAX_SHIELD_LIFE, TOTAL_STEPS


class Shields(Flag):
    DynamicShield = auto()
    NoShield = auto()
    HugeNegativeReward = auto()
    DynamicShieldHugeNegativeReward = auto()
    SafePadding = auto()
    AdaptiveDynamic = auto()


def setup_parser() -> argparse.ArgumentParser:
    """
        Define the parser of the arguments
    """
    parser = argparse.ArgumentParser(description='evaluate dynamic shielding with car_racing benchmark')
    parser.add_argument('--shield', type=str, default='all',
                        help='the shields to use [all (default) | dynamic | no | negative | negative_dynamic]')
    return parser


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
        return Shields.DynamicShieldHugeNegativeReward
    elif args.shield == 'safe_padding':
        return Shields.SafePadding
    elif args.shield == 'adaptive':
        return Shields.AdaptiveDynamic
    else:
        raise RuntimeError(f'Unknown shields {args.shield}')


def run(shield: Shields):
    if shield == Shields.NoShield:
        car_racing_no_shield(TASK_STRING, TOTAL_STEPS, num=0, lr=LEARNING_RATE,
                             huge_negative_reward=False)
    elif shield == Shields.DynamicShield:
        car_racing_dynamic_shield(TASK_STRING, TOTAL_STEPS, learning_rate=LEARNING_RATE, min_depth=0,
                                  max_shield_life=MAX_SHIELD_LIFE, not_use_deviating_shield=False,
                                  strict_specification=False)
    elif shield == Shields.AdaptiveDynamic:
        car_racing_dynamic_shield(TASK_STRING, TOTAL_STEPS, learning_rate=LEARNING_RATE, min_depth=None,
                                  max_shield_life=MAX_SHIELD_LIFE, not_use_deviating_shield=False,
                                  strict_specification=False)
    elif shield == Shields.DynamicShieldHugeNegativeReward:
        car_racing_dynamic_shield(TASK_STRING, TOTAL_STEPS, learning_rate=LEARNING_RATE, min_depth=0,
                                  max_shield_life=MAX_SHIELD_LIFE, not_use_deviating_shield=False,
                                  strict_specification=False, huge_negative_reward=True)
    elif shield == Shields.SafePadding:
        car_racing_safe_padding(TASK_STRING, TOTAL_STEPS, num=0, lr=LEARNING_RATE)
    elif shield == Shields.HugeNegativeReward:
        car_racing_no_shield(TASK_STRING, TOTAL_STEPS, num=0, lr=LEARNING_RATE,
                             huge_negative_reward=True)
    else:
        raise RuntimeError(f'Unknown shields {shield}')


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    shields = get_shields(args)
    for shield in Shields:
        if shield in shields:
            run(shield)
