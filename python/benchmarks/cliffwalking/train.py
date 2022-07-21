import sys
import argparse

sys.path.append('../../')
from typing import Union, Type

from benchmarks.common.train import run
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper, PostposedShieldWrapper
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from cliffwalking_specifications import dont_fall_formula
from cliffwalking_callbacks import CliffWalkingShieldingCallback
from cliffwalking_preemptive_wrappers import CliffWalkingDynamicPreemptiveShieldWrapper, \
    CliffWalkingAdaptiveDynamicPreemptiveShieldWrapper, CliffWalkingSafePaddingWrapper
from benchmarks.cliffwalking.cliffwalking_postposed_wrappers import CliffWalkingDynamicPostposedShieldWrapper


def setup_parser() -> argparse.ArgumentParser:
    """
        Define the parser of the arguments
    """
    parser = argparse.ArgumentParser(description='evaluate dynamic shielding with water_tank benchmarks')

    parser.add_argument('--steps', type=int, default=int(2e5),
                        help='number of steps that each environment is run.')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--shield', type=str, default='pre-adaptive',
                        help='the shield to be used [pre-adaptive (default) | pre-dynamic | safe-padding | '
                             'post-dynamic | no]')
    parser.add_argument('--shield-life', type=int, default=100,
                        help='frequency of shield reconstruction in terms of episodes.')
    parser.add_argument('--depths', nargs='+', default=[1, 3, 5, 7],
                        help='a list of min-depths for dynamic shield (usage: --depths 0 1 3)')
    parser.add_argument('--penalties', nargs='+', default=[0.0, 1.0, 10.0, 100.0],
                        help='a list of penalties that it is used in no shield (usage: --penalties 0.0 1.0 100.0)')
    return parser


def get_shield(shield) -> Union[None, Type[PreemptiveShieldWrapper],
                                Type[SafePaddingWrapper], Type[PostposedShieldWrapper]]:
    if shield == 'pre-dynamic':
        return CliffWalkingDynamicPreemptiveShieldWrapper
    elif shield == 'pre-adaptive':
        return CliffWalkingAdaptiveDynamicPreemptiveShieldWrapper
    elif shield == 'safe-padding':
        return CliffWalkingSafePaddingWrapper
    elif shield == 'post-dynamic':
        return CliffWalkingDynamicPostposedShieldWrapper
    elif shield == 'no':
        return None
    else:
        raise RuntimeError(f'Unknown shields {shield}')


if __name__ == "__main__":

    parser = setup_parser()
    args = parser.parse_args()

    steps = args.steps
    shield = get_shield(args.shield)
    learning_rate = args.learning_rate
    shield_life = args.shield_life
    depths = list(map(int, args.depths))
    penalties = list(map(float, args.penalties))

    run(shield=shield,
        ltl_formula=dont_fall_formula(),
        depths=depths,
        game='CliffWalkingExt-v0',
        callback=CliffWalkingShieldingCallback,
        penalties=penalties,
        total_steps=steps,
        learning_rate=learning_rate,
        shield_life=shield_life)

    sys.exit(0)
