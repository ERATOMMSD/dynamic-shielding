import sys
import argparse

from typing import Union, Type


sys.path.append('../../')
from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper, PostposedShieldWrapper
from benchmarks.water_tank.water_tank_specifications import safety_formula
from benchmarks.common.train import run
from water_tank_callbacks import WaterTankShieldingCallback
from benchmarks.water_tank.water_tank_preemptive_wrappers import WaterTankDynamicPreemptiveShieldWrapper, WaterTankSafePaddingWrapper, \
    WaterTankAdaptiveDynamicPreemptiveShieldWrapper
from benchmarks.water_tank.water_tank_postposed_wrappers import WaterTankDynamicPostposedShieldWrapper


def setup_parser() -> argparse.ArgumentParser:
    """
        Define the parser of the arguments
    """
    parser = argparse.ArgumentParser(description='evaluate dynamic shielding with water_tank benchmarks')
    parser.add_argument('--capacity', type=int, default=100,
                        help='the water tank can be initialized with capacity [20 | 50 | 100]')
    parser.add_argument('--steps', type=int, default=int(2e5),
                        help='Number of steps that each environment is run.')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--shield', type=str, default='pre-adaptive',
                        help='the shield to be used [pre-adaptive (default) | pre-dynamic | safe-padding | '
                             'post-dynamic | no]')
    parser.add_argument('--shield-life', type=int, default=10,
                        help='Frequency of shield reconstruction in terms of episodes.')
    parser.add_argument('--depths', nargs='+', default=[1, 3, 5, 7],
                        help='a list of min-depths for dynamic shield (usage: --depths 0 1 3)')
    parser.add_argument('--penalties', nargs='+', default=[1.0, 10.0, 100.0, 1000.0],
                        help='a list of penalties that it is used in no shield (usage: --penalties 0.0 1.0 100.0)')
    parser.add_argument('--pdb', type=bool,
                        help='If enabled, pdb will be launched when there is an exception.')

    return parser


def get_shield(shield) -> Union[None, Type[PreemptiveShieldWrapper],
                                Type[SafePaddingWrapper], Type[PostposedShieldWrapper]]:
    if shield == 'pre-dynamic':
        return WaterTankDynamicPreemptiveShieldWrapper
    elif shield == 'pre-adaptive':
        return WaterTankAdaptiveDynamicPreemptiveShieldWrapper
    elif shield == 'safe-padding':
        return WaterTankSafePaddingWrapper
    elif shield == 'post-dynamic':
        return WaterTankDynamicPostposedShieldWrapper
    elif shield == 'no':
        return None
    else:
        raise RuntimeError(f'Unknown shields {shield}')


# Currently available water tanks environments: c20i10, c50i25, c100i50 (See __init__.py)
def get_game(capacity) -> str:
    if capacity == 100:
        return f'WaterTank-c{100}-i{50}-v0'
    elif capacity == 50:
        return f'WaterTank-c{50}-i{25}-v0'
    elif capacity == 20:
        return f'WaterTank-c{20}-i{10}-v0'
    else:
        raise RuntimeError(f'Invalid capacity {capacity} for water tank.')


if __name__ == "__main__":

    parser = setup_parser()
    args = parser.parse_args()

    steps = args.steps
    game = get_game(args.capacity)
    shield = get_shield(args.shield)
    learning_rate = args.learning_rate
    shield_life = args.shield_life
    depths = list(map(int, args.depths))
    penalties = list(map(float, args.penalties))
    no_pdb = not args.pdb

    run(shield=shield,
        ltl_formula=safety_formula(),
        depths=depths,
        game=game,
        callback=WaterTankShieldingCallback,
        penalties=penalties,
        total_steps=steps,
        learning_rate=learning_rate,
        shield_life=shield_life)

    sys.exit(0)
